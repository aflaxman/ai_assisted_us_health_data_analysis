"""Build 01_load_and_merge.ipynb: load NHANES SMQ + DEMO across cycles
2007-2018, harmonize, write a combined parquet to data/derived/.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def md(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = []


CELLS.append(md(
    """\
# 01 — NHANES smoking + demographics: load and merge across cycles

Loads NHANES smoking-questionnaire (`SMQ_*.xpt`) and demographics
(`DEMO_*.xpt`) files for cycles 2007-2018, harmonizes the smoking-
status definition (current / former / never), and writes a combined
parquet to `../data/derived/nhanes_smoking.parquet` for the next
notebook to consume.

The smoking status follows the standard CDC definition built from two
SMQ items:

| SMQ020 (ever 100+ cig) | SMQ040 (now smoke)            | Status   |
| ---------------------- | ----------------------------- | -------- |
| Yes (1)                | Every day or some days (1, 2) | Current  |
| Yes (1)                | Not at all (3)                | Former   |
| No (2)                 | —                             | Never    |
| Refused / DK (7, 9)    | —                             | (drop)   |

We use the 2-year interview weight (`WTINT2YR`) divided by the
number of cycles for the combined survey weight. Adults 20+ only.
""",
    "intro",
))


CELLS.append(code(
    """\
import os, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(os.path.abspath(os.path.join("..", "data")))
RAW = DATA_DIR / "raw" / "nhanes"
DERIVED = DATA_DIR / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)


CYCLES = [
    ("2007_2008", "E"),
    ("2009_2010", "F"),
    ("2011_2012", "G"),
    ("2013_2014", "H"),
    ("2015_2016", "I"),
    ("2017_2018", "J"),
]
N_CYCLES = len(CYCLES)


def smoking_status(row: pd.Series) -> str | None:
    \"\"\"CDC 3-state status from SMQ020 / SMQ040. Refused / DK rows
    return ``None`` and are dropped downstream.\"\"\"
    if row["SMQ020"] == 2:
        return "never"
    if row["SMQ020"] == 1:
        if row["SMQ040"] in (1, 2):
            return "current"
        if row["SMQ040"] == 3:
            return "former"
    return None
""",
    "imports",
))


CELLS.append(code(
    """\
frames = []
for cycle_dir, suffix in CYCLES:
    demo = pd.read_sas(RAW / cycle_dir / f"DEMO_{suffix}.xpt")
    smq = pd.read_sas(RAW / cycle_dir / f"SMQ_{suffix}.xpt")
    df = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "WTINT2YR"]].merge(
        smq[["SEQN", "SMQ020", "SMQ040"]], on="SEQN", how="left"
    )
    df["cycle"] = cycle_dir
    frames.append(df)
combined = pd.concat(frames, ignore_index=True)
print(f"merged rows across {N_CYCLES} cycles: {len(combined):,}")

# Adults 20+ with valid smoking response and positive interview weight.
adult = combined[combined["RIDAGEYR"] >= 20].copy()
adult["smoking"] = adult.apply(smoking_status, axis=1)
adult["sex"] = adult["RIAGENDR"].map({1: "Male", 2: "Female"})
adult = adult.dropna(subset=["smoking", "WTINT2YR"])
adult = adult[adult["WTINT2YR"] > 0]

# Combined weight = per-cycle interview weight / number of cycles.
adult["weight"] = adult["WTINT2YR"] / N_CYCLES

# 5-year age band; cap at 90+ for trial-relevant tail (NHANES
# topcodes age at 80, but we keep numeric ages in their bin and
# downstream analyses treat 80+ as a single tail bin).
adult["age_start"] = (adult["RIDAGEYR"] // 5).astype(int) * 5
adult.loc[adult["age_start"] >= 80, "age_start"] = 80

print(f"adult-with-status rows: {len(adult):,}")
print()
print("Counts by cycle:")
print(adult.groupby("cycle").size().to_string())
""",
    "load_merge",
))


CELLS.append(code(
    """\
# Sanity check: per-cycle overall prevalence should be smooth-
# decreasing for current, smooth-increasing for former, near-flat
# for never (over the 12-year window).
def w_share(g, cat):
    return g.loc[g["smoking"] == cat, "weight"].sum() / g["weight"].sum()


by_cycle = adult.groupby("cycle").apply(
    lambda g: pd.Series({
        cat: w_share(g, cat) for cat in ["current", "former", "never"]
    })
).reset_index()
print(by_cycle.round(3).to_string(index=False))
""",
    "by_cycle",
))


CELLS.append(code(
    """\
# Persist as parquet for the next notebook.
keep = ["SEQN", "cycle", "sex", "RIDAGEYR", "age_start",
        "smoking", "weight"]
adult[keep].to_parquet(
    DERIVED / "nhanes_smoking.parquet", index=False
)
print(f"wrote {DERIVED / 'nhanes_smoking.parquet'} ({len(adult):,} rows)")
""",
    "write_parquet",
))


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "01_load_and_merge.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out} ({len(CELLS)} cells)")
