"""Build 05_extend_to_skeleton.ipynb: read the per-(sex, age band)
F4-calibrated lognormal table from notebook 04 and extend it with
placeholder rows so it covers every (sex, age_start) cell in the
GBD demographic skeleton — that's what the consuming simulation
project's loader requires (it is strict; no fallback).
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
# 05 — Extend LSM lookup to the full demographic skeleton

Notebook 04 fit F4-calibrated lognormals for adults 60+ — the only
ages where NHANES P_LUX has enough elastography data to anchor a
per-cell (mean, SD). The consuming simulation project's loader is
strict: it requires a row for every (sex, age_start) in the GBD
demographic skeleton (ages 0 through 95+), since stub fallbacks
were removed at the close of Model 4.

This notebook reads `liver_stiffness_age_sex_lognormal.csv` from
notebook 04 and rewrites it in place with placeholder rows for
every sub-60 skeleton bin. Each placeholder uses the corresponding
sex's youngest fitted band (60–64) for `mean_kpa` and `sd_kpa`;
`f4_share_target` is left as the same anchor F4 share. The
simulation only enrolls ages 65–80, so these placeholder cells are
never sampled at run time — they exist only to satisfy the
artifact build's coverage check.

Why a forward fill from the 60–64 row rather than a literature-based
younger-age curve?

- The trial doesn't enroll under 65, so any value is fine for
  artifact-build purposes — none of these cells affect the trial
  outcome.
- A constant fill is the most conservative thing we can do without
  another NHANES extraction. If a future model needs to enroll
  under 60, replace these rows with cell-fitted values from a
  younger LSM dataset (e.g., elastography from the regular NHANES
  cycles once they collect it across the full age range).
- The under-60 placeholders are clearly tagged via `f4_share_target`
  inheriting from the 60–64 anchor — readers can spot them in the
  CSV.

Skeleton age_start values come from
`vivarium_inputs.interface.get_demographic_dimensions("United States
of America")`: GBD age groups 0, 0.0192, 0.0767, 0.5, 1, 2, 5, 10,
15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90,
95. The loader caps `age_start` at 80 before lookup, so anything
≥80 already lands on the 80–89 row from notebook 04.
""",
    "intro",
))


CELLS.append(code(
    """\
from pathlib import Path

import pandas as pd

OUT = Path("outputs")
csv_path = OUT / "liver_stiffness_age_sex_lognormal.csv"

fitted = pd.read_csv(csv_path)
print("notebook-04 fitted rows:")
print(fitted.round(3).to_string(index=False))
""",
    "load",
))


CELLS.append(code(
    """\
# Sub-60 GBD age_start values (after the loader's age_start ≤ 80
# cap, but the cap doesn't bite below 60). Floats — the loader
# does NOT cast to int for LSM, so each row's age_start must match
# the skeleton's value exactly.
SUB_60_AGE_STARTS = [
    0.0, 0.01917808, 0.07671233, 0.5,
    1.0, 2.0, 5.0,
    10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0,
]

# Anchor each sex's placeholder rows to its 60–64 fitted row. That
# row has age_start 60.0 in notebook 04's output.
anchor = (
    fitted[fitted["age_start"] == 60.0]
    .set_index("sex")[["mean_kpa", "sd_kpa", "f4_share_target"]]
)
print("anchor rows (60–64):")
print(anchor.round(3).to_string())
""",
    "anchor",
))


CELLS.append(code(
    """\
# Build sub-60 placeholder rows: cross-product of (sex × sub-60 age
# bins). age_end is exclusive — the GBD bins below 60 are spaced as
# {0, 0.0192, 0.0767, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40,
# 45, 50, 55, 60}, so each row's age_end is the next age_start.
GBD_AGE_GRID = [
    0.0, 0.01917808, 0.07671233, 0.5,
    1.0, 2.0, 5.0,
    10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0,
]
age_end_for = dict(zip(GBD_AGE_GRID[:-1], GBD_AGE_GRID[1:]))

placeholder_rows = []
for sex in ("Female", "Male"):
    a = anchor.loc[sex]
    for age_lo in SUB_60_AGE_STARTS:
        placeholder_rows.append({
            "sex": sex,
            "age_start": age_lo,
            "age_end": age_end_for[age_lo],
            "mean_kpa": float(a["mean_kpa"]),
            "sd_kpa": float(a["sd_kpa"]),
            "f4_share_target": float(a["f4_share_target"]),
        })

placeholders = pd.DataFrame(placeholder_rows)
print(f"placeholder rows added: {len(placeholders)}")
print(placeholders.round(3).head(8).to_string(index=False))
""",
    "placeholders",
))


CELLS.append(code(
    """\
full = pd.concat([placeholders, fitted], ignore_index=True)
full = full.sort_values(["sex", "age_start"]).reset_index(drop=True)

# Defensive: every (sex, age_start) we care about must be present.
expected = set()
for sex in ("Female", "Male"):
    for a in SUB_60_AGE_STARTS + [60.0, 65.0, 70.0, 75.0, 80.0]:
        expected.add((sex, a))
got = set(zip(full["sex"], full["age_start"]))
missing = sorted(expected - got)
assert not missing, f"missing skeleton cells after merge: {missing}"

full.to_csv(csv_path, index=False)
print(f"rewrote {csv_path} with {len(full)} rows")
print()
print("first / last rows:")
print(full.head(8).round(3).to_string(index=False))
print("...")
print(full.tail(8).round(3).to_string(index=False))
""",
    "write_csv",
))


CELLS.append(md(
    """\
## Summary

The CSV at `outputs/liver_stiffness_age_sex_lognormal.csv` now
covers every (sex, age_start) cell in the GBD demographic skeleton
that the consuming simulation project's loader needs after the
age_start ≤ 80 cap. Sub-60 cells are placeholder rows
forward-filled from the corresponding sex's 60–64 fitted row;
fitted ages 60–89 come from notebook 04 unchanged.

If a future model enrolls under 60, replace these placeholder rows
with cell-fitted values from a wider-age LSM dataset.
""",
    "summary",
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

out = HERE / "05_extend_to_skeleton.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out} ({len(CELLS)} cells)")
