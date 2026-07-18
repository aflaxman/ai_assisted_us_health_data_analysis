"""Build 05_extend_to_skeleton.ipynb: extend BOTH the LSM and CAP per-cell tables
with placeholder rows so each covers every (sex, age_start) in the GBD demographic
skeleton, which the consuming simulation's strict loader requires. Sub-60 rows are
forward-filled from each sex's youngest fitted band (60-64); fitted ages 60+
(including the open-ended 80+ terminal band) come from notebook 04 unchanged.
"""
from pathlib import Path

from _nbtools import md, code, write_notebook

HERE = Path(__file__).parent
CELLS = []

CELLS.append(md(
    """\
# 05 - Extend LSM + CAP lookups to the full demographic skeleton

Notebook 04 fit per-(sex, 5-year band) distributions for adults **60+** -- the
ages where NHANES FibroScan has enough data to anchor a cell, and the only ages
the trial enrols (65-80, plus a 60-64 buffer). The consuming simulation's loader
is strict: it needs a row for **every** `(sex, age_start)` in the GBD skeleton
(ages 0 through 95+), because stub fallbacks were removed at the close of Model 4.

This notebook reads the two `outputs/` tables and rewrites each **in place** with
forward-filled placeholder rows for every sub-60 skeleton bin, using the
corresponding sex's youngest fitted band (60-64). The simulation only enrols
65-80, so these placeholders are never sampled at run time -- they exist only to
satisfy the artifact build's coverage check, and are tagged `source =
forward_filled` so readers can spot them.

The fitted ages 60+ are unchanged, including the **open-ended 80+ terminal band**
(`age_start=80, age_end=125`): NHANES top-codes age at 80, so that cell is the
whole 80+ mixture, not a 5-year bin, and the loader (which caps `age_start` at 80)
lands every 80+ simulant on it.
""",
    "intro",
))

CELLS.append(code(
    """\
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path('outputs')
LSM_CSV = OUT / 'liver_stiffness_age_sex_lognormal.csv'
CAP_CSV = OUT / 'cap_age_sex_distribution.csv'

# Sub-60 GBD age_start values and the age grid used to set each row's exclusive
# age_end (each row's age_end is the next skeleton age_start).
GBD_AGE_GRID = [
    0.0, 0.01917808, 0.07671233, 0.5, 1.0, 2.0, 5.0,
    10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0,
]
SUB_60_AGE_STARTS = GBD_AGE_GRID[:-1]
age_end_for = dict(zip(GBD_AGE_GRID[:-1], GBD_AGE_GRID[1:]))

def sub60_label(lo):
    hi = age_end_for[lo]
    return f'{lo:g}-{hi:g}'
""",
    "grid",
))

CELLS.append(code(
    """\
def extend_over_skeleton(csv_path):
    fitted = pd.read_csv(csv_path)
    # idempotent: drop any placeholder rows from a prior run before re-extending
    if 'source' in fitted.columns:
        fitted = fitted[fitted['source'] != 'forward_filled'].reset_index(drop=True)
    anchor = fitted[fitted['age_start'] == 60.0].set_index('sex')
    fill_cols = [c for c in fitted.columns
                 if c not in ('sex', 'age_start', 'age_end', 'age_group',
                              'top_coded', 'source', 'n', 'n_eff')]
    placeholders = []
    for sex in ('Female', 'Male'):
        a = anchor.loc[sex]
        for lo in SUB_60_AGE_STARTS:
            row = {'sex': sex, 'age_start': lo, 'age_end': age_end_for[lo],
                   'age_group': sub60_label(lo), 'top_coded': False,
                   'source': 'forward_filled', 'n': 0, 'n_eff': 0.0}
            for c in fill_cols:
                row[c] = a[c]
            placeholders.append(row)
    full = (pd.concat([pd.DataFrame(placeholders), fitted], ignore_index=True)
              .sort_values(['sex', 'age_start']).reset_index(drop=True))

    # Coverage check: every skeleton (sex, age_start) present after the loader's
    # age_start <= 80 cap (so anything >= 80 lands on the 80+ row).
    expected = {(s, a) for s in ('Female', 'Male')
                for a in SUB_60_AGE_STARTS + [60.0, 65.0, 70.0, 75.0, 80.0]}
    got = set(zip(full['sex'], full['age_start']))
    missing = sorted(expected - got)
    assert not missing, f'missing skeleton cells: {missing}'

    full = full[list(fitted.columns)]     # preserve column order
    full.to_csv(csv_path, index=False)
    return full

lsm_full = extend_over_skeleton(LSM_CSV)
cap_full = extend_over_skeleton(CAP_CSV)
print(f'LSM: {len(lsm_full)} rows  ({(lsm_full[\"source\"] == \"fitted\").sum()} fitted, '
      f'{(lsm_full[\"source\"] == \"forward_filled\").sum()} forward-filled)')
print(f'CAP: {len(cap_full)} rows  ({(cap_full[\"source\"] == \"fitted\").sum()} fitted, '
      f'{(cap_full[\"source\"] == \"forward_filled\").sum()} forward-filled)')
""",
    "extend",
))

CELLS.append(code(
    """\
# Show the transition around age 60 and the open-ended 80+ terminal row.
cols = ['sex', 'age_start', 'age_end', 'age_group', 'source', 'mean_kpa', 'sd_kpa', 'f4_share_target']
mid = lsm_full[(lsm_full['sex'] == 'Female') & (lsm_full['age_start'].between(50, 80))]
print('LSM, Female, ages 50-80 (forward-fill -> fitted, ending in open 80+):')
print(mid[cols].round(3).to_string(index=False))
print()
capcols = ['sex', 'age_start', 'age_end', 'age_group', 'source', 'cap_mean', 'cap_sd']
print('CAP, Female, ages 55-80:')
print(cap_full[(cap_full['sex'] == 'Female') & (cap_full['age_start'].between(55, 80))][capcols]
      .round(2).to_string(index=False))
""",
    "peek",
))

CELLS.append(md(
    """\
## Summary

Both `outputs/liver_stiffness_age_sex_lognormal.csv` and
`outputs/cap_age_sex_distribution.csv` now cover every `(sex, age_start)` cell the
strict loader needs after its `age_start <= 80` cap. Sub-60 cells are
forward-filled from the 60-64 fit and tagged `source = forward_filled`; fitted
ages 60+ (with the open-ended 80+ terminal band) are unchanged. If a future model
enrols under 60, replace the placeholders with cell fits from a wider-age dataset.
""",
    "summary",
))

if __name__ == "__main__":
    write_notebook(HERE / "05_extend_to_skeleton.ipynb", CELLS)
