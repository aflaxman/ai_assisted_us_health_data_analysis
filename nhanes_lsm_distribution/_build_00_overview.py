"""Build 00_overview.ipynb: the read-only project overview -- summary, notebook
map, and headline results loaded from the committed output tables.
"""
from pathlib import Path

from _nbtools import md, code, write_notebook

HERE = Path(__file__).parent
CELLS = []

CELLS.append(md(
    """\
# NHANES liver FibroScan (LSM + CAP) distributions - overview

Estimates the U.S.-adult population distributions of the two NHANES transient-
elastography measurements, per (sex, 5-year age band), for the consuming
microsimulation (`vivarium_csu_mace_rct`):

- **LSM** - liver stiffness (kPa, `LUXSMED`) -> **fibrosis stage** (F0-F4)
- **CAP** - controlled attenuation parameter (dB/m, `LUXCAPM`) -> hepatic **steatosis**

## What the simulation needs

The simulation routes each simulant to an excess-mortality rate by **fibrosis
stage**, so the load-bearing quantity is the population **stage-share vector**.
The project priority is accuracy at **F1/F2/F3** -- the stages carrying most of the
population -- ahead of F4 (cirrhosis). Stage cutoffs are the repo-standard
**6 / 8 / 10 / 15 kPa** (F0<6, F1 6-8, F2 8-10, F3 10-15, F4>=15), matching
`nhanes_fibrosis_modeling`.

## Data source: two pooled NHANES cycles (2017-2023)

FibroScan first appears in the **2017 - March 2020 pre-pandemic release
(`P_LUX`, weight `WTMECPRP`)** and continues in **2021 - August 2023 (`LUX_L`,
weight `WTMEC2YR`)**. We **pool both cycles** (each MEC weight halved), which
roughly doubles the analytic sample and tightens every per-cell estimate. Age is
**top-coded at 80** in both cycles, so the terminal band is an open-ended 80+ cell.

## Method (notebook 04)

LSM is fit as a two-parameter **lognormal** -- the downstream sampler's contract --
whose `(mean_kpa, sd_kpa)` are chosen to **minimise the weighted stage-share error,
prioritising F1/F2/F3** (with small-cell targets smoothed across age). CAP is a
moment-matched **Normal** per cell. A single lognormal cannot also reproduce the
heavy >=15 kPa tail, so F4 is deliberately traded for F1/F2/F3 accuracy
(quantified in notebook 06).

## Notebooks

| # | Notebook | Purpose |
| --- | --- | --- |
| 01 | `01_download_lux.ipynb` | Download + pool both cycles; carry LSM and CAP; write pooled parquet |
| 02 | `02_lsm_marginal.ipynb` | Weighted LSM + CAP age x sex marginals; fibrosis stage-share profiles |
| 03 | `03_lsm_transformations.ipynb` | Transformation/outlier robustness; stage-share goodness of fit |
| 04 | `04_lsm_age_sex_calibration.ipynb` | **Core fit**: multi-cutoff lognormal (LSM) + moment-matched Normal (CAP); writes outputs |
| 05 | `05_extend_to_skeleton.ipynb` | Forward-fill both tables over the GBD demographic skeleton |
| 06 | `06_method_comparison.ipynb` | Methods side by side on stage-share accuracy; why multi-cutoff |
| 07 | `07_categorical_comparison.ipynb` | Investigation: a two-level (categorical joint stage + within-stage continuous) alternative vs the continuous fit |

## Outputs (`outputs/`)

- `liver_stiffness_age_sex_lognormal.csv` - LSM loader table (`mean_kpa`, `sd_kpa`
  + empirical/fitted stage shares + provenance)
- `cap_age_sex_distribution.csv` - CAP `(cap_mean, cap_sd)` + steatosis-grade shares
- `lsm_cap_calibration.meta.json` - cutoff ladders, stage weights, calibration objective
""",
    "intro",
))

CELLS.append(md("## Headline results (loaded from the committed outputs)", "md_results"))

CELLS.append(code(
    """\
import json
from pathlib import Path
import pandas as pd

OUT = Path('outputs')
meta = json.load(open(OUT / 'lsm_cap_calibration.meta.json'))
print('LSM cutoffs (kPa):', meta['lsm_cutoffs_kpa'], '| stage weights:', meta['lsm_stage_weights'])
print('CAP family:', meta['cap_dist_family'], '| pooled LSM 60+ n =', meta.get('n_pooled_lsm_60plus'))
print()

lsm = pd.read_csv(OUT / 'liver_stiffness_age_sex_lognormal.csv')
fitted = lsm[lsm['source'] == 'fitted']
print('LSM fitted cells (mean_kpa, sd_kpa) with F1/F2/F3 empirical vs fitted shares:')
cols = ['sex', 'age_group', 'mean_kpa', 'sd_kpa',
        'lsm_f1_share', 'lsm_f1_fit', 'lsm_f2_share', 'lsm_f2_fit', 'lsm_f3_share', 'lsm_f3_fit']
print(fitted[cols].round(3).to_string(index=False))
""",
    "lsm_results",
))

CELLS.append(code(
    """\
cap = pd.read_csv(OUT / 'cap_age_sex_distribution.csv')
capf = cap[cap['source'] == 'fitted']
print('CAP fitted cells (dB/m):')
print(capf[['sex', 'age_group', 'cap_mean', 'cap_sd']].round(1).to_string(index=False))
print()
print('See notebook 06 for the method comparison: multi-cutoff calibration cuts the')
print('mean F1/F2/F3 stage-share error to ~1.3 pp (vs ~3.5-3.9 pp for the old')
print('F4-calibrated and log-moment-match methods), trading F4-tail accuracy.')
""",
    "cap_results",
))

if __name__ == "__main__":
    write_notebook(HERE / "00_overview.ipynb", CELLS)
