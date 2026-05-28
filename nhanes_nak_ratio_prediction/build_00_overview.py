"""Generate 00_overview.ipynb."""
from _build_notebook import build, md, code

cells = [
    md("""# 00 — Overview

**Project.** Linking NHANES and EHR data to predict the dietary sodium/potassium (Na/K) ratio
at the local level — a proof-of-concept feasibility check.

**Why it matters.** Dietary Na/K ratio is a stronger predictor of cardiovascular mortality
than sodium or potassium alone, and it is responsive to dietary policy. National estimates
(NHANES) anchor surveillance but cannot resolve sub-state geography; EHR data (e.g., Epic
Cosmos) offers the local scale but lacks 24-hour dietary recall. A prediction model trained
in NHANES on variables that *also* exist in EHR can transfer national estimates to
sub-national populations.

**This proof of concept.** Hold out the most recent regular NHANES cycle (2017-2018), fit a
model on the previous 5 cycles (2007-2016), and ask: how well do the model's predictions
generalise on individual and population levels?

---

## Files

| Notebook                  | Purpose                                                      |
|---------------------------|--------------------------------------------------------------|
| `01_data_download.ipynb`  | Download NHANES files, build pooled analytic parquet         |
| `02_model_fitting.ipynb`  | Train mean baseline, ridge, LightGBM; evaluate on holdout    |
| `03_report.ipynb`         | Build interactive `outputs/report.html`                      |
| `00_overview.ipynb`       | (this notebook) headline numbers + links to outputs          |

## Outputs

- `outputs/report.html` — interactive Plotly report (open in a browser)
- `outputs/nih_capsule.md` — concise summary for NIH scientific review
- `../data/derived/nak_pooled_2007_2018.parquet` — analytic dataset
- `../data/derived/nak_holdout_predictions.parquet` — per-person holdout predictions
- `../data/derived/nak_model.pkl` — pickled LightGBM + Ridge models

## Quickstart

```bash
uv venv && uv pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute 01_data_download.ipynb --inplace
.venv/bin/jupyter nbconvert --to notebook --execute 02_model_fitting.ipynb --inplace
.venv/bin/jupyter nbconvert --to notebook --execute 03_report.ipynb --inplace
```"""),
    md("""## Headline numbers"""),
    code("""import os
import numpy as np
import pandas as pd

DATA = os.path.abspath(os.path.join('..', 'data'))
DERIVED = os.path.join(DATA, 'derived')

pred = pd.read_parquet(os.path.join(DERIVED, 'nak_holdout_predictions.parquet'))
summary = pd.read_csv(os.path.join(DERIVED, 'nak_model_summary.csv'))
clust = pd.read_parquet(os.path.join(DERIVED, 'nak_catchment_sim.parquet'))

pred['wt'] = pred['weight_diet2d'].fillna(pred['weight_diet1d']).fillna(pred['weight'])
obs = np.average(pred['NAK_RATIO'], weights=pred['wt'])
prd = np.average(pred['pred_lgbm'],  weights=pred['wt'])
print(f"holdout cycle: 2017-2018, n = {len(pred):,} adults (>=20y, valid 24-hr recall)")
print()
print('Individual-level performance:')
print(summary.round(4).to_string(index=False))
print()
print('Survey-weighted overall means (LightGBM):')
print(f"  observed : {obs:.3f}")
print(f"  predicted: {prd:.3f}")
print(f"  bias     : {prd-obs:+.3f}  ({100*(prd-obs)/obs:+.1f}%)")
print()
print('Catchment-scale precision (random clusters from holdout):')
print(clust.groupby('N')['error']
      .agg(mean='mean', sd='std',
           ci_lo=lambda x: np.percentile(x,2.5),
           ci_hi=lambda x: np.percentile(x,97.5))
      .round(4).to_string())"""),
    md("""## Interpretation

- **Individual-level R² ≈ 0.10.** Expected: dietary intake on a given day has huge
  within-person variability that EHR predictors cannot capture. This is the same issue
  the National Cancer Institute "usual intake" method is designed to address through
  shrinkage toward subgroup means.
- **Population-level bias ≈ −2%.** The model under-predicts the 2017-2018 mean Na/K by
  about 0.03 mg/mg out of a 1.40 mean. This residual reflects (a) modest secular drift in
  the underlying diet, and (b) limits of the available predictors. Adding calendar-year or
  area-deprivation predictors should reduce this.
- **Catchment precision.** For random N=1000 catchments drawn from the holdout, the 95%
  prediction interval on the population mean is about ±0.03 Na/K units. For N=5000 (a
  large county or hospital catchment) the precision is ±0.02. These are the scales at
  which the proposed EHR application would operate.

**Verdict:** The proof-of-concept is supportive. The signal carried by the EHR-available
predictor set is enough to recover *population-mean* Na/K to within a few percent of the
NHANES gold standard, and to discriminate demographic subgroups by their typical exposure.
The next step is to validate against (a) the NHANES 2014 24-hour urinary sodium sub-study
and (b) Epic Cosmos populations stratified by geography."""),
]

build(cells, '00_overview.ipynb')
print('wrote 00_overview.ipynb')
