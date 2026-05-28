# NHANES dietary Na/K ratio — proof-of-concept prediction

Train on NHANES 2007-2016, hold out NHANES 2017-2018, and ask: how well do EHR-available
predictors recover dietary Na/K (mg sodium / mg potassium, Day-1 + Day-2 24-hour recall
average) at the individual and population level?

## Quickstart

```bash
uv venv && uv pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute 01_data_download.ipynb --inplace
.venv/bin/jupyter nbconvert --to notebook --execute 02_model_fitting.ipynb --inplace
.venv/bin/jupyter nbconvert --to notebook --execute 03_report.ipynb --inplace
.venv/bin/jupyter nbconvert --to notebook --execute 00_overview.ipynb --inplace
# open outputs/report.html in a browser
```

## Files

- `00_overview.ipynb` — headline numbers, summary
- `01_data_download.ipynb` — NHANES download + analytic parquet build
- `02_model_fitting.ipynb` — train + holdout evaluation (mean baseline, Ridge, LightGBM)
- `03_report.ipynb` — generates `outputs/report.html`
- `outputs/report.html` — interactive Plotly report
- `outputs/nih_capsule.md` — concise summary for NIH proposal scientific review

## Result snapshot (holdout = NHANES 2017-2018)

| Model            |   R²  | RMSE  |  MAE  |
|------------------|------:|------:|------:|
| Mean baseline    | 0.000 | 0.508 | 0.398 |
| Ridge regression | 0.087 | 0.485 | 0.379 |
| **LightGBM**     | **0.097** | **0.482** | **0.377** |

Survey-weighted overall Na/K: observed 1.394 vs predicted 1.367 (bias −1.9%).
Catchment N=1000 95% interval ±~0.03 Na/K units; N=5000 ±~0.02.

See `outputs/nih_capsule.md` for the NIH-proposal summary.
