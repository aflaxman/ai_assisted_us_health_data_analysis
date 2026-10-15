# NHANES benchmark for the MHANES testosterone-screening paper

Hennigar et al. (JAMA, 15 Oct 2026) report that 25.7% of active-duty
Army men have a single morning total testosterone below 300 ng/dL,
using the Military Health and Nutrition Examination Study (MHANES).
MHANES has no public-use file (ClinicalTrials.gov NCT06380322: IPD
sharing "No"; controlled access on application per the protocol
paper). This project recomputes every reported quantity on the nearest
public substitute: NHANES 2021–2023 men aged 18–59 drawn in the
morning fasting session, raked to the 2021 active-duty Army age and
race/ethnicity distribution.

Start with [00_overview.ipynb](00_overview.ipynb).

```
uv venv && uv pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    01_download_prepare.ipynb 02_army_calibrated_prevalence.ipynb 00_overview.ipynb
```

Raw NHANES files download into `../data/raw/nhanes/<cycle>/`; the
derived parquet goes to `../data/derived/` (both gitignored).
