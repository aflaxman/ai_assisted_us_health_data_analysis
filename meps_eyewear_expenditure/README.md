# Replication: Li, Li & Sansgiry (2024) — Predictors of Annual Eyewear Expenditures

**Source abstract:** Value in Health 27(6) Suppl., S153, ISPOR 2024.  
DOI: 10.1016/j.jval.2024.03.2289

## Quick summary

We replicate the analysis using MEPS Full-Year Consolidated files 2017–2021 (pooled).
Key descriptive estimates match exactly. Most logistic regression ORs overlap the abstract's
confidence intervals. Two discrepancies (age 65+ and education) are documented in
[`discrepancies.md`](discrepancies.md).

## Repository structure

```
meps_eyewear_expenditure/
├── code/
│   ├── 01_download_meps.py       Download MEPS Stata files from AHRQ
│   ├── 02_build_analytic_file.py Recode variables, pool years
│   ├── 03_descriptive_analysis.py Survey-weighted descriptive stats
│   ├── 04_logistic_regression.py  Survey-weighted logistic regression
│   ├── 05_compare_results.py     Side-by-side comparison table
│   ├── 06_fred_nhea_comparison.py MEPS vs NHEA (FRED) external validation
│   └── survey_utils.py           Taylor linearization survey statistics
├── outputs/                      Intermediate outputs (descriptive CSVs, OR tables)
├── results/                      Final comparison tables (CSV + markdown)
└── requirements.txt
```

Shared raw data: `../../data/raw/meps/` (MEPS .dta files — not tracked in git).  
Derived data: `../../data/derived/meps_eyewear_analytic.parquet` (gitignored).

## Data

**Source:** MEPS Full-Year Consolidated files, 2017–2021.

| Year | File code | HC number | N persons |
|------|-----------|-----------|-----------|
| 2017 | h201      | HC-201    | 31,880    |
| 2018 | h209      | HC-209    | 26,475    |
| 2019 | h216      | HC-216    | 27,805    |
| 2020 | h224      | HC-224    | 28,612    |
| 2021 | h233      | HC-233    | 32,222    |

Downloaded as Stata `.dta` files from:  
`https://meps.ahrq.gov/mepsweb/data_files/pufs/<hNNN>/<hNNN>dta.zip`

## Variables

| Concept | MEPS variable | Notes |
|---|---|---|
| Eyewear expenditures | `VISEXPyy` | Year-specific; e.g., `VISEXP17` for 2017 |
| Person weight | `PERWTyyF` | Year-specific; divided by 5 for pooled annual estimates |
| Stratum | `VARSTR` | Consistent across years |
| PSU | `VARPSU` | Consistent across years |
| Age | `AGE42X` | Age at 4th/5th round interview |
| Sex | `SEX` | 1=Male, 2=Female |
| Race/ethnicity | `RACETHX` | 5-level variable |
| Highest degree | `HIDEG` | Adult education level |
| Years of education | `EDUCYR` | Used for validation only |
| Poverty category | `POVCATyy` | Year-specific; 1–5 |
| Insurance coverage | `INSCOVyy` | Year-specific; 1=Private, 2=Public, 3=Uninsured |

## Analytic decisions (where abstract was silent)

### Weight pooling

We divide each year's person weight (`PERWTyyF`) by 5 before pooling. This is the standard
MEPS approach for multi-year analyses: the resulting weighted sums estimate **annual averages**,
not 5-year totals.

### Survey design for pooled file

MEPS reuses VARSTR codes across the 2019–2021 cohort. To avoid spurious cross-year
within-stratum variance, we create year-specific stratum IDs:
```
varstr_pool = year + "_" + int(VARSTR)   # e.g., "2019_2028"
```
PSU IDs are similarly made year-specific. This is equivalent to treating each year's sample
as independent (the appropriate assumption for a pooled multi-year analysis).

### Variable coding

**Age groups** (cut points not specified in abstract):
- `<18`, `18–44` (reference), `45–64`, `65+`

**Race/ethnicity** (`RACETHX`):
- 1=Hispanic (reference), 2=NH White, 3=NH Black, 4=NH Asian, 5=NH Other/multiple

**Education** (`HIDEG` coding verified from cross-tabulation with `EDUCYR`):
- 1=No degree (< HS) **(reference)**
- 2=GED, 3=HS diploma → coded as "HS"
- 4=Bachelor's, 5=Master's, 6=Doctoral, 7=Associate's/some college → "Some college+"
- 8=Not applicable (children under 18) → "Child (<18)"
- Negative codes = "Unknown"
- For persons age < 18: always coded "Child (<18)" regardless of HIDEG

**Poverty category** (`POVCATyy`):
- 1=Poor/negative income **(reference)**, 2=Near poor, 3=Low income, 4=Middle income, 5=High income

**Insurance** (`INSCOVyy`):
- 1=Any private **(reference for abstract comparison)**, 2=Public only, 3=Uninsured **(reference)**

### Logistic regression reference categories

Chosen to reproduce the abstract's reported ORs (all > 1), which constrains the reference:
- Age: 18–44 (abstract says "older age OR > 1", implying a younger reference)
- Sex: Male (abstract says female OR > 1)
- Race/eth: Hispanic (abstract says NH White OR > 1)
- Education: <HS (abstract says higher education OR > 1)
- Poverty: Poor/negative (abstract says "lower poverty OR > 1", i.e., higher income vs poor)
- Insurance: Uninsured (abstract says private insurance OR > 1)

### Variance estimation

We use Taylor series linearization with the standard stratified cluster formula:

```
V(T_hat) = sum_h [ n_h/(n_h-1) * sum_i (z_hi - z_bar_h)^2 ]
```

For logistic regression, the sandwich (linearization) estimator is used:
```
V(beta) = A^{-1} B A^{-1}
```
where A is the weighted Hessian and B is the design-based outer product of score totals.

Weights are normalized to mean 1 before fitting to improve numerical conditioning
(this does not change the estimates or their SEs).

### Singleton PSU handling

Zero singleton strata in the pooled file (all strata have ≥ 2 PSUs).

## How to reproduce

```bash
cd meps_eyewear_expenditure
uv venv .venv && uv pip install -r requirements.txt
.venv/bin/python code/01_download_meps.py   # ~50 MB download
.venv/bin/python code/02_build_analytic_file.py
.venv/bin/python code/03_descriptive_analysis.py
.venv/bin/python code/04_logistic_regression.py
.venv/bin/python code/05_compare_results.py
.venv/bin/python code/06_fred_nhea_comparison.py  # requires internet
```
