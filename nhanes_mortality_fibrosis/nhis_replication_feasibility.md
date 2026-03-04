# Can We Replicate This Analysis with NHIS Linked Mortality Data?

**Short answer: No** — not the FIB-4 analysis specifically. The NHIS lacks the
laboratory biomarkers required to compute FIB-4 and other non-invasive fibrosis
scores. But a complementary analysis using **self-reported liver disease** is
possible, and we built it in notebooks `04_nhis_data_download.ipynb` and
`05_nhis_pooled_analysis.ipynb`.

## What Our NHANES Analysis Requires

The pooled analysis in `03_pooled_analysis.ipynb` uses these variables:

| Variable | Purpose | Source in NHANES |
|---|---|---|
| AST, ALT | FIB-4 numerator/denominator | Lab (hepatic panel) |
| PLATELETS | FIB-4 denominator | Lab (CBC) |
| BMXBMI | Matching covariate, NFS input | Exam (anthropometry) |
| SBP_MEAN | Matching covariate | Exam (blood pressure) |
| LBDLDL | Step 5 covariate | Lab (lipids) |
| LBXGLU | Descriptive | Lab (plasma glucose) |
| SMOKE_EVER | Matching covariate | Questionnaire |
| RIDAGEYR, RIAGENDR | Matching + FIB-4 | Demographics |
| MORTSTAT, UCOD_LEADING, PERMTH_EXM | Outcome | Linked mortality file |

FIB-4 = (Age × AST) / (Platelets × √ALT). Without AST, ALT, and platelet
count the score cannot be computed.

## What NHIS Provides

The NHIS is a household interview survey. It collects no blood samples and
performs no physical examinations. Its strengths are large sample size
(~35,000–40,000 adults/year) and long time span (1957–present, with
public-use linked mortality files for 1986–2018, follow-up through 2019).

### Variables NHIS *does* collect annually

- **Demographics**: age, sex, race/ethnicity, education, income
- **BMI**: self-reported height and weight (available 1976+; sample-adult
  self-report only from 1997+)
- **Diabetes**: "Ever told by a doctor you have diabetes" (annually)
- **Smoking**: lifetime cigarette use, current status (annually)
- **Hypertension**: "Ever told you have hypertension" (annually)
- **Alcohol use**: frequency and quantity (most years)
- **Health care access/utilization**: insurance, usual source of care, ER visits

### Variables NHIS *does not* collect

- AST, ALT, or any liver enzyme
- Platelet count or any CBC component
- Albumin, LDL cholesterol, glucose, HbA1c
- Blood pressure measurement (self-report of hypertension diagnosis only)
- Any physical examination measurement

### Liver-specific questions in NHIS

- **Pre-2019**: `LIVERFATYRC` — chronic fatty liver in the past year
  (constructed from condition records by IPUMS NHIS)
- **2019+ redesign**: "Ever told by a doctor you have cirrhosis or any other
  kind of long-term liver condition?" — part of the annual chronic-conditions
  battery (rotating content)

These are self-reported diagnoses, not biomarker-derived classifications.

## NHIS Linked Mortality Files

NHIS data *are* linked to the National Death Index, with the same structure as
the NHANES linked mortality files:

- **Public-use**: NHIS 1986–2018, follow-up through December 31, 2019
- **Restricted-use**: NHIS 1986–2021, follow-up through December 31, 2022
- Variables: `ELIGSTAT`, `MORTSTAT`, `UCOD_LEADING` (113-group recode),
  `PERMTH_INT`, `DIABETES` (MCOD flag), `HYPERTEN` (MCOD flag)
- The same data-perturbation approach used for NHANES public-use files applies

Source: [CDC NCHS Public-Use Linked Mortality Files](https://www.cdc.gov/nchs/data-linkage/mortality-public.htm)

## Why a Direct Replication Is Not Feasible

The fundamental mismatch: our analysis defines exposure using a *biomarker-derived*
fibrosis score (FIB-4 ≥ 2.67). NHIS has no biomarkers. There is no way to
compute FIB-4, NFS, APRI, or any other non-invasive fibrosis index from NHIS
data alone.

There is also no NHIS–NHANES cross-linkage. The two surveys draw from separate
sampling frames, and participant identifiers are not shared.

## What *Could* Be Done with NHIS

Although we cannot replicate the FIB-4 analysis, NHIS offers a complementary
angle:

### 1. Self-reported liver disease and mortality

- **Exposure**: Self-reported diagnosis of liver condition (cirrhosis /
  long-term liver disease, 2019+ annual core; fatty liver, pre-1997 condition
  records)
- **Outcome**: All-cause and cause-specific mortality via linked NDI
- **Matching**: Age, sex, race, BMI (self-reported), diabetes, smoking,
  hypertension — all available
- **Advantage**: Massive sample size (~35k adults/year × 30+ years) and longer
  follow-up windows than NHANES offers
- **Limitation**: Self-reported liver disease captures only *diagnosed* cases,
  missing the large pool of undiagnosed NAFLD/fibrosis that the biomarker
  approach identifies

### 2. Diabetes/obesity subgroup mortality trends

NHIS data could support a time-trends analysis of mortality among adults with
diabetes and obesity (major NAFLD risk factors), matched to controls, using the
same propensity-score matching framework we built for NHANES. This would not
measure fibrosis directly but would characterize the mortality burden in the
population most at risk for it.

### 3. Validating the self-report question against NHANES

One useful cross-survey analysis: compare the prevalence of self-reported liver
disease in NHIS with the biomarker-derived fibrosis prevalence in NHANES for
overlapping years. This cannot use linked individual records (the surveys are
independent) but could compare population-level estimates and assess how much
subclinical fibrosis the self-report question misses.

## Summary

| Feature | NHANES (current analysis) | NHIS (potential) |
|---|---|---|
| Lab biomarkers (AST, ALT, platelets) | Yes | **No** |
| FIB-4 / NFS computable | Yes | **No** |
| Self-reported liver disease | Yes | Yes |
| Linked mortality (public-use) | 1999–2018 | 1986–2018 |
| Sample size per year | ~5,000 adults | ~35,000 adults |
| Physical exam (BMI, BP) | Measured | Self-reported |
| Demographics, smoking, diabetes | Yes | Yes |

**Bottom line**: NHIS linked mortality data cannot replicate the FIB-4–based
analysis. NHIS could support a *complementary* study of self-reported liver
disease and mortality, with the advantage of much larger sample sizes and
longer follow-up, but the disadvantage of capturing only diagnosed disease.
