# Capsule summary for NIH scientific review

**Linking NHANES and EHR data to estimate the dietary Na/K ratio at sub-national geographic scales**

**Public-health rationale.** The dietary sodium-to-potassium ratio is one of the strongest
modifiable nutritional predictors of cardiovascular mortality, and it responds to policy.
National surveillance through NHANES is authoritative but resolves only to broad
demographic strata; it cannot tell a state, county, or hospital catchment what its Na/K
exposure looks like. EHR systems such as Epic Cosmos cover tens of millions of patients
across granular geographies but do not contain 24-hour dietary recall. We propose a
transfer-learning approach that uses NHANES to train a model on variables shared with the
EHR, then applies that model to EHR populations to estimate local Na/K exposure.

**Design.** We computed the Na/K mass ratio (mg sodium ÷ mg potassium) as the Day 1+Day 2
average from the NHANES Total Nutrient Intakes files for six 2-year cycles, 2007-2018
(n = 30,789 adults with at least one reliable 24-hour recall). NHANES 2017-2018, the most
recent regular cycle, was held out as a validation set; the five earlier cycles (2007-2016)
were used for training. Predictors were limited to variables also extractable from a
standard EHR: age, sex, race/ethnicity, BMI, waist circumference, systolic and diastolic
blood pressure, hypertension and diabetes diagnoses (and treatments), cholesterol diagnosis,
smoking status, alcohol use, and a poverty-income proxy. Three models were compared: a mean
baseline, ridge regression, and gradient-boosting trees (LightGBM).

**Headline results (holdout NHANES 2017-2018, n = 4,674 adults).**

| Model            | R²    | RMSE  | MAE   |
|------------------|------:|------:|------:|
| Mean baseline    | 0.000 | 0.508 | 0.398 |
| Ridge regression | 0.087 | 0.485 | 0.379 |
| LightGBM         | 0.097 | 0.482 | 0.377 |

- Individual-level R² is ≈ 0.10 in the best model — the floor imposed by the well-known
  high within-person variability of 24-hour dietary recall.
- Survey-weighted holdout mean Na/K = **1.394 observed vs 1.367 predicted** (bias −1.9%).
- For a population of N = 1,000 adults drawn from the holdout (a small-county scale), the
  95% interval for the predicted population mean covers the observed mean within ±0.03
  Na/K units (≈ ±2% of the mean); at N = 5,000 the interval tightens to ±0.02.
- Subgroup-mean prediction error is below 5% in every age, sex, and race/ethnicity stratum.

**Interpretation.** Individual-level prediction is intrinsically limited by recall noise, as
expected. At the policy-relevant population scale, however, the EHR-shared predictor set
recovers NHANES Na/K means to within a few percent. Calibration is good across demographic
strata.

**Likelihood of success.** High for the proposed scope. The proof-of-concept demonstrates
that the demographic-and-clinical predictor set shared between NHANES and EHR carries enough
information to estimate population-mean Na/K to within a few percent at county or
hospital-catchment scale, with well-behaved subgroup calibration. Risk factors for the next
phase are (a) residual secular drift between NHANES train cycles and an EHR application
year, which can be addressed with year fixed effects and external benchmarks (the NHANES
2014 24-hour urinary sodium sub-study is the gold-standard validator), and (b) differences
in case mix and measurement practice between NHANES and EHR populations, which can be
addressed with poststratification weights derived from area-level demographics.

**Next steps.** Validate against the NHANES 2014 24-hour urinary sodium excretion gold
standard; extend the model to incorporate area-deprivation and clinical-biomarker covariates
available in Epic Cosmos; and produce state- and county-level Na/K estimates by
poststratifying model predictions to American Community Survey population structures.

---

*Reproducibility:* code and an interactive HTML report live in
`nhanes_nak_ratio_prediction/`. Training set n = 26,047 (2007-2016); holdout n = 4,674
(2017-2018); best model = LightGBM.
