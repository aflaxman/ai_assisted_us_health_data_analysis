# NHANES smoking exposure — per-(sex, age) prevalence

Survey-weighted three-state (current / former / never) smoking
prevalence for U.S. adults, derived from NHANES 2007-2018. Output
CSV at `outputs/smoking_age_sex_prevalence.csv` is consumed by the
project loader (`load_smoking_exposure`).

## Notebooks

- `01_load_and_merge.ipynb` — load `SMQ_*.xpt` + `DEMO_*.xpt` for
  cycles 2007-2018, harmonize the CDC 3-state status, write a
  combined parquet to `../data/derived/nhanes_smoking.parquet`.
- `02_prevalence.ipynb` — survey-weighted prevalence by (sex,
  5-year age band), comparison to the prior scalar stubs, final
  per-cell CSV.

## Smoking-status definition

Standard CDC three-state, built from two SMQ items:

| SMQ020 (ever 100+ cig) | SMQ040 (now smoke)            | Status   |
| ---------------------- | ----------------------------- | -------- |
| Yes (1)                | Every day or some days (1, 2) | Current  |
| Yes (1)                | Not at all (3)                | Former   |
| No (2)                 | —                             | Never    |
| Refused / DK (7, 9)    | —                             | (drop)   |

## TODO(MODEL_5+) — Project to 2028 with IHME Future Health Scenarios

Smoking declined consistently across the 2007-2018 NHANES window
(current: 22.8 % → 17.4 %, ~0.5 pp/yr). The trial enrolls in 2028,
~10 years past the NHANES window's end. Linearly extrapolating that
trajectory pushes current-smoker prevalence to roughly 12 % by 2028
— much closer to the all-adult NHANES average than to the
historical 22 %, but still not the right shape per (sex, age).

Replace the static NHANES prevalence with an FHS-style projected
trajectory:

- IHME's Future Health Scenarios produces per-(location, sex, age,
  year, scenario) projections for major risk factors out to 2050,
  including current-smoking prevalence.
- Use the FHS reference scenario at year 2028 for the trial enroll-
  ment vector; use the trajectory across 2026–2033 for the
  observation window if simulant aging is meant to track secular
  change.
- The FHS pull lives at the cluster's standard FHS data path; the
  output of this notebook can be replaced wholesale by the
  projected-2028 prevalence frame.

The smoking-status RR (currently literature placeholders in
`data_values.SMOKING_IS_RR`) is a separate Model 4 task.

## TODO(MODEL_5+) — Other risk factors with secular trends

Smoking is the most pronounced case but not the only one. Adult-
mean values from the same NHANES window:

| Risk         | 2007       | 2017       | Δ/10y    | Direction       | FHS candidate? |
| ------------ | ---------- | ---------- | -------- | --------------- | -------------- |
| Smoking (cur)| 22.8 %     | 17.4 %     | -5.4 pp  | strong decline  | **Yes**        |
| BMI          | 28.6       | 29.9       | +1.3     | steady increase | **Yes**        |
| LDL-C        | 115.9 mg/dL| 111.3 mg/dL| -4.6     | decline (statins)| **Yes**       |
| FPG          | 107.3 mg/dL| 109.6 mg/dL| +2.3     | slight increase | Yes (likely)   |
| SBP          | n/a        | n/a        | mixed    | flat-ish        | Maybe          |
| Lp(a)        | —          | —          | none     | genetic         | No             |
| LSM          | —          | —          | unknown  | one-cycle data  | Out of scope   |
| KD / CKD     | GBD-pulled | GBD-pulled | (in GBD) | (handled)       | (already)      |

For the trial-enrollment vector, all four "Yes" risks would benefit
from FHS-projected 2028 distributions. The biggest payoff is on the
ones whose mean / variance has shifted most over the analysis
window relative to the trial year:

- **BMI** has been climbing for decades. Using a 2007-2018 mean
  underestimates 2028 BMI by ~1 kg/m², which propagates through
  BMI's IS RR and through the BMI ↔ LDL-C correlation we measured.
- **LDL-C** trended down (statin uptake); the trial cohort's
  baseline LDL-C is over-estimated by historical NHANES, biasing
  the trial-arm absolute LDL-C reduction expectation.
- **FPG** trended up; affects FPG's threshold-EMR contribution to
  diabetes mortality and the FPG → IS RR.
- **Smoking** see above — the headline case.

Lp(a) is genetically determined and shows essentially no secular
trend; the NHANES III Phase II 1991-94 estimate stays valid for
2028. LSM has only one NHANES cycle (P_LUX 2017-March 2020) so we
have no time-trend signal — but the underlying NAFLD epidemic
suggests LSM is increasing and FHS-style projection would be
defensible if FHS publishes it. Kidney dysfunction is GBD-pulled
already, so any GBD-published time trend is implicit in the
artifact.

Adopting FHS projections is a Model 5+ task because it doubles as
a project-wide "trial cohort = 2028 distribution" alignment, which
is bigger than swapping one CSV. Keep this README as the master
location for the FHS migration plan; the project repo's loader
docstrings cross-reference it.
