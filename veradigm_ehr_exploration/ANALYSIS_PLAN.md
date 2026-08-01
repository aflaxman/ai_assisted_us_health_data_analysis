# Analysis plan — cancer-related lymphedema

**Licensed cohort:** ~2.9M deidentified cancer patients (breast, melanoma, female
genital, male genital, urinary), 2020–2025, ambulatory EHR.
**Term:** license effective 9 July 2026, expires ~9 July 2027.
**Purpose:** non-commercial research and academic publication only.

Read `CLAUDE.md` in this directory first. This plan is deliberately written
without vendor table names, field names, documented fill rates, or layout, so it
is safe to commit. Working notes that do contain those details are kept
un-committed and gitignored.

## Why this cohort suits this question

The five licensed site groups are precisely the cancers whose treatment causes
lymphedema: breast and melanoma via axillary or inguinal node dissection, and
female genital, male genital, and urinary via pelvic node dissection. This is a
lymphedema cohort whether or not it was assembled as one, and at ~2.9M patients
**statistical power is not the binding constraint — ascertainment is.**

Two consequences of the cohort definition shape everything downstream:

**There is no non-cancer comparison group.** Every patient has cancer, so the
excess risk of lymphedema attributable to cancer treatment versus background
population risk is not estimable here. All contrasts must be internal: between
site groups, between treatment intensities, across geography, across time.
Any design phrased as "cancer versus non-cancer" is dead on arrival.

**The window is five years and starts at the COVID onset.** Lymphedema has a
12–36 month latency, so the analyzable incident cohort is essentially patients
whose treatment episode falls in the early part of the window with follow-up
through its end. Patients entering late contribute almost no at-risk time, and
patients treated before the window open appear as prevalent rather than incident
cases. 2020 is also anomalous for cancer screening, diagnosis, and treatment
timing, and cannot be treated as a normal baseline year.

## Design constraints inherited from the data

These are established in the un-committed working notes and restated here in
substance only.

1. **The exposure is largely invisible.** Extent of nodal surgery — the dominant
   risk factor, and roughly a fourfold gradient — happens in hospitals and is not
   reliably present in ambulatory records. Radiation is likewise mostly absent.
2. **Procedure dates in structured history are sparse.** For a disease defined by
   its latency after a procedure, the index date is the weakest link. Adjuvant
   endocrine therapy initiation is the most promising surrogate index date for
   breast cancer, and an analogous treatment-initiation anchor should be sought
   for each other site group.
3. **The outcome is chronically under-coded.** Published validation of
   administrative codes against measurement-based diagnosis suggests sensitivity
   well below 100%. A single diagnosis code is an indicator, not a measurement.
4. **Laterality is absent**, so the standard within-patient contralateral-limb
   comparison is unavailable.
5. **Body-mass index is right-censored at a threshold** that coincides with the
   class III obesity boundary — the range where lymphedema risk is highest.
   Recorded weight is censored far less severely and is measured repeatedly, so
   **weight trajectory, not BMI, is the exposure of record.**
6. **There is no payer field**, so insurance-based designs require an age proxy.
7. **Loss to follow-up is invisible.** There is no enrollment file, so "still
   under observation" must be defined explicitly and defended.

## Phase 0 — governance, before any analysis

No Data is touched in this phase.

1. Resolve the three open license questions in writing (see `CLAUDE.md`). The
   first one gates Phase 2: whether a latent-variable measurement model, used as
   an analysis method for a publication with no model deliverable, falls outside
   the ML and algorithm-development bar. **If that answer is no, Phase 2 must be
   replaced by a deterministic phenotype and the study loses its methodological
   core** — so ask early, not after building it.
2. Confirm whether any external benchmarking is permitted at all, including
   area-level deprivation linkage and comparison against published incidence.
   Assume not until told otherwise.
3. Stand up the code-only working directory so that assistant tooling cannot
   reach the Data root, and verify it.
4. Write the statistical analysis plan and register it before looking at
   outcomes.
5. Create the **results manifest** described in Phase 7. Start it now, not at the
   end.

## Phase 1 — feasibility, aggregate counts only

A go/no-go phase. Every output is a count or a distribution with small cells
suppressed. Expect this to take days, not weeks, and to change the plan.

1. Cohort size by site group and year; distribution of observation-window length
   per patient.
2. **Distribution of at-risk follow-up time after the surrogate index date.**
   This is the query most likely to kill the study. Run it first. If the median
   patient has under a year of post-treatment follow-up, the incidence design is
   not viable and the study becomes cross-sectional prevalence only.
3. Coded lymphedema frequency by site group, and how much of it is preceded
   rather than followed by the cancer record.
4. Availability of each of the five candidate outcome signals: diagnosis codes,
   therapy referrals, compression-garment history, manual-therapy and
   bioimpedance procedure codes, and antibiotic courses consistent with limb
   cellulitis. A signal that is absent or negligible is dropped here.
5. Size of the treatment-anchored subcohort per site group, versus the
   diagnosis-coded cohort. If the treatment anchor is much larger, it becomes the
   cohort definition.
6. Practice-mix diagnostics: how concentrated is the cohort across practices, and
   how much between-practice variation is there in each signal's frequency.
   This sizes the clustering problem before it contaminates a result.

**Decision gate.** Proceed to Phase 2 only if at least three outcome signals are
usable and the at-risk time distribution supports an incidence design. Otherwise
descope to prevalence and care patterns.

## Phase 2 — outcome phenotype

The methodological core, and the part that is gated on the Phase 0 answer.

1. Build a deterministic multi-signal phenotype first, as a floor: a patient is
   a case if any of the qualifying signals appears. Report its frequency
   alongside each single-signal frequency, and the agreement structure between
   signals.
2. Then, if permitted, fit a **latent-variable measurement model** treating the
   true condition as unobserved and each signal as an imperfect indicator with
   its own sensitivity and specificity. The usual conditional-independence
   assumption is clearly violated here — every signal depends on engagement with
   care — so specify it as a **two-mechanism model**: one mechanism governing
   care engagement, one governing the condition. Identification comes from the
   signals having genuinely different error structures, not from assuming them
   independent.
3. Validate the estimator on synthetic fixtures where truth is known by
   construction, before applying it. Report bias and coverage from that exercise
   as part of the eventual publication.
4. Sensitivity: refit under alternative dependence structures and under
   signal-dropping, and report how much the prevalence estimate moves. If it
   moves a lot, that is the finding.

## Phase 3 — descriptive epidemiology

1. Lymphedema frequency by site group, age, sex, race and ethnicity, geography,
   and calendar time, using both the deterministic and model-based phenotypes.
2. Time from surrogate index date to first qualifying signal, with explicit
   handling of left truncation for patients whose treatment predates the window.
3. **Under-ascertainment as the substantive result.** The gap between coded and
   model-estimated frequency, and how that gap varies across practices, regions,
   and patient groups. Differential under-coding is itself a health-equity
   outcome, because an uncoded patient is an unreferred and untreated patient.
   Compare to published incidence in the discussion narrative only — not as a
   statistical calibration step.
4. Lower-limb and genital lymphedema after the non-breast site groups, which are
   badly understudied because single-institution cohorts are too small. Expect
   worse ascertainment here, since the diagnosis coding is less specific than for
   the postmastectomy presentation. Report it as a distinct, more uncertain
   stratum rather than pooling.

## Phase 4 — risk factors and equity

1. **Weight trajectory** as the primary modifiable exposure, using repeated
   measurements rather than the censored index. Model post-treatment weight
   change as a time-varying covariate. State plainly that the risk gradient
   across the highest obesity category is not estimable without either an
   approved recovery of the censored index or an extrapolation declared as such.
2. Adjust for observation intensity throughout. Sicker patients visit more and
   therefore accrue more chances to be diagnosed; visit frequency is a confounder
   for every outcome here, not a nuisance.
3. Practice-level random effects on every geographic result. Practices differ in
   documentation habits far more than their patients differ biologically.
4. **Treatment equity conditional on diagnosis:** among patients with a
   qualifying lymphedema signal, who goes on to receive therapy referral and
   compression management? Conditioning on diagnosis sidesteps part of the
   ascertainment problem, though not differential coding itself, and is a
   cleaner design than an incidence-disparities analysis.
5. Geography enters as a bare stratifier. Area-deprivation linkage is a
   comingling question and stays out until answered in writing.

## Phase 5 — policy evaluation

Medicare compression-garment coverage began 1 January 2024, inside the data
window with roughly two years of follow-up. Because there is no payer field, use
age as the exposure proxy in a difference-in-differences design, with the younger
population as control.

Report this as exploratory. Three threats deserve pre-registration: the age
proxy is imperfect, the oldest age stratum is collapsed by de-identification, and
an ambulatory record captures garment provision only indirectly — so a null
result may reflect measurement rather than absence of effect.

## Phase 6 — complications

Rates of limb cellulitis among phenotype-positive patients, and whether
prophylaxis is prescribed. Modest in scope, clinically actionable, and it rests
on the best-coded exposure in the dataset.

## Phase 7 — freeze, extraction, closeout

**Destruction is irreversible and due within 5 days of study completion, so
every aggregate needed for publication must be extracted before the freeze.**
There is no going back for a forgotten table.

1. Maintain the **results manifest** from Phase 0 onward: one row per table,
   figure, and reported number, with the script that produces it and whether it
   has been extracted. Treat an unextracted entry as a blocking defect.
2. Analysis freeze with time to spare before term expiry. Working backward from
   ~9 July 2027, freeze by roughly May 2027.
3. Extract everything in the manifest, aggregated, with small-cell suppression.
   Review each output by eye before it leaves the environment.
4. Independent check that the manifest is complete — ideally by drafting the
   full paper skeleton against the extracted aggregates and confirming no figure
   or number is missing.
5. Destroy Data and backups; issue the certificate of destruction; retain
   audit-readiness for 6 months.
6. Write from the extracted aggregates. Cite the data source per the license,
   subject to resolving the citation question in `CLAUDE.md`. Data-availability
   statement: licensed from the vendor and available only via license from them.

## Ideas dropped, and why

Earlier planning for this project assumed a general-population extract and a
permissive licensing posture. Most of it does not survive contact with either the
actual cohort or the license, and is recorded here so it is not revived by
accident.

| Dropped | Reason |
|---|---|
| Liver fibrosis burden calibrated to national survey data | Comingling, and wrong cohort |
| Survey-to-EHR transport of unmeasured exposures | Comingling — the entire design is a merge |
| Hypertension control cascade benchmarked externally | Comingling, and wrong cohort |
| General-population GLP-1 diffusion | Wrong cohort; may be revivable as a cancer-survivor question |
| Kidney-function staging and equation-change study | Wrong cohort |
| Immunization uptake validated against public estimates | Comingling, and wrong cohort |
| Mortality completeness against external life tables | Comingling; internal completeness checks may survive |
| Area-deprivation linkage at coarse geography | Comingling |
| Smoking-status extraction validated against a public survey | Extraction is fine; the external validation step is comingling |
| **Lab-name normalization as a model benchmark** | **Explicitly barred: algorithm development as a deliverable (§1c)** |

The last row is the sharpest: it was framed as building a reusable crosswalk and
reporting model accuracy, which is exactly the deliverable the license prohibits.
Text normalization as an internal preprocessing step for a publication is a
different thing, and falls under the same §1c(x) gray area as Phase 2.

## Working notes not in this repo

Detailed feasibility notes, the vendor-specific schema config, the synthetic-data
generator, and the environment profiler all contain vendor schema detail and are
gitignored pending refactor. See the compliance status section of the working
notes for what needs to change before any of it can be committed.
