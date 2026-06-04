# Methods note — BRFSS income, grouped-data EM, and a two-mechanism missingness model

This note records exactly what was used, so the pipeline is reproducible without
trusting any remembered code. Every variable coding below was verified against the
official 2023 codebook (see **Sources**), not from memory.

## Data

- **Survey / year:** Behavioral Risk Factor Surveillance System (BRFSS) **2023**,
  the combined landline + cellphone (LLCP) public-use file.
- **File:** `LLCP2023.XPT` (SAS transport), from `LLCP2023XPT.zip` (CDC), cached at
  `data/raw/brfss/2023/`. Total records **433,323**.
- **Codebook:** `USCODE23_LLCP_021924.HTML` (from `codebook23_llcp-v2-508.zip`).
- **Calculated variables:** `2023-calculated-variables-version4-508.pdf`.

Notebook `01_download.ipynb` reproduces the published `INCOME3` and `MENTHLTH`
frequencies from the raw file as an integrity check before any recode.

## Variables

**Income — `INCOME3`** (Demographics core, Q15). Eleven brackets plus two distinct
missing codes. Dollar intervals used as the endpoints of the grouped-data likelihood
(on the natural-log scale; the bottom bracket is left-open at 0 → `-inf`, the top
bracket right-open → `+inf`):

| code | interval ($) | | code | interval ($) |
|---|---|---|---|---|
| 1 | [0, 10,000) | | 7 | [50,000, 75,000) |
| 2 | [10,000, 15,000) | | 8 | [75,000, 100,000) |
| 3 | [15,000, 20,000) | | 9 | [100,000, 150,000) |
| 4 | [20,000, 25,000) | | 10 | [150,000, 200,000) |
| 5 | [25,000, 35,000) | | 11 | [200,000, ∞) **open top** |
| 6 | [35,000, 50,000) | | | |

- **`77` = Don't know / Not sure** — n = 36,316 (8.54%).
- **`99` = Refused** — n = 42,232 (9.93%).
- **blank = Not asked / Missing** — n = 8,075 (dropped).

The two missing codes are stored **separately** in the raw item (the published
collapsed variables merge them); keeping them distinct is the point of the study.
The item wording — *"If respondent refuses at any income level, code Refused."* —
confirms the separation is by design.

**Outcomes (two, analysed identically).**

- **Frequent mental distress (FMD)** from `MENTHLTH` (Healthy Days core, Q2.1): days
  of poor mental health in the past 30. Coding: 1–30 = days; 88 = none (→ 0 days);
  77 = Don't know; 99 = Refused; blank = Not asked. FMD = `MENTHLTH ≥ 14`, the
  standard CDC construct. Weighted prevalence ≈ 15.6%.
- **Serious difficulty seeing** from `BLIND` (Disability core, Q9.2): *"Are you blind
  or do you have serious difficulty seeing, even when wearing glasses?"* 1 = Yes,
  2 = No, 7 = Don't know, 9 = Refused, blank = Not asked. Outcome = (`BLIND` = 1).
  Verified counts: Yes 22,190 / No 395,423 / DK 1,113 / Refused 517 / blank 14,080;
  weighted prevalence ≈ 5.7%.

In both cases the outcome's own 7/9/blank are dropped; the income missingness
structure is identical because it is the same `INCOME3` item.

**Covariates** (income and outcome models share the same design matrix): age
(`_AGE80`, continuous 18–80, standardized, with a quadratic), sex (`SEXVAR`),
race/ethnicity (`_IMPRACE`, the imputed 6-category variable — no missing),
education (`_EDUCAG`, 4 levels), employment (`EMPLOY1`, collapsed to
employed / unemployed / homemaker-or-student / retired / unable), marital status
(`MARITAL`). Rows with a missing covariate are dropped.

**Survey design:** final weight `_LLCPWT`, stratum `_STSTR`, PSU `_PSU`. `_PSU` is
unique within a state-year, so the ultimate sampling unit is the individual and the
design reduces to stratified sampling with unequal weights. All point estimates are
survey-weighted; all CIs are design-based (below).

**Analysis sample:** 408,944 rows after dropping income "not asked", missing
outcome, and missing covariates — 338,056 bracketed, 33,430 don't-know, 37,458
refused. Weighted FMD prevalence ≈ 15.6%.

## Models

Let `I` = latent log household income, `X` = covariates (with intercept), `Y` = the
binary outcome (frequent mental distress, or difficulty seeing).

**Measurement model (the grouped-data part).** `I | X ~ Normal(Xβ, σ²)` (lognormal
income with a covariate-dependent mean). Each bracketed respondent contributes the
interval probability `Φ((U−Xβ)/σ) − Φ((L−Xβ)/σ)` — the grouped-continuous-data
likelihood (Heitjan 1989). The open top bracket contributes the proper right-tail
mass `1 − Φ((ln 200k − Xβ)/σ)`; **no finite midpoint is assigned**. Estimated by EM
(truncated-normal E-step, weighted-least-squares M-step) on the bracketed
respondents alone, so the income model is identified entirely by respondents who
reported a bracket. Estimated `σ ≈ 0.69`.

**Outcome model.** `logit P(Y=1) = θ·I + Xη`, with `I` latent. We integrate `I` out
against its conditional distribution using an equal-probability quadrature grid
(K = 32 nodes) and maximise the marginal likelihood by Newton steps with the analytic
Louis observed-information matrix. The **gradient** is `θ` (log-OR per unit log
income); we report `exp(θ·ln 2)` = the **odds ratio per doubling of income**.

**A / B / C** differ only in how each respondent's latent income enters the outcome
model:

- **A — midpoint + listwise.** Each bracket → its midpoint (bottom → \$5,000, open
  top → \$250,000); a single quadrature node at that point; bracketed respondents
  only. The standard-practice baseline.
- **B — grouped + listwise.** Bracketed respondents only, income integrated over the
  bracket-truncated lognormal posterior.
- **C — grouped + two mechanisms.** Adds the missing respondents: **don't-know** as
  MAR given covariates (income ~ full conditional `N(Xβ, σ²)`), and **refused** as
  MNAR via an exponential tilt `exp(γ·I)` on the conditional income distribution,
  which shifts refusers' mean log-income by `Δ = γσ²`. `γ = 0` reduces refused to
  MAR. `γ` is **not identified** from the data; notebook 04 sweeps it and reports the
  gradient as a function of the assumed shift (refusers earning ½× to 2× their
  covariate-predicted income). This is a pattern-mixture / selection sensitivity
  analysis in the Little tradition — explicitly *not* an estimate of the refusal
  mechanism.

**Design-based inference.** Primary 95% CIs are a stratified Taylor-linearised
sandwich on the outcome estimating equations (the standard survey-GLM variance, as
in `svyglm`), holding the income measurement model fixed. Because the income model
is pinned down by 338k bracketed respondents, fixing it understates variance only
negligibly — confirmed by a stratified bootstrap (resampling respondents within
`_STSTR`, refitting income *and* outcome models, warm-started), whose SEs agree with
the linearized ones. No naive IID standard errors are reported anywhere.

## Results

Authoritative figures live in `00_overview.ipynb` and the derived CSVs. Gradient =
**odds ratio of the outcome per doubling of household income**, design-based
(linearized) 95% CI.

| model | Frequent mental distress | Difficulty seeing |
|---|---|---|
| A. midpoint + listwise | 0.853 [0.833, 0.872] | 0.755 [0.727, 0.785] |
| B. grouped + listwise | 0.836 [0.816, 0.857] | 0.728 [0.697, 0.760] |
| C. grouped + two-mech (γ=0) | 0.837 [0.817, 0.858] | 0.727 [0.696, 0.760] |

(FMD: 338,056 bracketed / 408,944 total; vision: 338,342 / 409,359.)

**Did honoring the grouping matter (A → B)?** Yes, consistently, and in a predictable
direction: it **strengthens** the gradient. Replacing bracket midpoints (especially
the arbitrary \$250k value pinned on the open top bracket) with the grouped-data
lognormal measurement model lowers the OR per doubling from 0.853 to 0.836 for FMD
(−1.9%) and from 0.755 to 0.728 for difficulty seeing (−3.7%). The midpoint injects
income measurement error that attenuates the slope toward the null (classic
regression dilution); honoring the interval censoring removes it. The shift is larger
for the outcome with the steeper underlying gradient (vision).

**Did separating and modeling the two mechanisms matter (B → C)?** At the ignorable
assumption (don't-know MAR, refused γ=0), essentially **no** — the gradient moves
< 0.2% for both outcomes. The ~338k bracketed respondents already identify the
gradient tightly, so adding the ~70k don't-know/refused respondents under MAR barely
moves it. The value of model C is the honesty of its **sensitivity analysis**, not a
point shift.

**Under what refusal assumption does the conclusion change?** None that is plausible.
Sweeping the refusers' assumed income from ½× to 2× their covariate-predicted value,
the gradient stays in a narrow band — FMD 0.827–0.868, vision 0.726–0.759 — and never
approaches 1. Assuming refusers are *lower*-income (the MIHA pattern) attenuates the
gradient slightly; assuming they are *higher*-income strengthens it slightly. In no
case does the substantive conclusion (higher income → lower risk) flip or lose
significance.

**Same general pattern across outcomes?** Yes. Both outcomes show: a clear protective
income gradient, a modest A→B strengthening from honoring the grouping, a negligible
B→C change at MAR, and robustness of the conclusion across the refusal sweep. The
effect of the measurement model is *larger* where the gradient is steeper. The
practical lesson: for an income–health gradient on BRFSS, the bracket-midpoint
shortcut biases the slope toward the null by a few percent (worth fixing with the
grouped-data likelihood), while the refused-vs-don't-know distinction — though real
in the demographics of who is missing — does not move the gradient unless one is
willing to assume an implausibly strong, directional refusal mechanism.

## Sources

- BRFSS 2023 annual data: https://www.cdc.gov/brfss/annual_data/annual_2023.html
- Codebook (`USCODE23_LLCP_021924.HTML`) and Calculated Variables report, 2023.
- Heitjan, D.F. (1989). Inference from Grouped Continuous Data: A Review.
  *Statistical Science* 4(2):164–179.
- Little, R.J.A. & Rubin, D.B. *Statistical Analysis with Missing Data.*
- Riphahn, R.T. & Serfling, O. (2005). Item non-response on income and wealth
  questions. *Empirical Economics.*
- Tourangeau, R. & Yan, T. (2007). Sensitive questions in surveys. *Psych. Bulletin.*
- Louis, T.A. (1982). Finding the observed information matrix when using the EM
  algorithm. *JRSS-B* 44(2):226–233.
