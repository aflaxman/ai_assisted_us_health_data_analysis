# Discrepancies from Li, Li & Sansgiry (2024)

Two of our logistic regression ORs fall outside the abstract's 95% CIs. Notes below.

---

## Correction: MEPS sampling frame

An earlier version of this analysis incorrectly stated that MEPS switched from
NHIS-based to ACS-based sampling around 2017. This was wrong.

Per the HC-251 (2023 Full Year Consolidated) documentation released August 2025,
MEPS-HC continues to be drawn as a subsample of the prior year's NHIS:

> "The set of households selected for each panel of the MEPS HC is a subsample of
> households participating in the previous year's National Health Interview Survey
> (NHIS) … The NHIS sampling frame provides a nationally representative sample of
> the U.S. civilian noninstitutionalized population."

The ~$6B level shift visible in MEPS between 2017 ($17.7B) and 2018 ($23.7B)
is driven almost entirely by a jump in *prevalence* of eyewear spenders, not by
the per-spender amount. The weighted share of persons with VISEXP > 0 rises from
16.3% (2017) to 20.0% (2018) and then stays in the 17–19% range through 2022,
while the mean among spenders drifts smoothly from $339 (2017) to $364 (2018) —
no per-spender break.

A direct comparison of the MEPS Other Medical Expenses (OM) CAPI sections for
2017 (Nov 2017 specification; reflecting the harmonized 2016 design) and 2018
(P21R5/P22R3/P23R1 specification; the first year of the Spring 2018 redesign)
points to one specific instrument change as the most likely cause:

**Pre-redesign (2017 PUF, harmonized to 2016 design):** the OM section had no
direct purchase-screening question for eyewear. A respondent was routed into
OM01A/OM01B only if an eyewear item type had already been identified upstream
(through the Event Driver / EV20 chain — i.e., the purchase had to surface via
some other event such as a medical visit). OM01A then asked:

> "Of the times (PERSON) obtained glasses or contact lenses since (START DATE),
> how many were during {YEAR}?"

**Post-redesign (2018 PUF, OM item OM10, BLAISE name "Glasses"):** every
household member is directly screened with a YES/NO question:

> "Did {you/{PERSON}} purchase eyeglasses or contact lenses {since {START DATE}/
> between {START DATE} and {END DATE}}?"

If OM10 is coded YES, a new OM-record is created at BOX_30. Five additional
direct YES/NO screens were added in the same redesign for other OM categories
(home health care receiver, ambulance, disposable supplies, medical equipment,
and a residual "other OM" item), but the eyeglasses/contacts screen is the
relevant one here.

This change converts eyewear capture from an *event-triggered* to a
*direct-screened* design. The new design naturally identifies more eyewear
purchasers — drugstore reading glasses, online contact-lens refills, glasses
replaced without a new exam, and purchases the respondent simply did not think
to mention while reporting medical visits would all be missed by the old
event-driven flow and caught by the new direct prompt.

The MEPS HC-201 (2017) documentation acknowledges the partial transition: for
the Full-Year 2017 PUFs, "the Panel 22 Round 3 and Panel 21 Round 5 data were
transformed to the degree possible to conform to the previous year (2016)
design." 2018 is the first PUF that fully reflects the new instrument, which
is exactly where the prevalence jump appears in the data.

Auxiliary candidates (real online-retailer growth, NHIS 2016 sample-design
effects propagating through later panels, MEPS imputation/calibration changes)
remain plausible second-order contributors but cannot produce the specific
fingerprint observed: a one-year ~3.7-pp jump in spender prevalence with no
break in mean per-spender. The pre/post trend regressions in
`code/08_trend_regression.py` split at 2018 to isolate the two instrument
regimes empirically.

---

## 1. Age 65+ aOR: ours 1.23 (1.17–1.29) vs abstract 1.47 (1.34–1.60)

**Direction:** Both > 1 (older adults more likely to have EE). Ours is substantially lower.

**Hypotheses:**

1. **Different reference group.** The abstract does not specify a reference category for age.
   If the abstract used `<18` as the reference (instead of our `18–44`), then the 65+ aOR
   vs `<18` would be higher. In our model, `age_lt18` has OR=0.87 vs 18–44, so
   65+ vs <18 ≈ 1.23 / 0.87 ≈ 1.41 — closer to 1.47 but still outside the CI.

2. **Different age groupings.** The abstract might use a different cut point (e.g., 65–74,
   75+ combined; or 55+ as "older"). Pooling 45–74 together and comparing to 18–44
   would yield an intermediate OR.

3. **Continuous age.** If the abstract fit age as a continuous OR per decade, the
   reported OR of 1.47 might represent a ~3-decade increment (18→65), which
   would equal our exp(3 * per_decade_beta). With a log-linear age effect, our overall
   pattern is consistent.

4. **Interaction with other variables.** If the abstract's model excluded children from
   the analytic file (18+ only), the reference distribution shifts, potentially
   increasing the 65+ OR. An 18+-only model also changes the education reference category.

**Recommendation:** Re-run the regression restricting to adults 18+ with age as a continuous
variable or using 18–34 as the youngest reference group.

---

## 2. Some college+ aOR: ours 1.57 (1.46–1.69) vs abstract 1.99 (1.80–2.19)

**Direction:** Both > 1 (higher education → more EE). Ours is substantially lower.

**Hypotheses:**

1. **Finer education grouping.** Our "some college+" combines HIDEG=4 (bachelor's),
   HIDEG=5 (master's), HIDEG=6 (doctoral), and HIDEG=7 (associate's/some college).
   If the abstract defines "higher education" as bachelor's degree or higher (HIDEG=4–6),
   excluding associate's/some college, the OR vs <HS would be larger. Associate's/some
   college holders have lower EE rates than bachelor's holders, diluting our group OR.

2. **Different reference.** If the abstract uses "HS diploma" (not <HS) as the reference,
   the "some college+" comparison would be different. However, this would likely reduce
   the OR, not increase it.

3. **Adult-only model.** Including children in the logistic regression with a "Child (<18)"
   education category can distort the education ORs through its association with age.
   If the abstract restricted to adults (18+) and simply omitted the education term for
   those under 25 or those currently in school, the ORs for adult education would shift.

4. **Year fixed effects.** The abstract may include year fixed effects not in our model,
   which could alter education ORs if educational attainment trends differ across 2017–2021.

**Recommendation:** Re-run with HIDEG=4–6 (bachelor's+) as "higher education" and restrict
to adults 25+ (the conventional age for stable educational attainment).

---

## 3. Total annual EE: ours $21.82B vs abstract $21.56B (+1.2%)

Not a meaningful discrepancy — the CIs overlap substantially and 1.2% difference is within
normal rounding / vintage-update variation. The abstract's CI $20.80–22.19B contains our
point estimate $21.82B.

---

## 4. Abstract reports "45.77% age 65+" among EE > 0 spenders

Our estimate: **19.8%** of EE > 0 spenders are age 65+. The abstract's 45.77% is not
plausible as the share of spenders who are 65+ (the 65+ US population share is ~17%).

**Hypotheses:**
- The 45.77% likely refers to a different metric, possibly the *prevalence* of any EE
  among those age 45+ or 65+ (not the *share* of spenders in that age group).
- Alternatively the abstract has a typo and 45.77% is the combined 45–64 + 65+ share
  (our estimate: 32.5% + 19.8% = 52.3% — still not 45.77%, but closer).
- The 45.77% might refer to "at least one eyecare visit" rather than eyewear expenditure.

Our 58.21% female matches the abstract exactly, suggesting our overall methodology is correct.
The age discrepancy is likely a labeling issue in the abstract, not an analytic error.
