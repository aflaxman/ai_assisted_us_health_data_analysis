# Discrepancies from Li, Li & Sansgiry (2024)

Two of our logistic regression ORs fall outside the abstract's 95% CIs. Notes below.

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
