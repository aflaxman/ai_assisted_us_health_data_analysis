# NHANES Mortality by Liver Fibrosis Status — Pooled Analysis
### Pooled analysis: 10 NHANES cycles (1999–2018)

**Sample:** 59,064 eligible adults, 9,249 total deaths.
FIB-4 available for 52,335; high (≥2.67): 1,821;
low (<1.30): 37,524.

**Key findings from progressive matching:**

| Step | 24m RR (95% CI) | 36m RR (95% CI) |
|------|----------------|----------------|
| Crude | 14.07 [11.83–16.73] | 14.16 [12.3–16.31] |
| Step 4 (age+sex+BMI+SBP+smoking) | 1.68 [1.25–2.26] | 1.64 [1.28–2.1] |
| Step 5 (+ LDL-C, FPG) | 2.14 [1.31–3.5] | 2.03 [1.38–2.98] |

**Interpretation:**
1. The crude association (RR ~14.07×) reflects age confounding embedded in FIB-4.
2. Matching on age+sex produces the largest attenuation, as expected.
3. Additional metabolic covariates (BMI, SBP, smoking) produce modest further attenuation.
4. After full matching (Step 5), the residual association reflects fibrosis-specific mortality risk
   beyond metabolic confounding — pooling gives tighter CIs than individual-cycle analyses.

**Advantages of pooling:**
- ~1,821 high-FIB4 participants vs ~170–215 per cycle
- Enough deaths in both groups to produce meaningful confidence intervals
- Consistent variable names and definitions across all 10 cycles

**Limitations:**
1. Cycles span 20 years of enrollment — secular trends in mortality and disease
2. 2017-2018 has short follow-up (~48% censored before 24m)
3. UCOD coarsened for 2015–2018 cycles (full detail only for 1999–2014)
4. LDL-C and FPG restrict Step 5 to fasting subsample (~⅓)
5. No survey weights — unweighted, not nationally representative
