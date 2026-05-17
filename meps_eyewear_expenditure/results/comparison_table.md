# Results Comparison: Our Estimates vs. Li, Li & Sansgiry (2024)

**Data:** MEPS Full-Year Consolidated 2017–2021, pooled (annual averages)

**Survey design:** Taylor linearization with year-specific stratum IDs (`VARSTR_POOL = year + "_" + VARSTR`), PSU = VARPSU, weight = PERWTYYF / 5

## Descriptive Estimates

| Measure | Our estimate (95% CI) | Abstract (95% CI) | Notes |
|---|---|---|---|
| Persons with EE > 0 (annual, millions) | 59.6 (57.88–61.33) | 59.6 (57.3–61.9) | Point estimate identical; CIs overlap |
| Total annual EE ($B) | 21.82 (21.01–22.64) | 21.56 (20.80–22.19) | Our est +1.2% vs abstract; CIs overlap |
| Mean EE per capita ($) | 66.61 (64.73–68.48) | 66.61 (64.26–68.95) | Point estimate identical; confirms per-capita interpretation |
| Mean EE per spender ($) | 366.15 (358.31–373.99) | N/A (ambiguous) | ~$366; abstract $66.61 is per-capita not per-spender |

## Logistic Regression (Predictors of EE > 0)

Reference categories: age 18–44, Male, Hispanic, <HS education, poor/negative income, Uninsured.

| Predictor | Our aOR (95% CI) | Abstract aOR (95% CI) | Match? |
|---|---|---|---|
| NH White vs Hispanic | 1.17 (1.10–1.23) | 1.16 (1.09–1.23) | ✓ CI overlap |
| Female vs Male | 1.41 (1.36–1.46) | 1.40 (1.35–1.45) | ✓ CI overlap |
| Age 65+ vs 18-44 | 1.23 (1.17–1.29) | 1.47 (1.34–1.60) | ✗ Outside CI |
| Some college+ vs <HS | 1.57 (1.46–1.69) | 1.99 (1.80–2.19) | ✗ Outside CI |
| High income vs Poor | 1.38 (1.29–1.48) | 1.25 (1.17–1.33) | ✓ CI overlap |
| Private ins vs Uninsured | 2.00 (1.81–2.21) | 2.06 (1.84–2.29) | ✓ CI overlap |

## Notes

- aOR for Age 65+ and Some college+ fall outside the abstract's CIs.
  See `discrepancies.md` for hypotheses.
- Abstract does not specify reference categories; our choices are documented in `README.md`.