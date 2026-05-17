## Temporal Trend Regressions: log(expenditure) ~ year

All regressions: log-linear OLS with HC3 heteroscedasticity-robust SEs.

| Series | Years | N | Growth rate (%/yr) | 95% CI | R² | p |
|---|---|---|---|---|---|---|
| FRED total (2002–2021) | 2002–2021 | 20 | +3.07% | (+2.47% – +3.67%) | 0.904 | 0.0000 |
| MEPS total, all years (2002–2022) | 2002–2022 | 21 | +4.74% | (+4.01% – +5.47%) | 0.900 | 0.0000 |
| MEPS total, pre-2018 (2002–2017) | 2002–2017 | 16 | +3.43% | (+2.81% – +4.05%) | 0.889 | 0.0000 |
| MEPS total, post-2018 (2018–2022) | 2018–2022 | 5 | +2.72% | (-5.66% – +11.84%) | 0.196 | 0.5364 |
| FRED per-capita (2002–2021) | 2002–2021 | 20 | +2.30% | (+1.72% – +2.89%) | 0.852 | 0.0000 |
| MEPS per-capita, all years (2002–2022) | 2002–2022 | 21 | +3.98% | (+3.23% – +4.74%) | 0.858 | 0.0000 |

**Note on MEPS level shift around 2018:** The per-year total jumps from ~$18B (2017)
to ~$24B (2018) and stays elevated. MEPS-HC sampling has remained an NHIS subsample
throughout (confirmed through HC-251, 2023 data); the cause of the level shift is
unknown from the data alone. The NHIS questionnaire was redesigned for the 2019
data year (instrument changes piloted from Spring 2018 in the MEPS field period),
and the NHIS sample design changed in 2016 with effects propagating to MEPS over
subsequent panels. Other candidates include real spending growth driven by low-cost
online retailers and changes in MEPS imputation or weighting. Pre/post regressions
split at 2018 to isolate each era without asserting a cause.