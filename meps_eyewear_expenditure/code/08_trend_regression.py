"""
Temporal trend analysis: MEPS annual eyewear expenditures vs BEA HCSA (FRED).

Regressions:
  1. FRED total ($B) ~ year           (2002–2021, log-linear OLS)
  2. MEPS total ($B) ~ year           (2002–2022, log-linear OLS; all years)
  3. MEPS pre-redesign  ~ year        (2002–2016)
  4. MEPS post-redesign ~ year        (2017–2022)
  5. MEPS per-capita    ~ year        (full series)

All regressions use log(y) ~ year so the slope is an annual growth rate.
Heteroscedasticity-consistent (HC3) standard errors throughout.

Outputs:
  outputs/trend_comparison.png    — multi-panel figure
  results/trend_regressions.csv   — regression coefficients and CIs
  results/trend_regressions.md    — markdown table
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hc3

DERIV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'derived'))
OUT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
RES   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(OUT, exist_ok=True); os.makedirs(RES, exist_ok=True)

# ── BEA HCSA data (FRED COEYEQEXPHCSA / COEYEQPCHCSA, 2000–2021) ─────────────
BEA = pd.DataFrame({
    'year': list(range(2000, 2022)),
    'total_b': [19.91, 18.80, 20.26, 21.10, 22.20, 23.77, 24.91, 27.11,
                27.46, 28.24, 29.84, 31.61, 31.62, 32.08, 32.41, 32.61,
                32.41, 32.57, 34.05, 35.66, 33.90, 41.57],
    'percap':  [70.56, 65.97, 70.44, 72.73, 75.82, 80.44, 83.48, 90.00,
                90.30, 92.06, 96.86, 101.86, 101.15, 101.92, 102.22,
                102.09, 100.73, 100.59, 104.61, 109.03, 102.60, 125.63],
})

MEPS = pd.read_parquet(os.path.join(DERIV, 'meps_annual_eyewear.parquet'))

# ── Regression helper ──────────────────────────────────────────────────────────
def log_linear_reg(df: pd.DataFrame, y_col: str, year_col: str = 'year',
                   label: str = '') -> dict:
    """
    Fit log(y) ~ year with HC3 robust SEs.
    Returns dict with slope, intercept, SE, CIs, R², predicted series.
    """
    sub = df[[year_col, y_col]].dropna().copy()
    sub = sub[sub[y_col] > 0]
    y    = np.log(sub[y_col].values)
    yr   = sub[year_col].values.astype(float)
    X    = sm.add_constant(yr)
    res  = sm.OLS(y, X).fit(cov_type='HC3')
    slope_pct = (np.exp(res.params[1]) - 1) * 100     # % per year
    ci = res.conf_int()
    ci = pd.DataFrame(ci) if not isinstance(ci, pd.DataFrame) else ci
    slope_ci_lo = (np.exp(ci.iloc[1, 0]) - 1) * 100
    slope_ci_hi = (np.exp(ci.iloc[1, 1]) - 1) * 100
    # Predicted values on original scale
    yr_pred = np.arange(sub[year_col].min(), sub[year_col].max() + 1)
    y_pred  = np.exp(res.params[0] + res.params[1] * yr_pred)
    return {
        'label':       label,
        'n':           len(sub),
        'years':       f'{int(sub[year_col].min())}–{int(sub[year_col].max())}',
        'slope_pct':   slope_pct,
        'slope_lo':    slope_ci_lo,
        'slope_hi':    slope_ci_hi,
        'r2':          res.rsquared,
        'p_slope':     res.pvalues[1],
        'yr_pred':     yr_pred,
        'y_pred':      y_pred,
        'intercept':   res.params[0],
        'slope_raw':   res.params[1],
    }


# ── Run regressions ────────────────────────────────────────────────────────────
meps_overlap = MEPS[MEPS['year'] <= 2021]   # years with FRED counterpart

regs = {
    'FRED total (2002–2021)':
        log_linear_reg(BEA[BEA['year'] >= 2002], 'total_b', label='FRED (BEA HCSA)'),
    'MEPS total, all years (2002–2022)':
        log_linear_reg(MEPS, 'total_b', label='MEPS (all years)'),
    'MEPS total, pre-redesign (2002–2016)':
        log_linear_reg(MEPS[MEPS['year'] <= 2016], 'total_b', label='MEPS pre-2017'),
    'MEPS total, post-redesign (2017–2022)':
        log_linear_reg(MEPS[MEPS['year'] >= 2017], 'total_b', label='MEPS 2017+'),
    'FRED per-capita (2002–2021)':
        log_linear_reg(BEA[BEA['year'] >= 2002], 'percap', label='FRED per-capita'),
    'MEPS per-capita, all years (2002–2022)':
        log_linear_reg(MEPS, 'percap', label='MEPS per-capita (all)'),
}

print('=' * 75)
print(f"{'Regression':<42} {'Slope %/yr':>10} {'95% CI':>20} {'R²':>6} {'p':>8}")
print('=' * 75)
for name, r in regs.items():
    print(f"{name:<42} {r['slope_pct']:>9.2f}%  "
          f"({r['slope_lo']:+.2f}%–{r['slope_hi']:+.2f}%)  "
          f"{r['r2']:>5.3f}  {r['p_slope']:>8.4f}")


# ── Build figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
ax1 = fig.add_subplot(gs[0, :])   # top: total $B, full width
ax2 = fig.add_subplot(gs[1, 0])   # bottom-left: per capita
ax3 = fig.add_subplot(gs[1, 1])   # bottom-right: growth-rate bar chart

MEPS_COLOR   = '#E8631A'
BEA_COLOR    = '#1E6EB4'
MEPS_PRE     = '#E8631A'
MEPS_POST    = '#C0392B'

# — Panel 1: total $B ——————————————————————————————————————————————————————————
# Data points
ax1.errorbar(BEA['year'], BEA['total_b'], fmt='o-', color=BEA_COLOR,
             lw=1.5, ms=5, label='BEA HCSA (FRED)', zorder=3)
ax1.errorbar(MEPS['year'],
             MEPS['total_b'],
             yerr=[MEPS['total_b'] - MEPS['total_lci'],
                   MEPS['total_uci'] - MEPS['total_b']],
             fmt='s', color=MEPS_COLOR, ms=5, capsize=3,
             lw=1, label='MEPS survey (this analysis)', zorder=4)

# Regression lines (log-linear fits)
r_fred = regs['FRED total (2002–2021)']
ax1.plot(r_fred['yr_pred'], r_fred['y_pred'], '--', color=BEA_COLOR,
         lw=1.5, alpha=0.7,
         label=f"FRED trend: {r_fred['slope_pct']:+.1f}%/yr (R²={r_fred['r2']:.2f})")

r_pre  = regs['MEPS total, pre-redesign (2002–2016)']
ax1.plot(r_pre['yr_pred'], r_pre['y_pred'], '--', color='#F5A623',
         lw=1.5, alpha=0.8,
         label=f"MEPS 2002–2016: {r_pre['slope_pct']:+.1f}%/yr (R²={r_pre['r2']:.2f})")

r_post = regs['MEPS total, post-redesign (2017–2022)']
ax1.plot(r_post['yr_pred'], r_post['y_pred'], '--', color=MEPS_POST,
         lw=1.5, alpha=0.8,
         label=f"MEPS 2017–2022: {r_post['slope_pct']:+.1f}%/yr (R²={r_post['r2']:.2f})")

ax1.axvline(2016.5, color='gray', lw=1, ls=':', alpha=0.7)
ax1.text(2016.7, 12, 'MEPS redesign\n(2017)', fontsize=7.5, color='gray', va='bottom')
ax1.set_xlabel('Year')
ax1.set_ylabel('Total eyewear expenditures ($B, nominal)')
ax1.set_title('Total US Eyewear Expenditures: MEPS vs BEA HCSA (FRED)', fontsize=11)
ax1.legend(fontsize=8, loc='upper left')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}B'))

# — Panel 2: per capita ————————————————————————————————————————————————————————
ax2.errorbar(BEA['year'], BEA['percap'], fmt='o-', color=BEA_COLOR,
             lw=1.5, ms=4, label='BEA HCSA (FRED)')
ax2.errorbar(MEPS['year'],
             MEPS['percap'],
             yerr=[MEPS['percap'] - MEPS['percap_lci'],
                   MEPS['percap_uci'] - MEPS['percap']],
             fmt='s', color=MEPS_COLOR, ms=4, capsize=3,
             lw=1, label='MEPS survey')
r_bpc = regs['FRED per-capita (2002–2021)']
ax2.plot(r_bpc['yr_pred'], r_bpc['y_pred'], '--', color=BEA_COLOR, lw=1.2, alpha=0.6,
         label=f"FRED trend: {r_bpc['slope_pct']:+.1f}%/yr")
r_mpc = regs['MEPS per-capita, all years (2002–2022)']
ax2.plot(r_mpc['yr_pred'], r_mpc['y_pred'], '--', color=MEPS_COLOR, lw=1.2, alpha=0.6,
         label=f"MEPS trend: {r_mpc['slope_pct']:+.1f}%/yr")
ax2.axvline(2016.5, color='gray', lw=1, ls=':', alpha=0.7)
ax2.set_xlabel('Year')
ax2.set_ylabel('Per capita ($, nominal)')
ax2.set_title('Per Capita Eyewear Expenditures')
ax2.legend(fontsize=7.5)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}'))

# — Panel 3: growth rate comparison bar chart ——————————————————————————————————
labels_bar = ['FRED\n(2002–21)', 'MEPS\n(2002–22)', 'MEPS pre\n(2002–16)',
              'MEPS post\n(2017–22)']
reg_keys   = ['FRED total (2002–2021)', 'MEPS total, all years (2002–2022)',
              'MEPS total, pre-redesign (2002–2016)',
              'MEPS total, post-redesign (2017–2022)']
slopes  = [regs[k]['slope_pct']  for k in reg_keys]
lo_errs = [regs[k]['slope_pct'] - regs[k]['slope_lo'] for k in reg_keys]
hi_errs = [regs[k]['slope_hi']  - regs[k]['slope_pct'] for k in reg_keys]
colors_bar = [BEA_COLOR, MEPS_COLOR, '#F5A623', MEPS_POST]

x_pos = np.arange(len(labels_bar))
bars = ax3.bar(x_pos, slopes, color=colors_bar, width=0.55, alpha=0.85,
               yerr=[lo_errs, hi_errs], capsize=5, error_kw={'lw': 1.5})
ax3.axhline(0, color='k', lw=0.8)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels_bar, fontsize=8)
ax3.set_ylabel('Annual growth rate (%/yr, log-linear OLS)')
ax3.set_title('Annual Growth Rates\nwith 95% CI (HC3)')
for bar, slope in zip(bars, slopes):
    ax3.text(bar.get_x() + bar.get_width()/2, slope + (0.1 if slope >= 0 else -0.3),
             f'{slope:+.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

fig.suptitle(
    'US Eyewear Expenditure Trends: MEPS Survey vs BEA HCSA (FRED)\n'
    'Sources: MEPS Full-Year Consolidated 2002–2022; FRED COEYEQEXPHCSA/COEYEQPCHCSA',
    fontsize=10, y=1.01)

fig_path = os.path.join(OUT, 'trend_comparison.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nFigure saved to {fig_path}')


# ── Save regression table ──────────────────────────────────────────────────────
reg_rows = []
for name, r in regs.items():
    reg_rows.append({
        'Regression':         name,
        'N_years':            r['n'],
        'Years':              r['years'],
        'Slope_%_per_yr':     round(r['slope_pct'], 3),
        'Slope_CI_lo':        round(r['slope_lo'], 3),
        'Slope_CI_hi':        round(r['slope_hi'], 3),
        'R2':                 round(r['r2'], 4),
        'p_slope':            round(r['p_slope'], 5),
    })

reg_df = pd.DataFrame(reg_rows)
csv_path = os.path.join(RES, 'trend_regressions.csv')
reg_df.to_csv(csv_path, index=False)

md_lines = [
    '## Temporal Trend Regressions: log(expenditure) ~ year',
    '',
    'All regressions: log-linear OLS with HC3 heteroscedasticity-robust SEs.',
    '',
    '| Series | Years | N | Growth rate (%/yr) | 95% CI | R² | p |',
    '|---|---|---|---|---|---|---|',
]
for r in reg_rows:
    md_lines.append(
        f"| {r['Regression']} | {r['Years']} | {r['N_years']} "
        f"| {r['Slope_%_per_yr']:+.2f}% "
        f"| ({r['Slope_CI_lo']:+.2f}% – {r['Slope_CI_hi']:+.2f}%) "
        f"| {r['R2']:.3f} | {r['p_slope']:.4f} |"
    )
md_lines += [
    '',
    '**Note on MEPS structural break:** MEPS was redesigned in 2017 (new sampling frame,',
    'ACS-based selection replacing NHIS-based). The per-year total jumps from ~$16B (2016)',
    'to ~$18B (2017) and ~$24B (2018). Separate pre/post regressions isolate each era.',
]

md_path = os.path.join(RES, 'trend_regressions.md')
with open(md_path, 'w') as f:
    f.write('\n'.join(md_lines))

print(f'Regression table saved to {csv_path} and {md_path}')
print()
print('\n'.join(md_lines[:15]))
