"""Side-by-side plot: colleague's CSV vs our F4-cal table vs empirical NHANES."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / 'data' / 'derived' / 'nhanes_p_lux.parquet'
COLLEAGUE = ROOT.parent / 'vivarium_csu_mace_rct' / 'src' / 'vivarium_csu_mace_rct' / 'data' / 'nhanes_rf_distributions.csv'
OURS = Path(__file__).parent / 'outputs' / 'liver_stiffness_age_sex_lognormal.csv'

F4 = 12.5

# load
df = pd.read_parquet(PARQUET)
ana = df[df['exam_complete'] & df['LSM_KPA'].notna()
         & df['WTMECPRP'].fillna(0).gt(0)].copy()

coll = pd.read_csv(COLLEAGUE).rename(columns={'Sex': 'sex'})
coll = coll[['age5', 'sex', 'LSM', 'LSM_sd']].copy()
coll['mean'] = np.exp(coll['LSM'] + coll['LSM_sd']**2 / 2)
coll['sd']   = coll['mean'] * np.sqrt(np.exp(coll['LSM_sd']**2) - 1.0)
coll['f4']   = 1 - stats.lognorm.cdf(F4, coll['LSM_sd'], scale=np.exp(coll['LSM']))

ours = pd.read_csv(OURS).rename(columns={'mean_kpa': 'mean', 'sd_kpa': 'sd',
                                          'f4_share_target': 'f4'})
ours['age5'] = ours['age_start'].astype(int)

# empirical per 5-yr bin
def w_mean(y, w): return float(np.average(y, weights=w))
def w_sd(y, w):
    m = w_mean(y, w)
    return float(np.sqrt(np.average((y - m)**2, weights=w)))

edges = list(range(25, 85, 5)) + [85]
rows = []
for sex in ['Female', 'Male']:
    for a0, a1 in zip(edges[:-1], edges[1:]):
        sub = ana[(ana['sex'] == sex)
                  & (ana['age_years'] >= a0)
                  & (ana['age_years'] < a1)]
        if len(sub) < 30:
            continue
        y = sub['LSM_KPA'].values.astype(float)
        w = sub['WTMECPRP'].values.astype(float)
        n_eff = (w.sum()**2) / (w**2).sum()
        rows.append({
            'sex': sex, 'age5': a0, 'n': len(sub),
            'mean': w_mean(y, w),
            'sd':   w_sd(y, w),
            'f4':   float(np.average((y >= F4).astype(float), weights=w)),
            'mean_se': w_sd(y, w) / np.sqrt(n_eff),
        })
emp = pd.DataFrame(rows)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
COLORS = {'colleague': '#cc6677', 'F4-cal (ours)': '#117733', 'empirical': 'black'}

for row, sex in enumerate(['Female', 'Male']):
    e = emp[emp['sex'] == sex].sort_values('age5')
    c = coll[coll['sex'] == sex].sort_values('age5')
    o = ours[ours['sex'] == sex].sort_values('age5')

    ax = axes[row, 0]
    ax.errorbar(e['age5'], e['mean'], yerr=1.96*e['mean_se'],
                fmt='o-', color=COLORS['empirical'], label='NHANES empirical (±1.96 SE)',
                capsize=3, markersize=5)
    ax.plot(c['age5'], c['mean'], 's--', color=COLORS['colleague'],
            label='colleague (log-moment match)', alpha=0.85)
    ax.plot(o['age5'], o['mean'], '^--', color=COLORS['F4-cal (ours)'],
            label='ours (F4-calibrated)', alpha=0.85)
    ax.set_ylabel(f'{sex}\n\nmean LSM (kPa)')
    ax.set_title(f'Arithmetic mean — {sex}')
    if row == 0:
        ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    ax = axes[row, 1]
    ax.plot(e['age5'], e['sd'], 'o-', color=COLORS['empirical'], markersize=5)
    ax.plot(c['age5'], c['sd'], 's--', color=COLORS['colleague'], alpha=0.85)
    ax.plot(o['age5'], o['sd'], '^--', color=COLORS['F4-cal (ours)'], alpha=0.85)
    ax.set_ylabel('SD of LSM (kPa)')
    ax.set_title(f'Arithmetic SD — {sex}')
    ax.grid(True, alpha=0.3)

    ax = axes[row, 2]
    # binomial SE for empirical F4 share
    p = e['f4'].values
    n_eff_emp = (e['n'].values).astype(float)  # rough — full SE needs survey design
    se_f4 = np.sqrt(p * (1 - p) / n_eff_emp)
    ax.errorbar(e['age5'], 100*p, yerr=100*1.96*se_f4,
                fmt='o-', color=COLORS['empirical'], capsize=3, markersize=5)
    ax.plot(c['age5'], 100*c['f4'], 's--', color=COLORS['colleague'], alpha=0.85)
    ax.plot(o['age5'], 100*o['f4'], '^--', color=COLORS['F4-cal (ours)'], alpha=0.85)
    ax.set_ylabel('P[LSM ≥ 12.5 kPa] (%)')
    ax.set_title(f'F4 share — {sex}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

for ax in axes[-1, :]:
    ax.set_xlabel('5-yr age band start')

fig.suptitle('NHANES P_LUX (2017 – Mar 2020) — colleague vs our F4-cal vs empirical',
             y=1.00, fontsize=12)
fig.tight_layout()
out = Path(__file__).parent / 'outputs' / 'lsm_comparison_colleague_vs_ours.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
print(f'wrote {out}')
