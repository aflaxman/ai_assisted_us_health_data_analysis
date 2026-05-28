"""Re-run PART A with pooled P_LUX + LUX_L. Does pooling close the gap?

We compare three reconstructions of the colleague's columns:
  (A) P_LUX only                — single cycle, MEC pre-pandemic weight
  (B) LUX_L only                — single cycle, MEC 2-yr weight
  (C) pooled                    — stack both, halve each cycle's MEC weight
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / 'data' / 'derived'
COLLEAGUE = ROOT.parent / 'vivarium_csu_mace_rct' / 'src' / 'vivarium_csu_mace_rct' / 'data' / 'nhanes_rf_distributions.csv'

pooled = pd.read_parquet(DERIVED / 'nhanes_p_lux_plus_l.parquet')
m = pooled['exam_complete'].fillna(False) & pooled['LSM_KPA'].notna() & pooled['MEC_WT'].fillna(0).gt(0)
ana = pooled[m].copy()

# halve each cycle's weight for the pooled "average" — common NCHS practice
ana['MEC_WT_POOL'] = ana['MEC_WT'] * 0.5


def w_mean(y, w): return float(np.average(y, weights=w))
def w_sd(y, w):
    mu = w_mean(y, w)
    return float(np.sqrt(np.average((y - mu) ** 2, weights=w)))


edges = list(range(25, 85, 5)) + [85]


def per_bin(d, w_col):
    rows = []
    for sex in ['Female', 'Male']:
        for a0, a1 in zip(edges[:-1], edges[1:]):
            sub = d[(d['sex'] == sex)
                    & (d['age_years'] >= a0)
                    & (d['age_years'] < a1)]
            if len(sub) < 5:
                continue
            y = sub['LSM_KPA'].values.astype(float)
            w = sub[w_col].values.astype(float)
            log_y = np.log(y)
            rows.append({
                'sex': sex, 'age5': a0, 'n': len(sub),
                'log_mean': w_mean(log_y, w),
                'log_sd':   w_sd(log_y, w),
            })
    return pd.DataFrame(rows)


tab_A = per_bin(ana[ana['cycle'] == '2017_2020'], 'MEC_WT').add_suffix('_A')
tab_B = per_bin(ana[ana['cycle'] == '2021_2023'], 'MEC_WT').add_suffix('_B')
tab_C = per_bin(ana, 'MEC_WT_POOL').add_suffix('_C')

# colleague's columns
coll = pd.read_csv(COLLEAGUE).rename(columns={'Sex': 'sex'})[['age5', 'sex', 'LSM', 'LSM_sd']]
coll = coll.rename(columns={'LSM': 'log_mean_coll', 'LSM_sd': 'log_sd_coll'})

# unify keys (the suffix-renamed tables still have sex_A / age5_A — undo just enough)
for t, sfx in [(tab_A, '_A'), (tab_B, '_B'), (tab_C, '_C')]:
    t.rename(columns={f'sex{sfx}': 'sex', f'age5{sfx}': 'age5'}, inplace=True)

m = coll.merge(tab_A, on=['sex', 'age5'], how='left') \
        .merge(tab_B, on=['sex', 'age5'], how='left') \
        .merge(tab_C, on=['sex', 'age5'], how='left')

m['d_log_mean_A'] = m['log_mean_A'] - m['log_mean_coll']
m['d_log_mean_B'] = m['log_mean_B'] - m['log_mean_coll']
m['d_log_mean_C'] = m['log_mean_C'] - m['log_mean_coll']
m['d_log_sd_A']   = m['log_sd_A']   - m['log_sd_coll']
m['d_log_sd_B']   = m['log_sd_B']   - m['log_sd_coll']
m['d_log_sd_C']   = m['log_sd_C']   - m['log_sd_coll']

print('log_mean — recomputed minus colleague')
cols = ['sex', 'age5', 'n_A', 'n_B', 'd_log_mean_A', 'd_log_mean_B', 'd_log_mean_C']
print(m[cols].round(3).to_string(index=False))

print()
print('log_sd — recomputed minus colleague')
cols = ['sex', 'age5', 'n_A', 'n_B', 'd_log_sd_A', 'd_log_sd_B', 'd_log_sd_C']
print(m[cols].round(3).to_string(index=False))

print()
print('=' * 80)
print('RMS deviation across the 24 cells, for each candidate sample:')
for s in ['A', 'B', 'C']:
    rms_m = np.sqrt(np.mean(m[f'd_log_mean_{s}'].dropna() ** 2))
    rms_s = np.sqrt(np.mean(m[f'd_log_sd_{s}'].dropna() ** 2))
    print(f'  {s}: RMS Δlog_mean = {rms_m:.3f}   RMS Δlog_sd = {rms_s:.3f}')
print()
print('Legend: A = P_LUX only (2017–March 2020)')
print('        B = LUX_L only (2021–Aug 2023)')
print('        C = pooled, weight-halved')
