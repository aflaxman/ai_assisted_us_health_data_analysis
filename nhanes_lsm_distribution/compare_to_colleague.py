"""
Compare colleague's nhanes_rf_distributions.csv (LSM, LSM_sd in log-space)
against our liver_stiffness_age_sex_lognormal.csv (mean_kpa, sd_kpa in linear
space, F4-calibrated).

Goal: figure out where the age/sex differences come from. There are two
possible sources of divergence:

  (1) Different underlying NHANES sample / weighting / age binning.
      → check by re-computing log-mean and log-SD of LSM from our P_LUX
        parquet, on the SAME 5-year age bins (25-80) the colleague used,
        and seeing if those match his LSM / LSM_sd columns.

  (2) Different fitting method.
      → colleague: log-moment-match (lognormal whose mu, sigma equal the
        sample mean and SD of log(LSM)).
      → us:       F4-calibrated (mu = log(empirical median), sigma chosen
        so P(X >= 12.5) hits the empirical F4 share).
      → the back-transformed arithmetic mean and SD of these two lognormals
        differ in general, even on identical data.

This script reports both, side by side, for the overlap ages 60-79.
"""
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw' / 'nhanes' / '2017_2020_prepandemic'
DERIVED_DIR = DATA_DIR / 'derived'
RAW_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

PARQUET = DERIVED_DIR / 'nhanes_p_lux.parquet'
COLLEAGUE = ROOT.parent / 'vivarium_csu_mace_rct' / 'src' / 'vivarium_csu_mace_rct' / 'data' / 'nhanes_rf_distributions.csv'
OURS = Path(__file__).parent / 'outputs' / 'liver_stiffness_age_sex_lognormal.csv'

F4 = 12.5


# ---------------------------------------------------------------------------
# 0. ensure P_LUX parquet exists (notebook 01 logic, inline)
# ---------------------------------------------------------------------------
if not PARQUET.exists():
    files = {
        'P_DEMO.xpt': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.xpt',
        'P_LUX.xpt':  'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_LUX.xpt',
    }
    for fname, url in files.items():
        out = RAW_DIR / fname
        if out.exists():
            continue
        print(f'downloading {url}')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as r:
            out.write_bytes(r.read())
    demo = pd.read_sas(RAW_DIR / 'P_DEMO.xpt')[
        ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'WTMECPRP']
    ]
    lux = pd.read_sas(RAW_DIR / 'P_LUX.xpt')[
        ['SEQN', 'LUXSMED', 'LUAXSTAT']
    ]
    df = demo.merge(lux, on='SEQN', how='left')
    df['sex'] = df['RIAGENDR'].map({1.0: 'Male', 2.0: 'Female'})
    df['age_years'] = df['RIDAGEYR'].astype(float)
    df['exam_complete'] = df['LUAXSTAT'] == 1.0
    df['LSM_KPA'] = df['LUXSMED']
    df = df[['SEQN', 'sex', 'age_years', 'WTMECPRP',
             'LSM_KPA', 'exam_complete']]
    df.to_parquet(PARQUET, index=False)

df = pd.read_parquet(PARQUET)
ana = df[df['exam_complete'] & df['LSM_KPA'].notna()
         & df['WTMECPRP'].fillna(0).gt(0)].copy()
print(f'MEC-examined adults with LSM, all ages: n = {len(ana):,}')


# ---------------------------------------------------------------------------
# 1. weighted log-moment table on colleague's 5-year bins (age5 = 25, 30, …)
# ---------------------------------------------------------------------------
def w_mean(y, w):  return float(np.average(y, weights=w))
def w_sd(y, w):
    m = w_mean(y, w)
    return float(np.sqrt(np.average((y - m) ** 2, weights=w)))


# colleague's "age5" labels are 5-year bands starting at that age — verify by
# checking row spacing. He has 25, 30, …, 80; that's [25,30), [30,35), …, [80,85).
edges = list(range(25, 85, 5)) + [85]
rows = []
for sex in ['Female', 'Male']:
    for age_start, age_end in zip(edges[:-1], edges[1:]):
        sub = ana[(ana['sex'] == sex)
                  & (ana['age_years'] >= age_start)
                  & (ana['age_years'] < age_end)]
        if len(sub) < 5:
            continue
        y = sub['LSM_KPA'].values.astype(float)
        w = sub['WTMECPRP'].values.astype(float)
        log_y = np.log(y)
        o = np.argsort(y)
        cw = np.cumsum(w[o]) / w.sum()
        wmedian = float(np.interp(0.5, cw, y[o]))
        rows.append({
            'sex': sex, 'age5': age_start, 'n': len(sub),
            'log_mean':   w_mean(log_y, w),
            'log_sd':     w_sd(log_y, w),
            'mean_emp':   w_mean(y, w),
            'sd_emp':     w_sd(y, w),
            'f4_share':   float(np.average((y >= F4).astype(float),
                                            weights=w)),
            'median':     wmedian,
        })
ours_recompute = pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. load colleague's CSV; keep just LSM columns
# ---------------------------------------------------------------------------
coll = pd.read_csv(COLLEAGUE)
coll = coll.rename(columns={'age5': 'age5', 'Sex': 'sex'})
coll = coll[['age5', 'sex', 'LSM', 'LSM_sd']].copy()
coll['LSM_KPA_arith_mean'] = np.exp(coll['LSM'] + coll['LSM_sd']**2 / 2)
coll['LSM_KPA_arith_sd'] = coll['LSM_KPA_arith_mean'] * np.sqrt(
    np.exp(coll['LSM_sd']**2) - 1.0
)


# ---------------------------------------------------------------------------
# 3. load our published F4-calibrated table (60+ only)
# ---------------------------------------------------------------------------
ours = pd.read_csv(OURS)
ours = ours.rename(columns={'mean_kpa': 'ours_mean',
                            'sd_kpa': 'ours_sd',
                            'f4_share_target': 'ours_f4'})
ours['age5'] = ours['age_start'].astype(int)


# ---------------------------------------------------------------------------
# 4. side-by-side comparison
# ---------------------------------------------------------------------------
print()
print('=' * 110)
print('PART A — does our raw NHANES sample reproduce colleague\'s log-scale moments?')
print('=' * 110)
print('"recompute_*" = log mean / sd of LSM, weighted by WTMECPRP, on his 5-yr bins')
print('"coll_*"      = his published LSM / LSM_sd columns (log-space)')
print()
merged = ours_recompute.merge(coll, on=['sex', 'age5'], how='inner')
merged['d_log_mean'] = merged['log_mean'] - merged['LSM']
merged['d_log_sd']   = merged['log_sd']   - merged['LSM_sd']
print(merged[['sex', 'age5', 'n',
              'log_mean', 'LSM', 'd_log_mean',
              'log_sd',   'LSM_sd', 'd_log_sd']].round(3).to_string(index=False))

print()
print('=' * 110)
print('PART B — back-transformed arithmetic mean & SD: colleague vs our F4-cal (60-79)')
print('=' * 110)
print('coll_*  = exp(LSM + LSM_sd^2/2) etc. — colleague\'s implied arithmetic moments')
print('ours_*  = our F4-calibrated table from outputs/liver_stiffness_age_sex_lognormal.csv')
print()
overlap = coll.merge(ours, on=['sex', 'age5'], how='inner')
overlap['d_mean'] = overlap['LSM_KPA_arith_mean'] - overlap['ours_mean']
overlap['d_sd']   = overlap['LSM_KPA_arith_sd']   - overlap['ours_sd']
print(overlap[['sex', 'age5',
               'LSM_KPA_arith_mean', 'ours_mean', 'd_mean',
               'LSM_KPA_arith_sd',   'ours_sd',   'd_sd']].round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 5. for each 60+ cell, show the F4 share implied by each fit (the load-bearing
#    quantity in our project) on the SAME empirical sample
# ---------------------------------------------------------------------------
print()
print('=' * 110)
print('PART C — F4 share (P[LSM >= 12.5]) implied by each fit, on the SAME sample')
print('=' * 110)
print('"emp"    = empirical weighted share of LSM >= 12.5 kPa')
print('"coll"   = P[X>=12.5] under his lognormal (mu=LSM, sigma=LSM_sd)')
print('"ours"   = P[X>=12.5] under our F4-calibrated lognormal')
print()
# build moment-match of our own (mean, sd) → its implied F4 share
def f4_lognormal_mm(mean, sd):
    sigma2 = np.log(1 + (sd / mean) ** 2)
    mu = np.log(mean**2 / np.sqrt(mean**2 + sd**2))
    return 1 - stats.lognorm.cdf(F4, np.sqrt(sigma2), scale=np.exp(mu))

rows = []
# overlap ages 60, 65, 70, 75 in the colleague's CSV align to our
# 60-64, 65-69, 70-74, 75-79 cells.
for sex in ['Female', 'Male']:
    for age5 in [60, 65, 70, 75]:
        c = coll[(coll['sex'] == sex) & (coll['age5'] == age5)].iloc[0]
        o = ours[(ours['sex'] == sex) & (ours['age5'] == age5)].iloc[0]
        r = ours_recompute[(ours_recompute['sex'] == sex)
                           & (ours_recompute['age5'] == age5)].iloc[0]
        # colleague's implied F4 from his (mu, sigma)
        f4_coll = 1 - stats.lognorm.cdf(F4, c['LSM_sd'], scale=np.exp(c['LSM']))
        # ours implied F4 from back-transformed (mean, sd) via lognormal MM
        f4_ours = f4_lognormal_mm(o['ours_mean'], o['ours_sd'])
        rows.append({
            'sex': sex, 'age5': age5, 'n': int(r['n']),
            'emp_f4':         r['f4_share'],
            'coll_f4':        float(f4_coll),
            'ours_f4_target': o['ours_f4'],
            'ours_f4_impl':   float(f4_ours),
        })
print(pd.DataFrame(rows).round(4).to_string(index=False))


print()
print('=' * 110)
print('SUMMARY')
print('=' * 110)
print(
    'If PART A diffs (d_log_mean, d_log_sd) are tiny, both analyses see the same\n'
    'underlying NHANES sample, and the divergence in PART B is purely due to\n'
    'method — colleague uses log-moment-match (MLE-style on log scale), we use\n'
    'F4-calibration (anchor median + force tail probability).\n'
    '\n'
    'Compare PART C "coll_f4" vs "emp_f4": if colleague\'s lognormal materially\n'
    'over- or under-shoots the empirical F4 share, that is the practical\n'
    'consequence for any simulation that uses the F4 cutoff as a routing rule.'
)
