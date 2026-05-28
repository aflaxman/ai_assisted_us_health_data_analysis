"""Download NHANES 2021–Aug 2023 LUX_L + DEMO_L, append to P_LUX, save pooled parquet."""
import os, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'nhanes' / '2021_2023'
RAW.mkdir(parents=True, exist_ok=True)
DERIVED = ROOT / 'data' / 'derived'

FILES = {
    'DEMO_L.xpt': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt',
    'LUX_L.xpt':  'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/LUX_L.xpt',
}
for fname, url in FILES.items():
    out = RAW / fname
    if out.exists():
        print(f'cached: {out} ({out.stat().st_size:,} bytes)')
        continue
    print(f'downloading {url}')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as r:
        out.write_bytes(r.read())
    print(f'  saved {out.stat().st_size:,} bytes')

# parse the L cycle
demo_l = pd.read_sas(RAW / 'DEMO_L.xpt')
lux_l = pd.read_sas(RAW / 'LUX_L.xpt')
print('DEMO_L cols sample:', [c for c in demo_l.columns if c in
       ('SEQN','RIAGENDR','RIDAGEYR','WTMECPRP','WTMEC2YR','SDMVPSU','SDMVSTRA')])
print('LUX_L  cols sample:', [c for c in lux_l.columns if 'LUX' in c or 'LUA' in c])

# pick the L MEC weight (NHANES 2021–Aug 2023 uses WTMEC2YR)
mec_col = 'WTMEC2YR' if 'WTMEC2YR' in demo_l.columns else 'WTMECPRP'
demo_l = demo_l[['SEQN', 'RIAGENDR', 'RIDAGEYR', mec_col,
                 'SDMVPSU', 'SDMVSTRA']].rename(columns={mec_col: 'MEC_WT'})
lux_l = lux_l[['SEQN', 'LUXSMED', 'LUXSIQR', 'LUAXSTAT']]
df_l = demo_l.merge(lux_l, on='SEQN', how='left')
df_l['cycle'] = '2021_2023'
df_l['sex'] = df_l['RIAGENDR'].map({1.0: 'Male', 2.0: 'Female'})
df_l['age_years'] = df_l['RIDAGEYR'].astype(float)
df_l['exam_complete'] = df_l['LUAXSTAT'] == 1.0
df_l['LSM_KPA'] = df_l['LUXSMED']
df_l['LSM_IQR'] = df_l['LUXSIQR']

# load existing P_LUX parquet, give it the same column shape
p = pd.read_parquet(DERIVED / 'nhanes_p_lux.parquet')
p['cycle'] = '2017_2020'
p = p.rename(columns={'WTMECPRP': 'MEC_WT'})

KEEP = ['SEQN', 'cycle', 'sex', 'age_years', 'MEC_WT',
        'SDMVPSU', 'SDMVSTRA', 'LSM_KPA', 'LSM_IQR', 'exam_complete']
# P parquet didn't keep SDMVPSU/SDMVSTRA — re-load if missing
if 'SDMVPSU' not in p.columns:
    p['SDMVPSU'] = pd.NA
    p['SDMVSTRA'] = pd.NA
if 'LSM_IQR' not in p.columns:
    p['LSM_IQR'] = pd.NA

p = p[[c for c in KEEP if c in p.columns]]
df_l = df_l[[c for c in KEEP if c in df_l.columns]]

# combine with weight-rescaling: for naive pooling, just stack — the survey-weighted
# means/SDs are still population-representative within each cycle. For an "average
# of the two cycles" you'd halve each MEC weight; the absolute weight scale doesn't
# affect weighted moments.
pooled = pd.concat([p, df_l], ignore_index=True)
print()
print('cycle counts (with LSM, exam_complete, MEC_WT>0):')
m = pooled['exam_complete'].fillna(False) & pooled['LSM_KPA'].notna() & pooled['MEC_WT'].fillna(0).gt(0)
print(pooled[m].groupby('cycle').size())

out = DERIVED / 'nhanes_p_lux_plus_l.parquet'
pooled.to_parquet(out, index=False)
print(f'wrote {out} ({out.stat().st_size:,} bytes)')
