"""Build a single analytic frame for NHANES 2017-2018 with the 6 continuous
risks the user's correlation matrix uses (BMI, LDL-C, SBP, FPG, smoking_signed,
eGFR_signed) on the trial-band slice (age 65-80)."""
import numpy as np, pandas as pd
from pathlib import Path

RAW = Path('scratch/raw')

def read(fn, cols):
    df = pd.read_sas(RAW / fn)
    return df[['SEQN'] + cols]

demo  = read('DEMO_J.xpt',   ['RIDAGEYR','RIAGENDR','SDMVPSU','SDMVSTRA','WTMEC2YR','WTMECPRP'] if False else ['RIDAGEYR','RIAGENDR','SDMVPSU','SDMVSTRA','WTMEC2YR'])
bmx   = read('BMX_J.xpt',    ['BMXBMI'])
bpx_full = pd.read_sas(RAW / 'BPX_J.xpt')
sbp_cols = [c for c in ['BPXSY1','BPXSY2','BPXSY3','BPXSY4'] if c in bpx_full]
bpx_full['SBP_MEAN'] = bpx_full[sbp_cols].mean(axis=1)
bpx = bpx_full[['SEQN','SBP_MEAN']]
ldl   = read('TRIGLY_J.xpt', ['LBDLDL'])
glu   = read('GLU_J.xpt',    ['LBXGLU'])
smq   = read('SMQ_J.xpt',    ['SMQ020','SMQ040'])
bio   = read('BIOPRO_J.xpt', ['LBXSCR'])

df = (demo
      .merge(bmx, on='SEQN', how='left')
      .merge(bpx, on='SEQN', how='left')
      .merge(ldl, on='SEQN', how='left')
      .merge(glu, on='SEQN', how='left')
      .merge(smq, on='SEQN', how='left')
      .merge(bio, on='SEQN', how='left'))
df['AGE']    = df['RIDAGEYR']
df['FEMALE'] = (df['RIAGENDR'] == 2.0).astype(float)

def smoking_cat(r):
    if r['SMQ020'] == 2.0: return 3
    if r['SMQ020'] == 1.0:
        if r['SMQ040'] in (1.0, 2.0): return 1
        if r['SMQ040'] == 3.0: return 2
    return np.nan

def ckd_epi_2021(scr, age, female):
    if pd.isna(scr) or pd.isna(age) or pd.isna(female): return np.nan
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    sex_factor = 1.012 if female else 1.0
    ratio = scr / kappa
    return 142 * (min(ratio, 1) ** alpha) * (max(ratio, 1) ** -1.200) * (0.9938 ** age) * sex_factor

df['smoking_cat'] = df.apply(smoking_cat, axis=1)
df['eGFR'] = df.apply(lambda r: ckd_epi_2021(r['LBXSCR'], r['AGE'], r['FEMALE'] == 1.0), axis=1)
df['smoking_signed'] = 4 - df['smoking_cat']
df['eGFR_signed']    = -df['eGFR']

OUT = Path('outputs'); OUT.mkdir(exist_ok=True)
df.to_parquet(OUT / 'nhanes_2017_2018_merged.parquet')

# Trial band slice with all 6 risks measured
RISKS = ['BMXBMI','LBDLDL','SBP_MEAN','LBXGLU','smoking_signed','eGFR_signed']
trial = df[(df['AGE'].between(65, 80)) & df[RISKS].notna().all(axis=1) & df['WTMEC2YR'].gt(0)].copy()
print(f'total adults in cycle:    {len(df):,}')
print(f'  age 65-80:              {(df["AGE"].between(65,80)).sum():,}')
print(f'  trial-band w/ all 6:    {len(trial):,}')
print(f'  (fasting subsample is the binding constraint)')
trial.to_parquet(OUT / 'trial_band.parquet')
