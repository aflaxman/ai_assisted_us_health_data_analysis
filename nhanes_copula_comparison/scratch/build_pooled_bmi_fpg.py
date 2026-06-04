"""Pool BMI + FPG + age + survey weights across NHANES 2007-2018."""
import numpy as np, pandas as pd
from pathlib import Path
RAW = Path('scratch/raw')
CYCLES = [('2007_2008','E'),('2009_2010','F'),('2011_2012','G'),
          ('2013_2014','H'),('2015_2016','I'),('2017_2018','J')]
frames = []
for cyc, suf in CYCLES:
    demo = pd.read_sas(RAW/f'DEMO_{suf}.xpt')[['SEQN','RIDAGEYR','RIAGENDR','WTMEC2YR']]
    bmx  = pd.read_sas(RAW/f'BMX_{suf}.xpt')[['SEQN','BMXBMI']]
    glu  = pd.read_sas(RAW/f'GLU_{suf}.xpt')[['SEQN','LBXGLU','WTSAF2YR']]
    df = demo.merge(bmx, on='SEQN').merge(glu, on='SEQN')
    df['CYCLE'] = cyc
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
n_cycles = len(CYCLES)
# Use fasting weight WTSAF2YR for FPG-based analyses (NCHS analytic guideline)
df['weight'] = df['WTSAF2YR'] / n_cycles
df = df[df[['BMXBMI','LBXGLU','RIDAGEYR']].notna().all(axis=1) & (df['weight'] > 0)]
print(f'pooled n = {len(df):,}')
print('  by age band:')
for lo, hi in [(20,40),(40,55),(55,70),(70,85)]:
    print(f'    {lo}-{hi-1}: n = {df["RIDAGEYR"].between(lo,hi-1).sum():,}')
df.to_parquet('outputs/pooled_bmi_fpg_2007_2018.parquet')
