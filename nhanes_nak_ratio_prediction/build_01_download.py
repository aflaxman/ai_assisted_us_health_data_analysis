"""Generate 01_data_download.ipynb."""
from _build_notebook import build, md, code

cells = [
    md("""# 01 — Data download

Downloads NHANES files for the dietary sodium/potassium ratio prediction project.

We pull the following data across six cycles (2007–2018):

- **DEMO** — age, sex, race/ethnicity, poverty-income ratio, education, exam weights
- **DR1TOT, DR2TOT** — day-1 and day-2 24-hour dietary recall totals (sodium, potassium, energy)
- **BMX** — measured BMI and waist circumference
- **BPX** — measured systolic/diastolic blood pressure (mean of available readings)
- **BPQ** — self-reported hypertension and treatment
- **DIQ** — self-reported diabetes and treatment
- **SMQ** — smoking status
- **ALQ** — alcohol use

The most recent cycle (2017-2018, suffix `_J`) is held out as the validation set; earlier cycles
(2007-2016) are used for model training.

All files cache to `../data/raw/nhanes/`."""),
    code("""import os, requests
import numpy as np
import pandas as pd
import pyreadstat

DATA = os.path.abspath(os.path.join('..', 'data'))
RAW = os.path.join(DATA, 'raw', 'nhanes')
DERIVED = os.path.join(DATA, 'derived')
os.makedirs(DERIVED, exist_ok=True)

NHANES_BASE = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public'

def download(url, dest):
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = requests.get(url, timeout=180)
    if r.status_code != 200:
        return None
    with open(dest, 'wb') as f:
        f.write(r.content)
    return dest

def read_xpt(path, cols=None):
    if path is None or not os.path.exists(path):
        return None
    try:
        df, _ = pyreadstat.read_xport(path)
    except UnicodeDecodeError:
        df, _ = pyreadstat.read_xport(path, encoding='latin1')
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return df"""),
    md("""## Cycle definitions"""),
    code("""CYCLES = [
    {'label': '2007-2008', 'year': 2007, 'suffix': '_E'},
    {'label': '2009-2010', 'year': 2009, 'suffix': '_F'},
    {'label': '2011-2012', 'year': 2011, 'suffix': '_G'},
    {'label': '2013-2014', 'year': 2013, 'suffix': '_H'},
    {'label': '2015-2016', 'year': 2015, 'suffix': '_I'},
    {'label': '2017-2018', 'year': 2017, 'suffix': '_J'},
]
FILES = ['DEMO', 'DR1TOT', 'DR2TOT', 'BMX', 'BPX', 'BPQ', 'DIQ', 'SMQ', 'ALQ']

def furl(comp, year, suffix):
    return f'{NHANES_BASE}/{year}/DataFiles/{comp}{suffix}.xpt'

def fpath(label, comp, suffix):
    return os.path.join(RAW, label.replace('-', '_'), f'{comp}{suffix}.xpt')"""),
    md("""## Download all files"""),
    code("""missing = []
for c in CYCLES:
    for comp in FILES:
        dest = fpath(c['label'], comp, c['suffix'])
        result = download(furl(comp, c['year'], c['suffix']), dest)
        if result is None:
            missing.append((c['label'], comp))
print(f"downloaded {len(CYCLES)*len(FILES) - len(missing)} files; missing {len(missing)}")
for m in missing:
    print('  missing:', m)"""),
    md("""## Build pooled analytic dataset"""),
    code("""def smoking_cat(row):
    if row.get('SMQ020') == 2:
        return 'never'
    if row.get('SMQ020') == 1:
        if row.get('SMQ040') in (1, 2):
            return 'current'
        if row.get('SMQ040') == 3:
            return 'former'
    return None

frames = []
for c in CYCLES:
    s = c['suffix']
    demo = read_xpt(fpath(c['label'], 'DEMO', s),
                    cols=['SEQN', 'RIDSTATR', 'RIDAGEYR', 'RIAGENDR',
                          'RIDRETH1', 'DMDEDUC2', 'INDFMPIR',
                          'WTINT2YR', 'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA'])
    dr1 = read_xpt(fpath(c['label'], 'DR1TOT', s),
                   cols=['SEQN', 'WTDRD1', 'WTDR2D', 'DR1DRSTZ',
                         'DR1TKCAL', 'DR1TSODI', 'DR1TPOTA'])
    dr2 = read_xpt(fpath(c['label'], 'DR2TOT', s),
                   cols=['SEQN', 'DR2DRSTZ', 'DR2TKCAL', 'DR2TSODI', 'DR2TPOTA'])
    bmx = read_xpt(fpath(c['label'], 'BMX', s),
                   cols=['SEQN', 'BMXBMI', 'BMXWAIST', 'BMXHT', 'BMXWT'])
    bpx = read_xpt(fpath(c['label'], 'BPX', s),
                   cols=['SEQN', 'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4',
                                  'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4'])
    bpq = read_xpt(fpath(c['label'], 'BPQ', s),
                   cols=['SEQN', 'BPQ020', 'BPQ030', 'BPQ050A', 'BPQ080'])
    diq = read_xpt(fpath(c['label'], 'DIQ', s),
                   cols=['SEQN', 'DIQ010', 'DIQ050', 'DIQ070'])
    smq = read_xpt(fpath(c['label'], 'SMQ', s),
                   cols=['SEQN', 'SMQ020', 'SMQ040'])
    alq = read_xpt(fpath(c['label'], 'ALQ', s),
                   cols=['SEQN', 'ALQ101', 'ALQ110', 'ALQ120Q', 'ALQ130'])

    d = demo
    for sub in [dr1, dr2, bmx, bpx, bpq, diq, smq, alq]:
        if sub is not None:
            d = d.merge(sub, on='SEQN', how='left')

    sy_cols = [c2 for c2 in ['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4'] if c2 in d.columns]
    di_cols = [c2 for c2 in ['BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4'] if c2 in d.columns]
    d['SBP'] = d[sy_cols].replace(0, np.nan).mean(axis=1) if sy_cols else np.nan
    d['DBP'] = d[di_cols].replace(0, np.nan).mean(axis=1) if di_cols else np.nan

    d = d[d['RIDSTATR'] == 2]
    d = d[d['RIDAGEYR'] >= 20].copy()
    d['CYCLE'] = c['label']
    d['AGE'] = d['RIDAGEYR']
    d['FEMALE'] = (d['RIAGENDR'] == 2).astype(int)
    d['SEX'] = d['FEMALE'].map({0: 'Male', 1: 'Female'})
    race_map = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'NH White',
                4: 'NH Black', 5: 'Other/Multi'}
    d['RACE'] = d['RIDRETH1'].map(race_map)

    def col(name, default=np.nan):
        return d[name] if name in d.columns else pd.Series(default, index=d.index)

    sr_map = {1.0: 1.0, 2.0: 0.0}
    d['HTN_DX'] = col('BPQ020').map(sr_map)
    d['HTN_TRT'] = col('BPQ050A').map(sr_map)
    d['HC_DX'] = col('BPQ080').map(sr_map)
    d['DM_DX'] = col('DIQ010').map(sr_map)
    d['DM_INS'] = col('DIQ050').map(sr_map)
    d['DM_PILL'] = col('DIQ070').map(sr_map)
    d['DM_TRT'] = ((d['DM_INS'] == 1) | (d['DM_PILL'] == 1)).astype(float)
    d.loc[d['DM_DX'].isna(), 'DM_TRT'] = np.nan
    d['SMOKE'] = d.apply(smoking_cat, axis=1)

    # alcohol: ALQ130 = drinks per day on drinking days (past 12mo)
    alq130 = col('ALQ130')
    d['ALC_DRINKS_PER_DAY'] = alq130.where(alq130 < 70)

    # day-1 / day-2 sodium and potassium; require reliable recall (DRxDRSTZ == 1)
    d['NA1'] = d['DR1TSODI'].where(d['DR1DRSTZ'] == 1)
    d['K1']  = d['DR1TPOTA'].where(d['DR1DRSTZ'] == 1)
    d['KCAL1'] = d['DR1TKCAL'].where(d['DR1DRSTZ'] == 1)
    d['NA2'] = d['DR2TSODI'].where(d['DR2DRSTZ'] == 1)
    d['K2']  = d['DR2TPOTA'].where(d['DR2DRSTZ'] == 1)
    d['KCAL2'] = d['DR2TKCAL'].where(d['DR2DRSTZ'] == 1)

    # Day1+Day2 average where both available; else Day1 only
    d['NA_MGD']  = d[['NA1', 'NA2']].mean(axis=1)
    d['K_MGD']   = d[['K1', 'K2']].mean(axis=1)
    d['KCAL']    = d[['KCAL1', 'KCAL2']].mean(axis=1)
    d['NAK_RATIO'] = d['NA_MGD'] / d['K_MGD']
    d['NAK_MOLAR'] = (d['NA_MGD'] / 23.0) / (d['K_MGD'] / 39.1)

    # Day-1 only fallback for those missing Day 2
    d['NAK_RATIO_D1'] = d['NA1'] / d['K1']

    d['weight'] = d['WTMEC2YR'] / len(CYCLES)
    if 'WTDR2D' in d.columns:
        d['weight_diet2d'] = d['WTDR2D'] / len(CYCLES)
    if 'WTDRD1' in d.columns:
        d['weight_diet1d'] = d['WTDRD1'] / len(CYCLES)

    keep = ['SEQN', 'CYCLE', 'AGE', 'SEX', 'FEMALE', 'RACE',
            'DMDEDUC2', 'INDFMPIR',
            'BMXBMI', 'BMXWAIST', 'BMXHT', 'BMXWT',
            'SBP', 'DBP',
            'HTN_DX', 'HTN_TRT', 'HC_DX', 'DM_DX', 'DM_TRT',
            'SMOKE', 'ALC_DRINKS_PER_DAY',
            'NA1', 'K1', 'KCAL1', 'NA2', 'K2', 'KCAL2',
            'NA_MGD', 'K_MGD', 'KCAL', 'NAK_RATIO', 'NAK_MOLAR', 'NAK_RATIO_D1',
            'weight', 'weight_diet1d', 'weight_diet2d',
            'SDMVPSU', 'SDMVSTRA']
    frames.append(d[[k for k in keep if k in d.columns]])

pooled = pd.concat(frames, ignore_index=True)
print(f"pooled adults (age 20+): {len(pooled):,}")
print(pooled.groupby('CYCLE').size().to_string())"""),
    md("""## Save analytic parquet"""),
    code("""out = os.path.join(DERIVED, 'nak_pooled_2007_2018.parquet')
pooled.to_parquet(out)
print('wrote', out)

print('\\nNa/K ratio summary (full pooled, unweighted):')
print(pooled['NAK_RATIO'].describe().round(2).to_string())
print('\\nDay-1+Day-2 available:', pooled['NAK_RATIO'].notna().sum())
print('Day-1 only available  :', pooled['NAK_RATIO_D1'].notna().sum())"""),
]

build(cells, '01_data_download.ipynb')
print('wrote 01_data_download.ipynb')
