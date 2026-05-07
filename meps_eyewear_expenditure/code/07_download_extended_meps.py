"""
Download MEPS Full-Year Consolidated files for all available years (2000–2022),
extract just the eyewear expenditure total and per-capita by year, and save a
compact time-series parquet for trend analysis.

URL patterns by era:
  2018+  : https://meps.ahrq.gov/mepsweb/data_files/pufs/<hNNN>/<hNNN>dta.zip  (.dta)
  ≤2017  : https://meps.ahrq.gov/mepsweb/data_files/pufs/<hNNN>ssp.zip          (.ssp, XPORT)

MEPS file code mapping (FYC column from AHRQ GitHub meps_file_names.csv):
  year → file_code, era
"""

import os, sys, zipfile, io, requests, time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from survey_utils import survey_total, survey_mean

RAW_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'meps'))
DERIV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'derived'))
os.makedirs(RAW_DIR, exist_ok=True)

# Full Year Consolidated file codes from AHRQ (meps_file_names.csv)
# era='ssp' means XPORT, era='dta' means Stata native (available 2018+)
YEAR_META = [
    (2000,'h50','ssp'), (2001,'h60','ssp'), (2002,'h70','ssp'),
    (2003,'h79','ssp'), (2004,'h89','ssp'), (2005,'h97','ssp'),
    (2006,'h105','ssp'),(2007,'h113','ssp'),(2008,'h121','ssp'),
    (2009,'h129','ssp'),(2010,'h138','ssp'),(2011,'h147','ssp'),
    (2012,'h155','ssp'),(2013,'h163','ssp'),(2014,'h171','ssp'),
    (2015,'h181','ssp'),(2016,'h192','ssp'),(2017,'h201','dta'),
    (2018,'h209','dta'),(2019,'h216','dta'),(2020,'h224','dta'),
    (2021,'h233','dta'),(2022,'h243','dta'),
]

BASE = 'https://meps.ahrq.gov/mepsweb/data_files/pufs'


def local_path(code, era):
    ext = 'dta' if era == 'dta' else 'ssp'
    return os.path.join(RAW_DIR, f'{code}.{ext}')


def download_year(year, code, era):
    dest = local_path(code, era)
    if os.path.exists(dest):
        return dest
    if era == 'dta':
        url = f'{BASE}/{code}/{code}dta.zip'
    else:
        url = f'{BASE}/{code}ssp.zip'
    print(f'  Downloading {year} ({code}) from {url}')
    for attempt in range(4):
        try:
            r = requests.get(url, stream=True, timeout=180)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 3:
                print(f'  FAILED: {e}')
                return None
            time.sleep(2 ** attempt)
    zip_path = dest + '.zip'
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        ext_members = [m for m in members if m.lower().endswith(f'.{era if era=="dta" else "ssp"}')]
        if ext_members:
            zf.extract(ext_members[0], RAW_DIR)
            extracted = os.path.join(RAW_DIR, ext_members[0])
            if extracted != dest:
                os.rename(extracted, dest)
        else:
            print(f'  WARNING: no .{era} file found in zip; contents: {members}')
    os.remove(zip_path)
    return dest if os.path.exists(dest) else None


def read_year(year, code, era):
    """Read the file and return (visexp_series, weight_series, varstr_series, varpsu_series)."""
    path = local_path(code, era)
    yy = str(year)[2:]
    if era == 'dta':
        df = pd.read_stata(path, convert_categoricals=False)
        df.columns = [c.upper() for c in df.columns]
    else:
        # XPORT — read in chunks for memory efficiency
        chunks = []
        for chunk in pd.read_sas(path, format='xport', encoding='latin1', chunksize=5000):
            chunk.columns = [c.upper() for c in chunk.columns]
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

    # Identify variable names
    vis_col  = f'VISEXP{yy}' if f'VISEXP{yy}' in df.columns else 'VISEXP'
    wt_col   = f'PERWT{yy}F' if f'PERWT{yy}F' in df.columns else None
    if wt_col is None:
        candidates = [c for c in df.columns if c.startswith('PERWT') and c.endswith('F')]
        wt_col = candidates[0] if candidates else None

    missing = [v for v in [vis_col, wt_col, 'VARSTR', 'VARPSU'] if v not in df.columns]
    if missing or wt_col is None:
        print(f'  {year}: missing columns {missing}; VIS cols: {[c for c in df.columns if "VIS" in c][:5]}')
        return None

    sub = df[[vis_col, wt_col, 'VARSTR', 'VARPSU']].copy()
    sub.columns = ['visexp', 'perwt', 'varstr', 'varpsu']
    sub['varstr'] = sub['varstr'].astype(str)
    sub['varpsu'] = sub['varpsu'].astype(str)
    sub = sub[sub['visexp'] >= 0]
    return sub


def compute_estimates(year, df):
    tot = survey_total(df, 'visexp', 'perwt', 'varstr', 'varpsu')
    pc  = survey_mean(df,  'visexp', 'perwt', 'varstr', 'varpsu')
    n_pop = df['perwt'].sum()
    return {
        'year':      year,
        'total_b':   tot['est'] / 1e9,
        'total_lci': tot['lci'] / 1e9,
        'total_uci': tot['uci'] / 1e9,
        'percap':    pc['est'],
        'percap_lci':pc['lci'],
        'percap_uci':pc['uci'],
        'n_records': len(df),
        'pop_M':     n_pop / 1e6,
    }


def main():
    rows = []
    for year, code, era in YEAR_META:
        print(f'\n=== {year} ({code}, {era}) ===')
        path = download_year(year, code, era)
        if path is None:
            print(f'  Skipping {year}: download failed')
            continue
        df = read_year(year, code, era)
        if df is None:
            print(f'  Skipping {year}: variable extraction failed')
            continue
        est = compute_estimates(year, df)
        rows.append(est)
        print(f'  Total: ${est["total_b"]:.2f}B  Per-capita: ${est["percap"]:.2f}  Pop: {est["pop_M"]:.0f}M')

    out = pd.DataFrame(rows)
    out_path = os.path.join(DERIV_DIR, 'meps_annual_eyewear.parquet')
    out.to_parquet(out_path, index=False)
    print(f'\nSaved {len(out)} years to {out_path}')
    print(out[['year','total_b','percap','pop_M']].to_string(index=False))
    return out


if __name__ == '__main__':
    main()
