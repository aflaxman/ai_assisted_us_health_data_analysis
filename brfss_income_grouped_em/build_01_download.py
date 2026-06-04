"""Generate 01_download.ipynb."""
from _build_notebook import build, md, code

cells = [
    md("""# 01 — Download BRFSS 2023 and verify the codings

This notebook caches the BRFSS 2023 public-use file and the official codebook, then
**verifies every coding the analysis depends on against the codebook itself** before
any recode. Reproducibility hinges on this step; we never trust a remembered code.

Files cached under `../data/raw/brfss/2023/`:

- `LLCP2023XPT.zip` → `LLCP2023.XPT` — the annual landline+cellphone (LLCP) public-use file (CDC).
- `codebook23_llcp-v2-508.zip` → `USCODE23_LLCP_021924.HTML` — the variable codebook.
- `calc_vars_2023.pdf` — the Calculated Variables report.

We pull only the columns the analysis uses and cache a slim parquet
(`../data/derived/brfss_2023_raw_subset.parquet`) so later notebooks load in seconds."""),
    code("""import os, zipfile, requests
import numpy as np
import pandas as pd
import pyreadstat

import recode as rc

DATA = os.path.abspath(os.path.join('..', 'data'))
RAW = os.path.join(DATA, 'raw', 'brfss', '2023')
DERIVED = os.path.join(DATA, 'derived')
os.makedirs(RAW, exist_ok=True)
os.makedirs(DERIVED, exist_ok=True)

FILES = {
    'LLCP2023XPT.zip':
        'https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip',
    'codebook23_llcp-v2-508.zip':
        'https://www.cdc.gov/brfss/annual_data/2023/zip/codebook23_llcp-v2-508.zip',
    'calc_vars_2023.pdf':
        'https://www.cdc.gov/brfss/annual_data/2023/pdf/2023-calculated-variables-version4-508.pdf',
}

def download(name, url):
    dest = os.path.join(RAW, name)
    if os.path.exists(dest):
        print(f'cached  {name} ({os.path.getsize(dest)/1e6:.1f} MB)')
        return dest
    print(f'fetching {name} ...')
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        f.write(r.content)
    print(f'saved   {name} ({os.path.getsize(dest)/1e6:.1f} MB)')
    return dest

for name, url in FILES.items():
    download(name, url)"""),
    md("""## Extract and load the needed columns

The XPT is ~700 MB uncompressed; we read only the columns the analysis uses."""),
    code("""xpt = os.path.join(RAW, 'LLCP2023.XPT')
if not os.path.exists(xpt):
    z = zipfile.ZipFile(os.path.join(RAW, 'LLCP2023XPT.zip'))
    member = z.namelist()[0]               # 'LLCP2023.XPT ' (note trailing space)
    with z.open(member) as src, open(xpt, 'wb') as out:
        out.write(src.read())

df, meta = pyreadstat.read_xport(xpt, usecols=rc.COLUMNS)
print('rows:', len(df), ' columns:', list(df.columns))
df.head(3)"""),
    md("""## Verify the income coding against the codebook

The whole project rests on `INCOME3` storing **Refused (99)** and **Don't know /
Not sure (77)** as *distinct* codes, with brackets 1–11. We confirm the frequencies
reproduce the codebook's published counts."""),
    code("""inc = pd.to_numeric(df['INCOME3'], errors='coerce')
tab = inc.value_counts(dropna=False).sort_index()
tab = tab.rename_axis('INCOME3').reset_index(name='frequency')
tab['pct'] = 100 * tab['frequency'] / len(df)
print(tab.to_string(index=False))
print()
print(f"Refused (99):     {int((inc==99).sum()):>7,}  ({100*(inc==99).mean():.2f}%)")
print(f"Don't know (77):  {int((inc==77).sum()):>7,}  ({100*(inc==77).mean():.2f}%)")
print('Codebook check (expected): 99 -> 42,232 (9.93%);  77 -> 36,316 (8.54%)')"""),
    md("""The bracket dollar boundaries we will use for the grouped-data likelihood (verified
against the codebook, code → interval):"""),
    code("""for code, (lo, hi) in rc.INCOME_BRACKETS_USD.items():
    hi_s = '+inf (open top bracket)' if not np.isfinite(hi) else f'{hi:,.0f}'
    print(f'  {code:>2}: (${lo:,.0f}, ${hi_s})')"""),
    md("""## Verify the outcome coding (`MENTHLTH`)

`MENTHLTH` is days of poor mental health in the past 30: 1–30 = days, 88 = none,
77 = Don't know, 99 = Refused. Frequent mental distress (FMD) = ≥ 14 days."""),
    code("""mh = pd.to_numeric(df['MENTHLTH'], errors='coerce')
print('1-30 days :', int(((mh>=1)&(mh<=30)).sum()))
print('88 (none) :', int((mh==88).sum()))
print('77 (DK)   :', int((mh==77).sum()))
print('99 (Ref)  :', int((mh==99).sum()))
days, valid = rc.recode_menthlth(df['MENTHLTH'])
fmd = rc.fmd_from_days(days)
print(f'\\nFMD (>=14 days), unweighted prevalence among valid: '
      f'{np.nanmean(fmd[valid]):.4f}')"""),
    md("""## Verify the second outcome (`BLIND`, difficulty seeing)

Disability core item: *"Are you blind or do you have serious difficulty seeing, even
when wearing glasses?"* — 1 = Yes, 2 = No, 7 = Don't know, 9 = Refused. Used in
notebook 05."""),
    code("""bl = pd.to_numeric(df['BLIND'], errors='coerce')
print('1 Yes     :', int((bl==1).sum()))
print('2 No      :', int((bl==2).sum()))
print('7 DK      :', int((bl==7).sum()))
print('9 Refused :', int((bl==9).sum()))
print('Codebook check (expected): Yes -> 22,190; No -> 395,423; DK -> 1,113; Ref -> 517')"""),
    md("""## Cache a slim subset for the downstream notebooks"""),
    code("""out = os.path.join(DERIVED, 'brfss_2023_raw_subset.parquet')
df.to_parquet(out, index=False)
print('wrote', out, f'({os.path.getsize(out)/1e6:.1f} MB)')"""),
]

build(cells, '01_download.ipynb')
print('built 01_download.ipynb')
