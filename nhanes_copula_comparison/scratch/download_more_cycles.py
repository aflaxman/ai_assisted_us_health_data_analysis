"""Download BMX, GLU, DEMO for cycles 2007-2018 to enable the
conditional-copula analysis (pair: BMI x FPG)."""
import requests
from pathlib import Path
ROOT = Path('scratch/raw'); ROOT.mkdir(parents=True, exist_ok=True)
# (cycle_year, file_suffix, base_path_year)
CYCLES = [('2007', 'E'), ('2009', 'F'), ('2011', 'G'),
          ('2013', 'H'), ('2015', 'I')]  # J=2017 already downloaded
FILES = ['DEMO', 'BMX', 'GLU']

for y, suf in CYCLES:
    base = f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{y}/DataFiles/'
    for f in FILES:
        fn = f'{f}_{suf}.xpt'
        fp = ROOT / fn
        if fp.exists() and fp.stat().st_size > 1000:
            print(f'cached  {fn}')
            continue
        url = base + fn
        r = requests.get(url, timeout=120)
        if not r.ok:
            print(f'  FAIL {fn} ({r.status_code})  url={url}')
            continue
        fp.write_bytes(r.content)
        print(f'wrote   {fn}  ({len(r.content):,} bytes)')
