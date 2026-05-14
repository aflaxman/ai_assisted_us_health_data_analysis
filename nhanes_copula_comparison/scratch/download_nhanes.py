"""Download the NHANES 2017-2018 risk-factor files we need for the copula
comparison. Stores .xpt in scratch/, derives a single parquet."""
import os, requests
from pathlib import Path

OUT = Path('scratch/raw'); OUT.mkdir(parents=True, exist_ok=True)
BASE = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/'
FILES = {
    'DEMO_J.xpt':   'demographics + survey design',
    'BMX_J.xpt':    'BMI (BMXBMI)',
    'BPX_J.xpt':    'systolic BP (BPXSY1..4)',
    'TRIGLY_J.xpt': 'LDL-C (LBDLDL) — fasting subsample',
    'GLU_J.xpt':    'fasting glucose (LBXGLU) — fasting subsample',
    'SMQ_J.xpt':    'smoking history',
    'BIOPRO_J.xpt': 'creatinine (LBXSCR) for eGFR',
}
for fn in FILES:
    fp = OUT / fn
    if fp.exists() and fp.stat().st_size > 1000:
        print(f'  cached {fn} ({fp.stat().st_size:,} bytes)')
        continue
    url = BASE + fn
    print(f'  GET {url}')
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    fp.write_bytes(r.content)
    print(f'    wrote {fn} ({len(r.content):,} bytes)')

print('all files present')
