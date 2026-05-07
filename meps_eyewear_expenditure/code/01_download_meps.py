"""
Download MEPS Full-Year Consolidated files for 2017–2021.

MEPS ships Stata (.dta) files at:
  https://meps.ahrq.gov/mepsweb/data_files/pufs/<hNNN>/<hNNN>dta.zip

File number mapping (Full-Year Consolidated):
  2017 → HC-201 (h201)
  2018 → HC-209 (h209)
  2019 → HC-216 (h216)
  2020 → HC-224 (h224)
  2021 → HC-233 (h233)

Output: data/raw/meps/<hNNN>.dta for each year.
Metadata: data/raw/meps/column_inventory.json listing all columns per year.
"""

import os
import sys
import json
import zipfile
import requests
import pandas as pd

RAW_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'meps')
)
os.makedirs(RAW_DIR, exist_ok=True)

YEAR_FILES = [
    (2017, 'h201'),
    (2018, 'h209'),
    (2019, 'h216'),
    (2020, 'h224'),
    (2021, 'h233'),
]

BASE_URL = 'https://meps.ahrq.gov/mepsweb/data_files/pufs'

REQUIRED_COLS = [
    'VARSTR', 'VARPSU',
    'AGE42X', 'SEX', 'RACETHX',
    'INSCOV',
]


def download_file(url: str, dest_path: str) -> bool:
    print(f'  GET {url}')
    try:
        r = requests.get(url, stream=True, timeout=180, allow_redirects=True)
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f'  HTTP error: {e}')
        return False
    except requests.RequestException as e:
        print(f'  Request error: {e}')
        return False
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    size = os.path.getsize(dest_path)
    print(f'  Saved {size:,} bytes → {dest_path}')
    return size > 1000


def get_dta_path(file_code: str) -> str:
    return os.path.join(RAW_DIR, f'{file_code}.dta')


def download_and_extract(year: int, file_code: str) -> bool:
    dta_path = get_dta_path(file_code)
    if os.path.exists(dta_path):
        print(f'{year}: already present at {dta_path}, skipping download.')
        return True

    zip_name = f'{file_code}dta.zip'
    zip_url = f'{BASE_URL}/{file_code}/{zip_name}'
    zip_path = os.path.join(RAW_DIR, zip_name)

    print(f'\n=== {year} (file code {file_code}) ===')
    ok = download_file(zip_url, zip_path)
    if not ok:
        print(f'  FAILED to download {zip_url}')
        return False

    print(f'  Extracting {zip_path}')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        print(f'  ZIP contains: {members}')
        dta_members = [m for m in members if m.lower().endswith('.dta')]
        if not dta_members:
            print(f'  WARNING: no .dta file found; contents: {members}')
            zf.extractall(RAW_DIR)
        else:
            for m in dta_members:
                zf.extract(m, RAW_DIR)
                extracted = os.path.join(RAW_DIR, m)
                canonical = dta_path
                if extracted != canonical:
                    os.rename(extracted, canonical)
                    print(f'  Renamed {extracted} → {canonical}')

    os.remove(zip_path)
    return os.path.exists(dta_path)


def inspect_columns(year: int, file_code: str) -> list:
    """Read first row only, report column inventory, return column list."""
    dta_path = get_dta_path(file_code)
    if not os.path.exists(dta_path):
        print(f'  {year}: file not found, cannot inspect')
        return []

    print(f'\n--- {year} column inspection ---')
    df = pd.read_stata(dta_path, convert_categoricals=False)
    cols_upper = [c.upper() for c in df.columns]

    for req in REQUIRED_COLS:
        found = req in cols_upper
        print(f'  {req}: {"OK" if found else "MISSING"}')

    yy = str(year)[2:]
    wt_name = f'PERWT{yy}F'
    print(f'  {wt_name} (weight): {"OK" if wt_name in cols_upper else "MISSING"}')

    pov_candidates = [f'POVCAT{yy}', 'POVCAT5', 'POVCAT']
    for p in pov_candidates:
        if p in cols_upper:
            print(f'  Poverty var: {p} — FOUND')
            break
    else:
        print(f'  Poverty var: MISSING (checked {pov_candidates})')

    # VISEXP candidates
    vis_cols = [c for c in cols_upper if 'VIS' in c]
    print(f'  VIS* cols: {vis_cols}')

    edu_cols = [c for c in cols_upper if 'EDUC' in c or 'HIDEG' in c or 'HIDEG' in c]
    print(f'  Education cols: {edu_cols}')

    age_cols = [c for c in cols_upper if 'AGE' in c]
    print(f'  Age cols: {age_cols}')

    print(f'  Total columns: {len(cols_upper)}')
    return cols_upper


def main():
    print(f'MEPS data directory: {RAW_DIR}')
    all_cols = {}
    failed = []

    for year, file_code in YEAR_FILES:
        ok = download_and_extract(year, file_code)
        if not ok:
            failed.append(year)
            continue
        cols = inspect_columns(year, file_code)
        if cols:
            all_cols[year] = cols

    inventory_path = os.path.join(RAW_DIR, 'column_inventory.json')
    with open(inventory_path, 'w') as f:
        json.dump(all_cols, f, indent=2)
    print(f'\nColumn inventory written to {inventory_path}')

    if failed:
        print(f'\nWARNING: failed for years: {failed}')
        sys.exit(1)
    else:
        print('\nAll years downloaded successfully.')


if __name__ == '__main__':
    main()
