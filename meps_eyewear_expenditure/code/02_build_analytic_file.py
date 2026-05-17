"""
Build pooled analytic file from MEPS 2017-2021 Full-Year Consolidated files.

Key decisions (documented in README.md):
- Eyewear expenditure: VISEXPyy (year-specific variable name)
- Weight pooling: divide PERWTyyF by 5 (standard MEPS multi-year approach for annual estimates)
- Age groups: <18, 18-44, 45-64, 65+
- Race/ethnicity: RACETHX (1=Hispanic, 2=NH White, 3=NH Black, 4=NH Asian, 5=NH Other)
- Education: HIDEG recoded to <HS/HS/Some college+ for adults 25+; "Child (<18)" for under 18
- Poverty: POVCATyy (1=poor/neg, 2=near poor, 3=low, 4=middle, 5=high)
- Insurance: INSCOVyy (1=Any private, 2=Public only, 3=Uninsured)
- INSCOV missing indicator: persons not enrolled in any category (rare; kept as separate flag)
"""

import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'meps')
)
DERIVED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'derived')
)
os.makedirs(DERIVED_DIR, exist_ok=True)

YEAR_FILES = [
    (2017, 'h201'),
    (2018, 'h209'),
    (2019, 'h216'),
    (2020, 'h224'),
    (2021, 'h233'),
]

N_YEARS = len(YEAR_FILES)


def load_year(year: int, file_code: str) -> pd.DataFrame:
    yy = str(year)[2:]
    path = os.path.join(RAW_DIR, f'{file_code}.dta')
    print(f'Loading {year} ({path})...')

    df = pd.read_stata(path, convert_categoricals=False)
    df.columns = [c.upper() for c in df.columns]

    keep = {
        'DUPERSID': 'dupersid',
        f'VISEXP{yy}': 'visexp',
        f'PERWT{yy}F': 'perwt_annual',
        'VARSTR': 'varstr',
        'VARPSU': 'varpsu',
        'AGE42X': 'age',
        'SEX': 'sex',
        'RACETHX': 'racethx',
        'EDUCYR': 'educyr',
        'HIDEG': 'hideg',
        f'POVCAT{yy}': 'povcat',
        f'INSCOV{yy}': 'inscov',
    }

    # Subset to available columns
    available = {k: v for k, v in keep.items() if k in df.columns}
    missing = [k for k in keep if k not in df.columns]
    if missing:
        print(f'  WARNING: missing columns for {year}: {missing}')

    sub = df[list(available.keys())].rename(columns=available)
    sub['year'] = year
    return sub


def recode(df: pd.DataFrame) -> pd.DataFrame:
    """Recode raw MEPS values to analysis categories."""

    # Age groups (years at panel close, AGE42X = age at round 4/5 interview)
    df['age_cat'] = pd.cut(
        df['age'],
        bins=[-1, 17, 44, 64, 120],
        labels=['<18', '18-44', '45-64', '65+'],
    )

    # Sex: 1=Male, 2=Female
    df['female'] = (df['sex'] == 2).astype(int)
    df['sex_cat'] = df['sex'].map({1: 'Male', 2: 'Female'})

    # Race/ethnicity (RACETHX):
    # 1=Hispanic, 2=NH White, 3=NH Black, 4=NH Asian, 5=NH Other/multiple
    race_map = {
        1: 'Hispanic',
        2: 'NH White',
        3: 'NH Black',
        4: 'NH Asian',
        5: 'NH Other',
    }
    df['race_eth'] = df['racethx'].map(race_map)

    # Education (HIDEG):
    # -1=inapplicable (age<16), 1=no degree, 2=GED, 3=HS diploma,
    # 4=some college, 5=associate's, 6=bachelor's, 7=master's, 8=doctoral/professional
    # Recode for adults 18+:
    #   <HS: no degree (1), GED treated as HS in the abstract's spirit
    #   HS: GED (2), HS diploma (3)
    #   Some college+: 4-8
    # Children (<18): "Child (<18)"
    edu_adult = pd.cut(
        df['hideg'],
        bins=[-2, 1.5, 3.5, 8.5],
        labels=['<HS', 'HS', 'Some college+'],
    )
    df['edu_cat'] = np.where(df['age'] < 18, 'Child (<18)', edu_adult.astype(str))
    # Handle inapplicable/missing: -1 for under 16, -9 for not ascertained
    df.loc[df['hideg'].isin([-1, -9, -8, -7]), 'edu_cat'] = np.where(
        df.loc[df['hideg'].isin([-1, -9, -8, -7]), 'age'] < 18,
        'Child (<18)',
        'Unknown',
    )

    # Poverty category (POVCATyy):
    # 1=poor/negative, 2=near poor, 3=low income, 4=middle income, 5=high income
    pov_map = {
        1: 'Poor/neg',
        2: 'Near poor',
        3: 'Low income',
        4: 'Middle income',
        5: 'High income',
    }
    df['pov_cat'] = df['povcat'].map(pov_map)

    # Insurance (INSCOVyy): 1=Any private, 2=Public only, 3=Uninsured
    ins_map = {
        1: 'Private',
        2: 'Public only',
        3: 'Uninsured',
    }
    df['ins_cat'] = df['inscov'].map(ins_map)

    # Outcome: any eyewear expenditure
    df['ee_any'] = (df['visexp'] > 0).astype(int)

    return df


def main():
    frames = []
    for year, file_code in YEAR_FILES:
        yr_df = load_year(year, file_code)
        frames.append(yr_df)

    df = pd.concat(frames, ignore_index=True)
    print(f'\nRaw pooled shape: {df.shape}')

    # Weight pooling: divide annual weight by 5 to get pooled annual estimate
    df['perwt_pooled'] = df['perwt_annual'] / N_YEARS

    # Create year-unique stratum IDs: MEPS reuses VARSTR codes across the
    # 2019-2021 cohort, so we prefix by year to avoid spurious cross-year
    # within-stratum variance when the pooled file is analyzed as one design.
    df['varstr_pool'] = df['year'].astype(str) + '_' + df['varstr'].astype(int).astype(str)
    df['varpsu_pool'] = df['year'].astype(str) + '_' + df['varpsu'].astype(int).astype(str)

    df = recode(df)

    # Drop records with VISEXP < 0 (negative = not ascertained)
    n_before = len(df)
    df = df[df['visexp'] >= 0].copy()
    n_drop = n_before - len(df)
    if n_drop:
        print(f'Dropped {n_drop} records with negative VISEXP')

    # Summary
    print(f'\nFinal analytic file: {len(df):,} person-years across {df["year"].nunique()} years')
    print(f'  VISEXP > 0: {df["ee_any"].sum():,} records ({df["ee_any"].mean()*100:.1f}%)')
    print(f'  Weighted count (pooled annual): {df["perwt_pooled"].sum()/1e6:.1f}M persons')
    print(f'  Weighted EE > 0: {(df["perwt_pooled"] * df["ee_any"]).sum()/1e6:.1f}M persons')
    print(f'  Total weighted VISEXP: ${(df["perwt_pooled"] * df["visexp"]).sum()/1e9:.2f}B')

    # Distribution checks
    print('\nAge distribution (unweighted):')
    print(df['age_cat'].value_counts().sort_index())
    print('\nRace/eth distribution (unweighted):')
    print(df['race_eth'].value_counts())
    print('\nInsurance distribution (unweighted):')
    print(df['ins_cat'].value_counts())

    out_path = os.path.join(DERIVED_DIR, 'meps_eyewear_analytic.parquet')
    df.to_parquet(out_path, index=False)
    print(f'\nSaved to {out_path}')
    return df


if __name__ == '__main__':
    main()
