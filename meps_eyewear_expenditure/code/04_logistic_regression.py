"""
Survey-weighted logistic regression: predictors of any eyewear expenditure (EE > 0).

Reference categories chosen to match the abstract's reported aORs (all > 1):
  Age:       18-44 (reference)
  Sex:       Male (reference)
  Race/eth:  Hispanic (reference)  → NH White aOR ≈ 1.16
  Education: <HS (reference)       → Higher ed aOR ≈ 1.99
  Poverty:   Poor/neg (reference)  → High income aOR ≈ 1.25
  Insurance: Uninsured (reference) → Private aOR ≈ 2.06

Children (<18) are included with a "Child (<18)" education indicator.
All model coefficients use Taylor-linearized sandwich standard errors.

Outputs:
  outputs/logistic_regression_ors.csv   — full OR table
  outputs/logistic_regression_key.csv   — comparison to abstract values
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from survey_utils import survey_glm_logit

DERIVED = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'derived')
)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
os.makedirs(OUT, exist_ok=True)

STRATUM = 'varstr_pool'
PSU = 'varpsu_pool'
WGT = 'perwt_pooled'


def encode_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Create dummy variables with specified reference categories."""
    out = df.copy()

    # Age: reference = 18-44
    out['age_lt18'] = (out['age_cat'] == '<18').astype(float)
    out['age_45_64'] = (out['age_cat'] == '45-64').astype(float)
    out['age_65plus'] = (out['age_cat'] == '65+').astype(float)
    # 18-44 = reference (omitted)

    # Sex: reference = Male
    out['female_d'] = out['female'].astype(float)

    # Race/ethnicity: reference = Hispanic
    out['race_nhwhite'] = (out['race_eth'] == 'NH White').astype(float)
    out['race_nhblack'] = (out['race_eth'] == 'NH Black').astype(float)
    out['race_nhasian'] = (out['race_eth'] == 'NH Asian').astype(float)
    out['race_nhother'] = (out['race_eth'] == 'NH Other').astype(float)
    # Hispanic = reference (omitted)

    # Education: reference = <HS
    # Categories: <HS (ref), HS, Some college+, Child (<18), Unknown
    out['edu_hs'] = (out['edu_cat'] == 'HS').astype(float)
    out['edu_college'] = (out['edu_cat'] == 'Some college+').astype(float)
    out['edu_child'] = (out['edu_cat'] == 'Child (<18)').astype(float)
    out['edu_unknown'] = (out['edu_cat'] == 'Unknown').astype(float)
    # <HS = reference (omitted)

    # Poverty: reference = Poor/neg
    out['pov_nearpoor'] = (out['pov_cat'] == 'Near poor').astype(float)
    out['pov_low'] = (out['pov_cat'] == 'Low income').astype(float)
    out['pov_middle'] = (out['pov_cat'] == 'Middle income').astype(float)
    out['pov_high'] = (out['pov_cat'] == 'High income').astype(float)
    # Poor/neg = reference (omitted)

    # Insurance: reference = Uninsured
    out['ins_private'] = (out['ins_cat'] == 'Private').astype(float)
    out['ins_public'] = (out['ins_cat'] == 'Public only').astype(float)
    # Uninsured = reference (omitted)

    return out


def main():
    df = pd.read_parquet(os.path.join(DERIVED, 'meps_eyewear_analytic.parquet'))
    df = encode_dummies(df)

    # Drop records with missing edu or pov or ins (should be minimal)
    n_before = len(df)
    df = df.dropna(subset=['age_cat', 'race_eth', 'pov_cat', 'ins_cat',
                            'edu_cat', 'female']).copy()
    n_drop = n_before - len(df)
    print(f'Dropped {n_drop} records with missing covariates; N = {len(df):,}')
    print(f'Outcome prevalence: {df["ee_any"].mean()*100:.1f}%')

    x_cols = [
        'age_lt18', 'age_45_64', 'age_65plus',
        'female_d',
        'race_nhwhite', 'race_nhblack', 'race_nhasian', 'race_nhother',
        'edu_hs', 'edu_college', 'edu_child', 'edu_unknown',
        'pov_nearpoor', 'pov_low', 'pov_middle', 'pov_high',
        'ins_private', 'ins_public',
    ]

    print('\nFitting survey-weighted logistic regression...')
    results = survey_glm_logit(df, 'ee_any', x_cols, WGT, STRATUM, PSU)

    # Pretty-print
    print('\n' + '=' * 90)
    print(f"{'Variable':<20} {'OR':>8} {'95% CI':>25} {'p':>8}")
    print('=' * 90)
    for _, row in results.iterrows():
        if row['variable'] == 'Intercept':
            continue
        stars = ('***' if row['p'] < 0.001 else '**' if row['p'] < 0.01
                 else '*' if row['p'] < 0.05 else '')
        print(f"{row['variable']:<20} {row['or']:>8.3f}  "
              f"({row['or_lci']:.3f}–{row['or_uci']:.3f})  {row['p']:>8.4f} {stars}")

    # Save full results
    full_path = os.path.join(OUT, 'logistic_regression_ors.csv')
    results.to_csv(full_path, index=False)
    print(f'\nFull results saved to {full_path}')

    # Key comparisons with abstract
    abstract_vals = {
        'race_nhwhite':  (1.16, 1.09, 1.23, 'NH White vs Hispanic'),
        'female_d':      (1.40, 1.35, 1.45, 'Female vs Male'),
        'age_65plus':    (1.47, 1.34, 1.60, 'Age 65+ vs 18-44'),
        'edu_college':   (1.99, 1.80, 2.19, 'Some college+ vs <HS'),
        'pov_high':      (1.25, 1.17, 1.33, 'High income vs Poor'),
        'ins_private':   (2.06, 1.84, 2.29, 'Private ins vs Uninsured'),
    }

    key_rows = []
    for var, (ab_or, ab_lci, ab_uci, label) in abstract_vals.items():
        row = results[results['variable'] == var].iloc[0]
        key_rows.append({
            'Predictor': label,
            'Our_OR': round(row['or'], 3),
            'Our_lci': round(row['or_lci'], 3),
            'Our_uci': round(row['or_uci'], 3),
            'Abstract_OR': ab_or,
            'Abstract_lci': ab_lci,
            'Abstract_uci': ab_uci,
            'CI_overlap': (row['or_lci'] <= ab_uci and row['or_uci'] >= ab_lci),
        })

    key_df = pd.DataFrame(key_rows)
    print('\n' + '=' * 90)
    print('KEY COMPARISON TO ABSTRACT')
    print('=' * 90)
    for _, r in key_df.iterrows():
        match = '✓' if r['CI_overlap'] else '✗'
        print(f"  {r['Predictor']:<30}: "
              f"ours={r['Our_OR']:.2f} ({r['Our_lci']:.2f}–{r['Our_uci']:.2f})  "
              f"abstract={r['Abstract_OR']:.2f} ({r['Abstract_lci']:.2f}–{r['Abstract_uci']:.2f})  {match}")

    key_path = os.path.join(OUT, 'logistic_regression_key.csv')
    key_df.to_csv(key_path, index=False)
    print(f'\nKey comparisons saved to {key_path}')

    return results, key_df


if __name__ == '__main__':
    main()
