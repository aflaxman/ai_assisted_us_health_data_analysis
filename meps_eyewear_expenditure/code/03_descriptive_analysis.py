"""
Descriptive survey-weighted analysis of MEPS eyewear expenditures, 2017-2021.

Produces:
  outputs/descriptive_summary.csv  — key estimates with 95% CIs
  outputs/demographic_distribution.csv — weighted demographics among EE > 0 spenders
"""

import os
import sys
import pandas as pd
import numpy as np

# Add code dir to path for survey_utils
sys.path.insert(0, os.path.dirname(__file__))
from survey_utils import survey_total, survey_mean, survey_count, survey_proportion

DERIVED = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'derived')
)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
os.makedirs(OUT, exist_ok=True)

STRATUM = 'varstr_pool'
PSU = 'varpsu_pool'
WGT = 'perwt_pooled'


def fmt_ci(est, lci, uci, scale=1, decimals=2):
    return (f"{est*scale:,.{decimals}f} "
            f"({lci*scale:,.{decimals}f}–{uci*scale:,.{decimals}f})")


def main():
    df = pd.read_parquet(os.path.join(DERIVED, 'meps_eyewear_analytic.parquet'))

    print("=" * 70)
    print("DESCRIPTIVE ANALYSIS — MEPS Eyewear Expenditures 2017–2021")
    print("=" * 70)
    print(f"Total person-year records: {len(df):,}")
    print(f"Pooled annual weight sum (population): "
          f"{df[WGT].sum()/1e6:.1f}M")

    # -----------------------------------------------------------------------
    # 1. Number of persons with EE > 0
    # -----------------------------------------------------------------------
    cnt = survey_count(df, 'ee_any', WGT, STRATUM, PSU)
    print(f"\n1. Persons with EE > 0 (annual avg):")
    print(f"   {fmt_ci(cnt['est'], cnt['lci'], cnt['uci'], scale=1/1e6, decimals=1)}M")
    print(f"   Abstract target: 59.6M (57.3–61.9M)")

    # -----------------------------------------------------------------------
    # 2. Total annual eyewear expenditure
    # -----------------------------------------------------------------------
    tot = survey_total(df, 'visexp', WGT, STRATUM, PSU)
    print(f"\n2. Total annual EE:")
    print(f"   ${fmt_ci(tot['est'], tot['lci'], tot['uci'], scale=1/1e9, decimals=2)}B")
    print(f"   Abstract target: $21.56B (20.80–22.19B)")

    # -----------------------------------------------------------------------
    # 3a. Mean EE per capita (full population)
    # -----------------------------------------------------------------------
    mean_pop = survey_mean(df, 'visexp', WGT, STRATUM, PSU)
    print(f"\n3a. Mean EE per capita (full population):")
    print(f"   ${fmt_ci(mean_pop['est'], mean_pop['lci'], mean_pop['uci'], decimals=2)}")
    print(f"    Abstract reports $66.61 (64.26–68.95) — likely this measure")

    # -----------------------------------------------------------------------
    # 3b. Mean EE per spender (among EE > 0)
    # -----------------------------------------------------------------------
    spenders = df[df['ee_any'] == 1].copy()
    mean_spender = survey_mean(spenders, 'visexp', WGT, STRATUM, PSU)
    print(f"\n3b. Mean EE per spender (EE > 0 only):")
    print(f"   ${fmt_ci(mean_spender['est'], mean_spender['lci'], mean_spender['uci'], decimals=2)}")
    print(f"    ($66.61 would be implausibly low for per-spender; ~$360 expected)")

    # -----------------------------------------------------------------------
    # 4. Demographic distribution among EE > 0 spenders
    # -----------------------------------------------------------------------
    print("\n4. Demographic distribution among EE > 0 spenders:")

    demo_rows = []

    def wt_pct(sub_df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
        """Weighted percent distribution of cat_col in sub_df."""
        cats = sub_df[cat_col].dropna().unique()
        total_w = (sub_df[WGT].sum())
        rows = []
        for cat in sorted(cats, key=str):
            mask = sub_df[cat_col] == cat
            wt = sub_df.loc[mask, WGT].sum()
            pct = wt / total_w * 100
            rows.append({'category': f'{cat_col}: {cat}', 'pct': pct, 'n': mask.sum()})
        return pd.DataFrame(rows)

    # Use spenders subset for demographic distribution
    for col in ['age_cat', 'sex_cat', 'race_eth', 'edu_cat', 'pov_cat', 'ins_cat']:
        grp = wt_pct(spenders, col)
        demo_rows.append(grp)
        print(f"\n  {col}:")
        for _, row in grp.iterrows():
            print(f"    {row['category']:<35} {row['pct']:5.1f}%  (n={row['n']:,})")

    demo_df = pd.concat(demo_rows, ignore_index=True)

    # -----------------------------------------------------------------------
    # Abstract comparison table
    # -----------------------------------------------------------------------
    summary = pd.DataFrame([
        {
            'Measure': 'Persons with EE > 0 (M)',
            'Our_est': f"{cnt['est']/1e6:.2f}",
            'Our_95CI': f"({cnt['lci']/1e6:.2f}–{cnt['uci']/1e6:.2f})",
            'Abstract_est': '59.6',
            'Abstract_95CI': '(57.3–61.9)',
        },
        {
            'Measure': 'Total annual EE ($B)',
            'Our_est': f"{tot['est']/1e9:.2f}",
            'Our_95CI': f"({tot['lci']/1e9:.2f}–{tot['uci']/1e9:.2f})",
            'Abstract_est': '21.56',
            'Abstract_95CI': '(20.80–22.19)',
        },
        {
            'Measure': 'Mean EE per capita ($)',
            'Our_est': f"{mean_pop['est']:.2f}",
            'Our_95CI': f"({mean_pop['lci']:.2f}–{mean_pop['uci']:.2f})",
            'Abstract_est': '66.61',
            'Abstract_95CI': '(64.26–68.95)',
        },
        {
            'Measure': 'Mean EE per spender ($)',
            'Our_est': f"{mean_spender['est']:.2f}",
            'Our_95CI': f"({mean_spender['lci']:.2f}–{mean_spender['uci']:.2f})",
            'Abstract_est': 'N/A',
            'Abstract_95CI': '(ambiguous; see discrepancies.md)',
        },
    ])

    out_path = os.path.join(OUT, 'descriptive_summary.csv')
    summary.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")

    demo_path = os.path.join(OUT, 'demographic_distribution.csv')
    demo_df.to_csv(demo_path, index=False)
    print(f"Saved demographics to {demo_path}")

    return summary, demo_df


if __name__ == '__main__':
    main()
