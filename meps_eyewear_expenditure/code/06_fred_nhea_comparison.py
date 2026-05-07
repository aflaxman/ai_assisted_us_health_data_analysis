"""
Compare our MEPS survey-based eyewear expenditure estimates to the BEA Health Care
Satellite Account (HCSA) series published on FRED.

FRED series used:
  COEYEQEXPHCSA  — Corrective Eyeglasses & Contact Lenses Expenditures ($B, nominal)
  COEYEQPCHCSA   — Same, per capita ($)
  COEYEQREXHCSA  — Same, real (billions of chained 2017 $)
  COEYEQPIHCSA   — Price index (2017 = 100)

All four series are annual, not seasonally adjusted, from BEA HCSA (MEPS Account), 2000–2021.

Methodology note:
  The BEA HCSA MEPS Account re-estimates national eyewear spending from MEPS microdata
  using a system-of-accounts framework that may differ from simple survey-weighted totals:
    - BEA applies price deflators and imputation methods beyond MEPS sampling weights
    - BEA scope: all US residents (including some institutional); MEPS scope: civilian
      non-institutionalized population only
    - The "Blended Account" (not separately available on FRED as a distinct eyewear series)
      additionally incorporates large claims databases and typically produces higher estimates

Outputs:
  results/fred_comparison.csv   — year-by-year table
  results/fred_comparison.md    — markdown table
  outputs/fred_comparison.png   — time-series chart
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))
from survey_utils import survey_total, survey_mean

DERIVED = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'derived')
)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# ----------------------------------------------------------------------------
# BEA HCSA data (manually entered from FRED, 2017-2021)
# Source: https://fred.stlouisfed.org/series/COEYEQEXPHCSA
#         https://fred.stlouisfed.org/series/COEYEQPCHCSA
# ----------------------------------------------------------------------------
BEA_DATA = pd.DataFrame({
    'year': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
             2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
             2020, 2021],
    'bea_total_b':  [19.91, 18.80, 20.26, 21.10, 22.20, 23.77, 24.91, 27.11,
                     27.46, 28.24, 29.84, 31.61, 31.62, 32.08, 32.41, 32.61,
                     32.41, 32.57, 34.05, 35.66, 33.90, 41.57],
    'bea_percap':   [70.56, 65.97, 70.44, 72.73, 75.82, 80.44, 83.48, 90.00,
                     90.30, 92.06, 96.86, 101.86, 101.15, 101.92, 102.22,
                     102.09, 100.73, 100.59, 104.61, 109.03, 102.60, 125.63],
    'bea_series': ['COEYEQEXPHCSA / COEYEQPCHCSA'] * 22,
})

STRATUM = 'varstr_pool'
PSU     = 'varpsu_pool'
WGT     = 'perwt_pooled'


def year_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute survey-weighted totals and per-capita means by year."""
    rows = []
    for yr in sorted(df['year'].unique()):
        sub = df[df['year'] == yr].copy()
        # Use the ANNUAL weight (not pooled) for year-specific estimates
        sub['wt_yr'] = sub['perwt_annual']
        tot = survey_total(sub, 'visexp', 'wt_yr', STRATUM, PSU)
        pc  = survey_mean(sub, 'visexp', 'wt_yr', STRATUM, PSU)
        rows.append({
            'year': yr,
            'meps_total_b':      tot['est'] / 1e9,
            'meps_total_b_lci':  tot['lci'] / 1e9,
            'meps_total_b_uci':  tot['uci'] / 1e9,
            'meps_percap':       pc['est'],
            'meps_percap_lci':   pc['lci'],
            'meps_percap_uci':   pc['uci'],
        })
    return pd.DataFrame(rows)


def pooled_estimates(df: pd.DataFrame) -> dict:
    """Pooled annual average (5-year)."""
    tot = survey_total(df, 'visexp', WGT, STRATUM, PSU)
    pc  = survey_mean(df, 'visexp', WGT, STRATUM, PSU)
    return {
        'total_b':      tot['est'] / 1e9,
        'total_b_lci':  tot['lci'] / 1e9,
        'total_b_uci':  tot['uci'] / 1e9,
        'percap':       pc['est'],
        'percap_lci':   pc['lci'],
        'percap_uci':   pc['uci'],
    }


def main():
    df = pd.read_parquet(os.path.join(DERIVED, 'meps_eyewear_analytic.parquet'))

    print('Computing year-specific MEPS estimates...')
    meps_yr = year_estimates(df)
    pooled  = pooled_estimates(df)

    # Merge with BEA data
    study_years = BEA_DATA[BEA_DATA['year'].between(2017, 2021)].copy()
    comp = study_years.merge(meps_yr, on='year', how='left')

    print('\n' + '=' * 90)
    print('YEAR-BY-YEAR COMPARISON: MEPS survey vs BEA HCSA (FRED)')
    print('=' * 90)
    print(f"{'Year':<6} {'MEPS Total ($B)':>18} {'BEA HCSA ($B)':>15} "
          f"{'MEPS $/capita':>15} {'BEA $/capita':>14}")
    print('-' * 90)
    for _, r in comp.iterrows():
        print(f"{int(r['year']):<6} "
              f"{r['meps_total_b']:>7.2f} ({r['meps_total_b_lci']:.2f}–{r['meps_total_b_uci']:.2f})  "
              f"{r['bea_total_b']:>13.2f}  "
              f"{r['meps_percap']:>9.2f} ({r['meps_percap_lci']:.2f}–{r['meps_percap_uci']:.2f})  "
              f"{r['bea_percap']:>12.2f}")

    bea_avg = study_years['bea_total_b'].mean()
    bea_pc_avg = study_years['bea_percap'].mean()
    print(f"\n{'2017-2021 avg':<6} "
          f"{pooled['total_b']:>7.2f} ({pooled['total_b_lci']:.2f}–{pooled['total_b_uci']:.2f})  "
          f"{bea_avg:>13.2f}  "
          f"{pooled['percap']:>9.2f} ({pooled['percap_lci']:.2f}–{pooled['percap_uci']:.2f})  "
          f"{bea_pc_avg:>12.2f}")

    ratio = bea_avg / pooled['total_b']
    print(f"\nBEA / MEPS ratio (total): {ratio:.2f}x")
    print('  Likely reasons: BEA includes broader population scope (including')
    print('  institutional); different price-weighting methodology; BEA may')
    print('  include commercial/retail supply-side data beyond MEPS household reports.')

    # -----------------------------------------------------------------------
    # Chart: Total expenditures over time
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('US Eyewear Expenditures: MEPS Survey vs BEA Health Care Satellite Account\n'
                 'Source: MEPS Full-Year Consolidated 2017–2021; FRED COEYEQEXPHCSA/COEYEQPCHCSA',
                 fontsize=10)

    # --- Left: total $B ---
    ax = axes[0]
    yr_full = BEA_DATA.copy()
    ax.plot(yr_full['year'], yr_full['bea_total_b'], 'o-', color='steelblue',
            label='BEA HCSA (FRED)', lw=2)
    ax.errorbar(comp['year'], comp['meps_total_b'],
                yerr=[comp['meps_total_b'] - comp['meps_total_b_lci'],
                      comp['meps_total_b_uci'] - comp['meps_total_b']],
                fmt='s-', color='darkorange', label='MEPS survey (this analysis)',
                lw=2, capsize=4)
    ax.axvspan(2016.5, 2021.5, alpha=0.08, color='gray', label='Abstract study years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Expenditures ($B, nominal)')
    ax.set_title('Total Eyewear Expenditures')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}B'))

    # --- Right: per capita ---
    ax = axes[1]
    ax.plot(yr_full['year'], yr_full['bea_percap'], 'o-', color='steelblue',
            label='BEA HCSA (FRED)', lw=2)
    ax.errorbar(comp['year'], comp['meps_percap'],
                yerr=[comp['meps_percap'] - comp['meps_percap_lci'],
                      comp['meps_percap_uci'] - comp['meps_percap']],
                fmt='s-', color='darkorange', label='MEPS survey (this analysis)',
                lw=2, capsize=4)
    ax.axvspan(2016.5, 2021.5, alpha=0.08, color='gray', label='Abstract study years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Per capita ($/person, nominal)')
    ax.set_title('Eyewear Expenditures per Capita')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}'))

    plt.tight_layout()
    fig_path = os.path.join(OUT, 'fred_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nChart saved to {fig_path}')

    # -----------------------------------------------------------------------
    # Save tables
    # -----------------------------------------------------------------------
    comp_out = comp[['year', 'meps_total_b', 'meps_total_b_lci', 'meps_total_b_uci',
                     'bea_total_b', 'meps_percap', 'meps_percap_lci', 'meps_percap_uci',
                     'bea_percap']].copy()
    csv_path = os.path.join(RES, 'fred_comparison.csv')
    comp_out.to_csv(csv_path, index=False)
    print(f'CSV saved to {csv_path}')

    md_lines = [
        '## MEPS Survey vs BEA Health Care Satellite Account (FRED), 2017–2021',
        '',
        '**FRED series:** COEYEQEXPHCSA (total $B), COEYEQPCHCSA (per capita $).',
        'Both use the BEA HCSA MEPS Account methodology.',
        '',
        '| Year | MEPS total ($B) | BEA HCSA ($B) | MEPS per capita | BEA per capita |',
        '|------|----------------|---------------|-----------------|----------------|',
    ]
    for _, r in comp.iterrows():
        md_lines.append(
            f"| {int(r['year'])} "
            f"| {r['meps_total_b']:.2f} ({r['meps_total_b_lci']:.2f}–{r['meps_total_b_uci']:.2f}) "
            f"| {r['bea_total_b']:.2f} "
            f"| {r['meps_percap']:.2f} ({r['meps_percap_lci']:.2f}–{r['meps_percap_uci']:.2f}) "
            f"| {r['bea_percap']:.2f} |"
        )
    md_lines += [
        f"| **Avg** "
        f"| **{pooled['total_b']:.2f}** ({pooled['total_b_lci']:.2f}–{pooled['total_b_uci']:.2f}) "
        f"| **{bea_avg:.2f}** "
        f"| **{pooled['percap']:.2f}** ({pooled['percap_lci']:.2f}–{pooled['percap_uci']:.2f}) "
        f"| **{bea_pc_avg:.2f}** |",
        '',
        f'**BEA / MEPS ratio:** {ratio:.2f}×',
        '',
        '**Interpretation:** The BEA HCSA estimates are roughly 1.6× higher than our direct',
        'MEPS survey totals. Key reasons:',
        '- BEA scope is broader (all US residents including some institutional population;',
        '  MEPS covers only civilian non-institutionalized)',
        '- BEA uses price-based imputation and retail markup adjustments beyond survey data',
        '- The MEPS VISEXP variable captures household-reported payments; BEA may impute',
        '  expenditures for categories with low survey response rates',
        '- Li et al. abstract uses MEPS direct estimates ($21.56B), not BEA estimates,',
        '  consistent with our finding of $21.82B annual average from MEPS microdata',
    ]

    md_path = os.path.join(RES, 'fred_comparison.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f'Markdown saved to {md_path}')

    return comp


if __name__ == '__main__':
    main()
