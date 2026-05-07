"""
Compile final comparison table between our estimates and the abstract's values.
Outputs:
  results/comparison_table.csv   — side-by-side comparison
  results/comparison_table.md    — markdown-formatted version
"""

import os
import pandas as pd
import numpy as np

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
IN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
os.makedirs(OUT_DIR, exist_ok=True)

DESC_SUMMARY = os.path.join(IN_DIR, 'descriptive_summary.csv')
OR_KEY = os.path.join(IN_DIR, 'logistic_regression_key.csv')


def fmt_pair(est, lci, uci, decimals=2, scale=1):
    return f"{est*scale:.{decimals}f} ({lci*scale:.{decimals}f}–{uci*scale:.{decimals}f})"


def main():
    desc = pd.read_csv(DESC_SUMMARY)
    ors = pd.read_csv(OR_KEY)

    # -----------------------------------------------------------------------
    # Section 1: Descriptive estimates
    # -----------------------------------------------------------------------
    desc_rows = [
        {
            'Section': 'Descriptive',
            'Measure': 'Persons with EE > 0 (annual, millions)',
            'Our estimate (95% CI)': f"{desc.loc[0,'Our_est']} {desc.loc[0,'Our_95CI']}",
            'Abstract (95% CI)': f"{desc.loc[0,'Abstract_est']} {desc.loc[0,'Abstract_95CI']}",
            'Match?': 'Point estimate identical; CIs overlap',
        },
        {
            'Section': 'Descriptive',
            'Measure': 'Total annual EE ($B)',
            'Our estimate (95% CI)': f"{desc.loc[1,'Our_est']} {desc.loc[1,'Our_95CI']}",
            'Abstract (95% CI)': f"{desc.loc[1,'Abstract_est']} {desc.loc[1,'Abstract_95CI']}",
            'Match?': 'Our est +1.2% vs abstract; CIs overlap',
        },
        {
            'Section': 'Descriptive',
            'Measure': 'Mean EE per capita ($)',
            'Our estimate (95% CI)': f"{desc.loc[2,'Our_est']} {desc.loc[2,'Our_95CI']}",
            'Abstract (95% CI)': f"{desc.loc[2,'Abstract_est']} {desc.loc[2,'Abstract_95CI']}",
            'Match?': 'Point estimate identical; confirms per-capita interpretation',
        },
        {
            'Section': 'Descriptive',
            'Measure': 'Mean EE per spender ($)',
            'Our estimate (95% CI)': f"{desc.loc[3,'Our_est']} {desc.loc[3,'Our_95CI']}",
            'Abstract (95% CI)': 'N/A (ambiguous)',
            'Match?': '~$366; abstract $66.61 is per-capita not per-spender',
        },
    ]

    # -----------------------------------------------------------------------
    # Section 2: Logistic regression ORs
    # -----------------------------------------------------------------------
    for _, row in ors.iterrows():
        our_ci = f"{row['Our_OR']:.2f} ({row['Our_lci']:.2f}–{row['Our_uci']:.2f})"
        ab_ci = f"{row['Abstract_OR']:.2f} ({row['Abstract_lci']:.2f}–{row['Abstract_uci']:.2f})"
        match = '✓ CI overlap' if row['CI_overlap'] else '✗ Outside CI'
        desc_rows.append({
            'Section': 'Logistic regression (aOR)',
            'Measure': row['Predictor'],
            'Our estimate (95% CI)': our_ci,
            'Abstract (95% CI)': ab_ci,
            'Match?': match,
        })

    df = pd.DataFrame(desc_rows)

    # CSV
    csv_path = os.path.join(OUT_DIR, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved CSV to {csv_path}')

    # Markdown
    md_lines = [
        '# Results Comparison: Our Estimates vs. Li, Li & Sansgiry (2024)',
        '',
        '**Data:** MEPS Full-Year Consolidated 2017–2021, pooled (annual averages)',
        '',
        '**Survey design:** Taylor linearization with year-specific stratum IDs '
        '(`VARSTR_POOL = year + "_" + VARSTR`), PSU = VARPSU, weight = PERWTYYF / 5',
        '',
        '## Descriptive Estimates',
        '',
        '| Measure | Our estimate (95% CI) | Abstract (95% CI) | Notes |',
        '|---|---|---|---|',
    ]
    for r in desc_rows:
        if r['Section'] == 'Descriptive':
            md_lines.append(
                f"| {r['Measure']} | {r['Our estimate (95% CI)']} "
                f"| {r['Abstract (95% CI)']} | {r['Match?']} |"
            )

    md_lines += [
        '',
        '## Logistic Regression (Predictors of EE > 0)',
        '',
        'Reference categories: age 18–44, Male, Hispanic, <HS education, '
        'poor/negative income, Uninsured.',
        '',
        '| Predictor | Our aOR (95% CI) | Abstract aOR (95% CI) | Match? |',
        '|---|---|---|---|',
    ]
    for r in desc_rows:
        if r['Section'] == 'Logistic regression (aOR)':
            md_lines.append(
                f"| {r['Measure']} | {r['Our estimate (95% CI)']} "
                f"| {r['Abstract (95% CI)']} | {r['Match?']} |"
            )

    md_lines += [
        '',
        '## Notes',
        '',
        '- aOR for Age 65+ and Some college+ fall outside the abstract\'s CIs.',
        '  See `discrepancies.md` for hypotheses.',
        '- Abstract does not specify reference categories; our choices are documented in `README.md`.',
    ]

    md_path = os.path.join(OUT_DIR, 'comparison_table.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f'Saved markdown to {md_path}')

    print('\n' + '\n'.join(md_lines))
    return df


if __name__ == '__main__':
    main()
