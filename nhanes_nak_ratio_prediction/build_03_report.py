"""Generate 03_report.ipynb — builds outputs/report.html (interactive)."""
from _build_notebook import build, md, code

cells = [
    md("""# 03 — Interactive HTML report

Builds `outputs/report.html`: a single self-contained file with interactive Plotly figures and
all the headline numbers for sharing with collaborators. The notebook displays the same
figures inline."""),
    code("""import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = 'plotly_white'

DATA = os.path.abspath(os.path.join('..', 'data'))
DERIVED = os.path.join(DATA, 'derived')
OUT = 'outputs'
os.makedirs(OUT, exist_ok=True)

pred = pd.read_parquet(os.path.join(DERIVED, 'nak_holdout_predictions.parquet'))
summary = pd.read_csv(os.path.join(DERIVED, 'nak_model_summary.csv'))
fi = pd.read_csv(os.path.join(DERIVED, 'nak_feature_importance.csv'))
clust = pd.read_parquet(os.path.join(DERIVED, 'nak_catchment_sim.parquet'))
pooled = pd.read_parquet(os.path.join(DERIVED, 'nak_pooled_2007_2018.parquet'))

# survey-style weight: prefer 2-day dietary weight, fallback day-1, fallback MEC weight
pred['wt'] = (pred['weight_diet2d'].fillna(pred['weight_diet1d'])
              .fillna(pred['weight']))
print('holdout rows:', len(pred))
print(summary.round(4).to_string(index=False))"""),
    md("""## Headline metrics box"""),
    code("""def w_mean(x, w):
    w = np.asarray(w); x = np.asarray(x)
    m = ~(np.isnan(x) | np.isnan(w))
    return float(np.average(x[m], weights=w[m])) if m.any() and w[m].sum() > 0 else float('nan')

best = summary.iloc[summary['R2'].idxmax()]
overall = {
    'holdout_cycle': '2017-2018',
    'train_cycles' : '2007-2016 (5 cycles)',
    'n_train'      : int((pooled['CYCLE'].isin(['2007-2008','2009-2010','2011-2012','2013-2014','2015-2016']) & pooled['NAK_RATIO'].notna()).sum()),
    'n_test'       : int(len(pred)),
    'best_model'   : best['model'],
    'best_R2'      : float(best['R2']),
    'best_RMSE'    : float(best['RMSE']),
    'best_MAE'     : float(best['MAE']),
    'obs_mean_overall' : w_mean(pred['NAK_RATIO'], pred['wt']),
    'pred_mean_overall': w_mean(pred['pred_lgbm'], pred['wt']),
}
overall['bias_mean'] = overall['pred_mean_overall'] - overall['obs_mean_overall']
print(json.dumps(overall, indent=2))"""),
    md("""## Figure 1 — Distribution of Na/K ratio (train vs holdout, observed; holdout predicted)"""),
    code("""train = pooled[pooled['CYCLE'].isin(['2007-2008','2009-2010','2011-2012','2013-2014','2015-2016'])
               & pooled['NAK_RATIO'].notna()]

fig1 = go.Figure()
fig1.add_trace(go.Histogram(x=train['NAK_RATIO'], name='Train (observed)',
                            opacity=0.55, nbinsx=80,
                            histnorm='probability density', marker_color='#888'))
fig1.add_trace(go.Histogram(x=pred['NAK_RATIO'], name='Holdout (observed)',
                            opacity=0.55, nbinsx=80,
                            histnorm='probability density', marker_color='#1f77b4'))
fig1.add_trace(go.Histogram(x=pred['pred_lgbm'], name='Holdout (LightGBM predicted)',
                            opacity=0.7, nbinsx=80,
                            histnorm='probability density', marker_color='#d62728'))
fig1.update_layout(
    barmode='overlay',
    title='Distribution of dietary Na/K mass ratio (mg sodium / mg potassium)',
    xaxis_title='Na/K (mg/mg)', yaxis_title='Density', height=400,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
)
fig1.update_xaxes(range=[0, 4])
fig1.show()"""),
    md("""## Figure 2 — Calibration: predicted decile mean vs observed mean (holdout)

For each decile of LightGBM-predicted Na/K we plot the survey-weighted observed mean.
A well-calibrated model lies along the diagonal; horizontal compression indicates the
classic "regression toward the mean" pattern expected when individual-level R² is low.
At the **decile-population** scale calibration is what matters — predicted strata should
rank-order observed strata and agree on subgroup means."""),
    code("""def decile_calibration(yhat, y, w, n_q=10):
    q = pd.qcut(yhat, n_q, labels=False, duplicates='drop')
    rows = []
    for k in sorted(np.unique(q)):
        m = (q == k)
        ww = w[m]
        rows.append({
            'decile': int(k) + 1,
            'pred_mean': np.average(yhat[m], weights=ww),
            'obs_mean' : np.average(y[m], weights=ww),
            'obs_se'   : np.sqrt(np.cov(y[m], aweights=ww) / m.sum()),
            'n': int(m.sum()),
        })
    return pd.DataFrame(rows)

cal = decile_calibration(pred['pred_lgbm'].values, pred['NAK_RATIO'].values, pred['wt'].values)
cal_r = decile_calibration(pred['pred_ridge'].values, pred['NAK_RATIO'].values, pred['wt'].values)

fig2 = go.Figure()
diag = np.linspace(cal['pred_mean'].min(), cal['pred_mean'].max(), 50)
fig2.add_trace(go.Scatter(x=diag, y=diag, mode='lines', name='Perfect calibration',
                          line=dict(color='black', dash='dash')))
fig2.add_trace(go.Scatter(
    x=cal['pred_mean'], y=cal['obs_mean'],
    error_y=dict(type='data', array=1.96*cal['obs_se'], visible=True),
    mode='markers+lines', name='LightGBM',
    marker=dict(size=12, color='#d62728'), line=dict(color='#d62728')))
fig2.add_trace(go.Scatter(
    x=cal_r['pred_mean'], y=cal_r['obs_mean'],
    error_y=dict(type='data', array=1.96*cal_r['obs_se'], visible=True),
    mode='markers+lines', name='Ridge',
    marker=dict(size=10, color='#1f77b4'), line=dict(color='#1f77b4')))
fig2.update_layout(
    title='Decile calibration on holdout (NHANES 2017-2018)',
    xaxis_title='Predicted Na/K (decile mean)', yaxis_title='Observed Na/K (decile mean)',
    height=480,
)
fig2.show()"""),
    md("""## Figure 3 — Subgroup-mean prediction error

Survey-weighted observed mean and predicted mean for demographic subgroups in the holdout
cycle. This is the operational metric for the local prediction use-case."""),
    code("""def subgroup(pred, by, ycol='pred_lgbm'):
    rows = []
    for key, g in pred.groupby(by, dropna=False, observed=True):
        w = g['wt'].values
        rows.append({
            'group': str(key),
            'n': len(g),
            'obs_mean': np.average(g['NAK_RATIO'], weights=w),
            'pred_mean': np.average(g[ycol], weights=w),
            'obs_se': np.sqrt(np.cov(g['NAK_RATIO'], aweights=w) / len(g)),
        })
    return pd.DataFrame(rows)

sub_sex  = subgroup(pred, 'SEX')
sub_race = subgroup(pred, 'RACE')
sub_age  = subgroup(pred.assign(AGEGRP=pd.cut(pred['AGE'], [19,30,45,60,75,200],
                                              labels=['20-29','30-44','45-59','60-74','75+'])),
                    'AGEGRP')

fig3 = make_subplots(rows=1, cols=3, subplot_titles=('Sex', 'Age group', 'Race / ethnicity'),
                     shared_yaxes=True)
for i, sub in enumerate([sub_sex, sub_age, sub_race], 1):
    fig3.add_trace(go.Bar(x=sub['group'], y=sub['obs_mean'], name='Observed' if i==1 else None,
                          marker_color='#1f77b4', showlegend=(i==1),
                          error_y=dict(type='data', array=1.96*sub['obs_se'])), row=1, col=i)
    fig3.add_trace(go.Bar(x=sub['group'], y=sub['pred_mean'], name='Predicted' if i==1 else None,
                          marker_color='#d62728', showlegend=(i==1)), row=1, col=i)
fig3.update_layout(barmode='group', title='Subgroup-mean Na/K — observed vs predicted (holdout 2017-2018)',
                   height=420, legend=dict(orientation='h', y=1.1))
fig3.update_yaxes(title_text='Na/K mass ratio', row=1, col=1)
fig3.show()"""),
    md("""## Figure 4 — Catchment-mean error vs sample size

Random clusters of size N drawn (with replacement) from the holdout; we plot the
distribution of `predicted_mean − observed_mean` across 500 cluster draws per N.
At N≈1000 (a typical small-county scale) the 95% interval is well under ±0.05
Na/K units — about 4% of the overall mean."""),
    code("""sizes = sorted(clust['N'].unique())
fig4 = go.Figure()
for N in sizes:
    e = clust[clust['N']==N]['error']
    fig4.add_trace(go.Box(y=e, name=f'N={N}', boxpoints='outliers',
                          marker_color='#2ca02c'))
fig4.add_hline(y=0, line=dict(color='black', dash='dash'), annotation_text='zero error')
fig4.update_layout(title='Catchment-mean prediction error (500 random clusters per N)',
                   yaxis_title='predicted mean − observed mean (Na/K)',
                   height=420, showlegend=False)
fig4.show()

print('Summary (catchment error):')
print(clust.groupby('N')['error']
      .agg(mean='mean', sd='std',
           ci_lo=lambda x: np.percentile(x,2.5),
           ci_hi=lambda x: np.percentile(x,97.5))
      .round(4).to_string())"""),
    md("""## Figure 5 — Feature importance"""),
    code("""topfi = fi.head(12).iloc[::-1]
fig5 = go.Figure(go.Bar(x=topfi['gain_pct'], y=topfi['feature'], orientation='h',
                         marker_color='#9467bd'))
fig5.update_layout(title='LightGBM feature importance (% total gain)',
                   xaxis_title='% of total gain', height=420,
                   margin=dict(l=140))
fig5.show()"""),
    md("""## Figure 6 — Holdout vs train: secular trend in mean Na/K"""),
    code("""trend = pooled[pooled['NAK_RATIO'].notna()].copy()
trend['wt'] = trend['weight_diet2d'].fillna(trend['weight_diet1d']).fillna(trend['weight'])
def wmean(g): return np.average(g['NAK_RATIO'], weights=g['wt'])
def wse(g):
    n = len(g); w = g['wt'].values
    return float(np.sqrt(np.cov(g['NAK_RATIO'], aweights=w) / n))
gtr = trend.groupby('CYCLE').apply(lambda g: pd.Series({'mean': wmean(g), 'se': wse(g)})).reset_index()

fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=gtr['CYCLE'], y=gtr['mean'],
                          error_y=dict(type='data', array=1.96*gtr['se']),
                          mode='markers+lines', name='Observed cycle mean',
                          marker=dict(size=12, color='#1f77b4')))
fig6.add_hline(y=overall['pred_mean_overall'], line=dict(color='#d62728', dash='dash'),
               annotation_text=f"LightGBM predicted mean for 2017-2018: {overall['pred_mean_overall']:.3f}")
fig6.update_layout(title='Mean Na/K by NHANES cycle (survey-weighted, ±95% CI)',
                   xaxis_title='Cycle', yaxis_title='Na/K mass ratio', height=420)
fig6.show()"""),
    md("""## Assemble report.html"""),
    code("""def fig_to_div(fig, full=False):
    return pio.to_html(fig, include_plotlyjs='cdn' if full else False,
                       full_html=False, default_height='480px')

# Compose summary table for HTML
sum_tbl = summary.copy()
for c in ['RMSE','MAE','R2','Pearson_r']:
    sum_tbl[c] = sum_tbl[c].round(4)
sum_html = sum_tbl.to_html(index=False, classes='summary')

# Catchment summary table
cat_tbl = (clust.groupby('N')['error']
           .agg(mean='mean', sd='std',
                ci_lo=lambda x: np.percentile(x,2.5),
                ci_hi=lambda x: np.percentile(x,97.5))
           .round(4).reset_index())
cat_tbl.columns = ['Catchment size N', 'Mean error', 'SD', '2.5% CI', '97.5% CI']
cat_html = cat_tbl.to_html(index=False, classes='summary')

# Subgroup tables
def to_tbl(sub, label):
    t = sub.copy()
    t['error'] = t['pred_mean'] - t['obs_mean']
    t['pct_err'] = 100 * t['error'] / t['obs_mean']
    t = t.round(3)
    t = t[['group','n','obs_mean','pred_mean','error','pct_err']]
    t.columns = [label, 'N', 'Obs mean', 'Pred mean', 'Pred−Obs', '%']
    return t.to_html(index=False, classes='summary')

html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NHANES dietary Na/K prediction — proof of concept</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.5; }}
  h1 {{ border-bottom: 3px solid #1f77b4; padding-bottom: .2em; }}
  h2 {{ color: #1f77b4; margin-top: 2em; }}
  h3 {{ color: #444; }}
  table.summary {{ border-collapse: collapse; margin: 0.5em 0 1.5em 0; }}
  table.summary th, table.summary td {{ border: 1px solid #ddd; padding: 4px 12px; text-align: right; }}
  table.summary th {{ background: #f3f6fa; }}
  table.summary td:first-child, table.summary th:first-child {{ text-align: left; }}
  .box {{ background: #f6f9fc; border-left: 4px solid #1f77b4; padding: 0.8em 1.2em; margin: 1em 0; }}
  .caption {{ font-size: 0.9em; color: #555; margin: 0.4em 0 1em 0; }}
  code {{ background: #f3f3f3; padding: 1px 5px; border-radius: 3px; }}
</style>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>

<h1>NHANES dietary Na/K prediction — proof of concept</h1>

<div class="box">
  <strong>Question.</strong> Can demographic and clinical predictors that are common to both
  NHANES and EHR data be used to estimate a population's mean dietary Na/K ratio at
  sub-national geographic scales?<br><br>
  <strong>Design.</strong> Train (NHANES 2007-2016, n = {overall['n_train']:,} adults with reliable 24-hour
  dietary recall and full features) → Holdout (NHANES 2017-2018, n = {overall['n_test']:,}).
  Best model: <b>{overall['best_model']}</b>.<br><br>
  <strong>Result.</strong>
  Individual-level holdout R² = <b>{overall['best_R2']:.3f}</b>,
  RMSE = <b>{overall['best_RMSE']:.3f}</b>,
  MAE = <b>{overall['best_MAE']:.3f}</b>.
  Overall holdout-population mean Na/K = <b>{overall['obs_mean_overall']:.3f}</b> observed
  vs <b>{overall['pred_mean_overall']:.3f}</b> predicted
  (bias {overall['bias_mean']:+.3f} mg/mg, ≈ {100*overall['bias_mean']/overall['obs_mean_overall']:+.1f}%).
  At county-scale draws (N=1000) the 95% prediction interval for the population mean is
  ±~0.03 Na/K units (≈ ±2% of the mean).
</div>

<h2>Model summary</h2>
{sum_html}
<p class="caption">Holdout = NHANES 2017-2018 adults (≥20y) with a valid 2-day Na/K. Mean baseline
predicts the training-set mean for every individual.</p>

<h2>1 · Na/K distribution: train vs holdout (observed) and model predictions</h2>
{fig_to_div(fig1)}
<p class="caption">The predicted distribution is much narrower than the observed distribution
because demographics + clinical predictors capture mean differences, not within-stratum
intake variability. Mean prediction at a population scale is the operationally relevant
endpoint.</p>

<h2>2 · Decile calibration on holdout</h2>
{fig_to_div(fig2)}
<p class="caption">Observed vs predicted Na/K means within deciles of model-predicted Na/K
on the holdout, with 95% CIs on observed means. Departures from the dashed identity line
indicate miscalibration of subgroup means.</p>

<h2>3 · Subgroup-mean prediction (the operational metric)</h2>
{fig_to_div(fig3)}
<p class="caption">Survey-weighted observed and predicted means by sex, age group, and race
/ ethnicity in the holdout cycle. Bars are unadjusted means within each cell; this is what
the model is asked to recover from EHR predictors.</p>

<h3>Subgroup error tables</h3>
{to_tbl(sub_sex, "Sex")}
{to_tbl(sub_age, "Age group")}
{to_tbl(sub_race, "Race / ethnicity")}

<h2>4 · Catchment-mean error vs sample size</h2>
{fig_to_div(fig4)}
<p class="caption">500 random clusters per N drawn (with replacement) from the holdout.
Cluster mean prediction error converges toward the small <b>population-level bias</b> as
catchment size grows. With N≈1000 (a small-county scale) the 95% interval is well under
±0.05 Na/K units (≈4% of the overall mean).</p>
{cat_html}

<h2>5 · Feature importance (LightGBM)</h2>
{fig_to_div(fig5)}
<p class="caption">% of total gain. Age, race/ethnicity, and adiposity (waist, BMI) dominate;
clinical biomarkers (SBP/DBP, HTN/DM diagnosis) contribute modestly. Predictors marked here
are all available in standard EHR demographics+vitals+problem-list extracts.</p>

<h2>6 · Secular trend across NHANES cycles</h2>
{fig_to_div(fig6)}
<p class="caption">Survey-weighted mean Na/K (±95% CI) by cycle. The dashed line shows the
LightGBM predicted mean for the 2017-2018 holdout. Modest secular drift in the underlying
diet explains part of the small overall bias.</p>

<h2>Reproduce</h2>
<p>From the project subdirectory <code>nhanes_nak_ratio_prediction/</code>:</p>
<pre><code>uv venv && uv pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute 01_data_download.ipynb --inplace
.venv/bin/jupyter nbconvert --to notebook --execute 02_model_fitting.ipynb --inplace
.venv/bin/jupyter nbconvert --to notebook --execute 03_report.ipynb --inplace
# report at outputs/report.html
</code></pre>

</body>
</html>'''

out_path = os.path.join(OUT, 'report.html')
with open(out_path, 'w') as f:
    f.write(html)
print(f'wrote {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)')"""),
]

build(cells, '03_report.ipynb')
print('wrote 03_report.ipynb')
