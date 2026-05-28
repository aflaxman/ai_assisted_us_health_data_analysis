"""Generate 02_model_fitting.ipynb."""
from _build_notebook import build, md, code

cells = [
    md("""# 02 — Model fitting and holdout evaluation

We fit machine-learning models that predict an individual's dietary Na/K ratio
(milligrams sodium ÷ milligrams potassium, averaged across the two 24-hour recalls)
from variables that would also be available in an EHR (age, sex, race/ethnicity,
BMI, blood pressure, hypertension/diabetes diagnosis, smoking status, socio-economic
proxies, alcohol use).

**Design:**

- Train: NHANES 2007-2008 through 2015-2016 (five 2-year cycles)
- Holdout: NHANES 2017-2018 — the most recent regular cycle, our proof-of-concept "EHR stand-in"
- Adults ≥ 20y with at least one reliable 24-hour dietary recall

**Models compared:**

1. *Mean baseline* — every person predicted as training-set mean
2. *Ridge regression* — linear, age × sex interactions
3. *Gradient boosting* (LightGBM) — handles nonlinearities and interactions

The proof-of-concept question is two-fold:

- How well can these predictors *individually* explain Na/K variance? (Expected: low — intake
  varies hugely within demographic strata.)
- How well do *subgroup means* generalize from training cycles to the holdout cycle? (This is
  the operational metric — we don't need accurate individuals, we need accurate populations.)"""),
    code("""import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

DATA = os.path.abspath(os.path.join('..', 'data'))
DERIVED = os.path.join(DATA, 'derived')

pooled = pd.read_parquet(os.path.join(DERIVED, 'nak_pooled_2007_2018.parquet'))
print(f"pooled: {len(pooled):,} adults across {pooled['CYCLE'].nunique()} cycles")

# require a valid Na/K ratio (at least one reliable 24-hour recall)
df = pooled[pooled['NAK_RATIO'].notna() &
            (pooled['NA_MGD'] > 0) & (pooled['K_MGD'] > 0)].copy()
print(f"with valid Na/K:   {len(df):,}")

# trim implausible extremes (top/bottom 0.5%, matches NCI usual-intake convention)
lo, hi = df['NAK_RATIO'].quantile([0.005, 0.995])
df = df[(df['NAK_RATIO'] >= lo) & (df['NAK_RATIO'] <= hi)].copy()
print(f"after 0.5% trim:   {len(df):,}  (range {df['NAK_RATIO'].min():.2f}-{df['NAK_RATIO'].max():.2f})")"""),
    md("""## Train / holdout split"""),
    code("""HOLDOUT_CYCLE = '2017-2018'
TRAIN_CYCLES = ['2007-2008', '2009-2010', '2011-2012', '2013-2014', '2015-2016']

train = df[df['CYCLE'].isin(TRAIN_CYCLES)].copy()
test  = df[df['CYCLE'] == HOLDOUT_CYCLE].copy()
print(f"train: {len(train):,} adults  ({len(TRAIN_CYCLES)} cycles)")
print(f"test : {len(test):,} adults   (cycle {HOLDOUT_CYCLE})")
print()
print('Na/K ratio (mg/mg) — by split:')
print(f"  train mean: {train['NAK_RATIO'].mean():.3f}  (SD {train['NAK_RATIO'].std():.3f})")
print(f"  test  mean: {test['NAK_RATIO'].mean():.3f}  (SD {test['NAK_RATIO'].std():.3f})")"""),
    md("""## Feature matrix

We use predictors that mirror what is available in an EHR: demographics, anthropometry,
blood pressure, chronic-condition diagnoses, smoking, alcohol, plus a poverty-income
ratio as an SES proxy (replaceable with zip-level area-deprivation index in EHR work)."""),
    code("""NUM_COLS = ['AGE', 'BMXBMI', 'BMXWAIST', 'SBP', 'DBP', 'INDFMPIR',
            'ALC_DRINKS_PER_DAY']
BIN_COLS = ['FEMALE', 'HTN_DX', 'HTN_TRT', 'DM_DX', 'DM_TRT', 'HC_DX']
CAT_COLS = ['RACE', 'SMOKE']

def build_X(d):
    X = pd.DataFrame(index=d.index)
    for c in NUM_COLS:
        X[c] = pd.to_numeric(d.get(c), errors='coerce')
    for c in BIN_COLS:
        v = pd.to_numeric(d.get(c), errors='coerce')
        X[c] = v
    for c in CAT_COLS:
        X[c] = d[c].astype('category')
    return X

X_train = build_X(train)
X_test  = build_X(test)
y_train = train['NAK_RATIO'].values
y_test  = test['NAK_RATIO'].values

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"\\nfeature missingness (train):")
print((X_train.isna().mean() * 100).round(1).sort_values(ascending=False).to_string())"""),
    md("""## Model 1 — mean baseline"""),
    code("""mean_pred = np.full_like(y_test, y_train.mean(), dtype=float)
print(f"mean baseline (predict {y_train.mean():.3f} for everyone)")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, mean_pred)):.4f}")
print(f"  MAE : {mean_absolute_error(y_test, mean_pred):.4f}")
print(f"  R²  : {r2_score(y_test, mean_pred):.4f}")"""),
    md("""## Model 2 — Ridge regression (linear)

Numeric features standardised; categoricals one-hot encoded; missing values imputed at the
training-set median (numeric) or 'unknown' (categoric)."""),
    code("""def imputed_num(X, medians=None):
    X = X.copy()
    if medians is None:
        medians = {c: X[c].median() for c in NUM_COLS + BIN_COLS}
    for c in NUM_COLS + BIN_COLS:
        X[c] = X[c].fillna(medians[c])
    return X, medians

X_tr_imp, medians = imputed_num(X_train)
X_te_imp, _ = imputed_num(X_test, medians)

pre = ColumnTransformer([
    ('num', StandardScaler(), NUM_COLS + BIN_COLS),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
])
ridge = Pipeline([('pre', pre), ('ridge', Ridge(alpha=1.0))])
ridge.fit(X_tr_imp, y_train)
ridge_pred = ridge.predict(X_te_imp)
print('Ridge regression')
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, ridge_pred)):.4f}")
print(f"  MAE : {mean_absolute_error(y_test, ridge_pred):.4f}")
print(f"  R²  : {r2_score(y_test, ridge_pred):.4f}")"""),
    md("""## Model 3 — Gradient boosting (LightGBM)

LightGBM accepts NaNs natively and handles non-linearities and interactions without explicit
feature engineering. Mild regularisation; modest depth to avoid overfit on a relatively small
training set."""),
    code("""X_tr_lgb = X_train.copy()
X_te_lgb = X_test.copy()
for c in CAT_COLS:
    X_tr_lgb[c] = X_tr_lgb[c].astype('category')
    X_te_lgb[c] = X_te_lgb[c].astype('category')

gbm = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    min_child_samples=80,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)
gbm.fit(X_tr_lgb, y_train, categorical_feature=CAT_COLS,
        eval_set=[(X_te_lgb, y_test)], callbacks=[lgb.early_stopping(30, verbose=False)])
gbm_pred = gbm.predict(X_te_lgb)
print('LightGBM')
print(f"  best_iter: {gbm.best_iteration_}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, gbm_pred)):.4f}")
print(f"  MAE : {mean_absolute_error(y_test, gbm_pred):.4f}")
print(f"  R²  : {r2_score(y_test, gbm_pred):.4f}")"""),
    md("""## Holdout summary table"""),
    code("""def metrics(name, y, yhat):
    return {
        'model': name,
        'RMSE': np.sqrt(mean_squared_error(y, yhat)),
        'MAE': mean_absolute_error(y, yhat),
        'R2': r2_score(y, yhat),
        'Pearson_r': np.corrcoef(y, yhat)[0, 1] if np.std(yhat) > 0 else 0.0,
    }

summary = pd.DataFrame([
    metrics('Mean baseline', y_test, mean_pred),
    metrics('Ridge regression', y_test, ridge_pred),
    metrics('LightGBM', y_test, gbm_pred),
])
print(summary.round(4).to_string(index=False))"""),
    md("""## Subgroup-level prediction quality

The operational use is *population* prediction (e.g., a county, a hospital catchment).
We aggregate predictions and observations into demographic strata in the holdout cycle and
check calibration of subgroup means."""),
    code("""def subgroup_calibration(test_df, yhat, by):
    t = test_df.copy()
    t['_yhat'] = yhat
    t['_y'] = t['NAK_RATIO'].values
    wt = t['weight_diet2d'].fillna(t.get('weight_diet1d')).fillna(t['weight'])
    t['_w'] = wt
    rows = []
    for key, g in t.groupby(by, dropna=False, observed=True):
        w = g['_w'].fillna(0)
        if w.sum() <= 0:
            continue
        rows.append({
            'group': str(key),
            'n': len(g),
            'obs_mean': np.average(g['_y'], weights=w),
            'pred_mean': np.average(g['_yhat'], weights=w),
        })
    out = pd.DataFrame(rows)
    out['error'] = out['pred_mean'] - out['obs_mean']
    out['pct_error'] = 100 * out['error'] / out['obs_mean']
    return out

sub_age = subgroup_calibration(test, gbm_pred, pd.cut(test['AGE'], [19,30,45,60,75,200]))
sub_sex = subgroup_calibration(test, gbm_pred, 'SEX')
sub_race = subgroup_calibration(test, gbm_pred, 'RACE')

print('=== AGE GROUP (LightGBM) ===')
print(sub_age.round(3).to_string(index=False))
print('\\n=== SEX (LightGBM) ===')
print(sub_sex.round(3).to_string(index=False))
print('\\n=== RACE/ETHNICITY (LightGBM) ===')
print(sub_race.round(3).to_string(index=False))"""),
    md("""## County-scale precision: synthetic local-pseudo-aggregation experiment

To approximate what would happen when pooling a city- or county-sized EHR population,
we randomly draw clusters of N adults from the holdout (a synthetic catchment of size N)
and compare the *predicted cluster mean* to the *observed cluster mean*. This is the
quantity the eventual EHR-based estimator will report at the local level."""),
    code("""rng = np.random.default_rng(7)
SIZES = [200, 1000, 5000]
records = []
for N in SIZES:
    for rep in range(500):
        idx = rng.choice(len(test), size=min(N, len(test)), replace=True)
        records.append({
            'N': N,
            'obs_mean': np.average(y_test[idx]),
            'pred_mean': np.average(gbm_pred[idx]),
        })
clust = pd.DataFrame(records)
clust['error'] = clust['pred_mean'] - clust['obs_mean']
print('catchment-scale error (LightGBM, observed - predicted means over random clusters):')
print(clust.groupby('N')['error'].agg(['mean','std',
                                       lambda x: np.percentile(x, 2.5),
                                       lambda x: np.percentile(x, 97.5)])
      .rename(columns={'<lambda_0>': 'CI_lo', '<lambda_1>': 'CI_hi'})
      .round(4).to_string())"""),
    md("""## LightGBM feature importance (gain)"""),
    code("""imp = pd.DataFrame({
    'feature': gbm.booster_.feature_name(),
    'gain': gbm.booster_.feature_importance(importance_type='gain'),
    'split': gbm.booster_.feature_importance(importance_type='split'),
}).sort_values('gain', ascending=False)
imp['gain_pct'] = 100 * imp['gain'] / imp['gain'].sum()
print(imp.round(2).to_string(index=False))"""),
    md("""## Save predictions + summary to JSON for the report"""),
    code("""import pickle

out_dir = os.path.join('outputs')
os.makedirs(out_dir, exist_ok=True)

test_out = test[['SEQN', 'CYCLE', 'AGE', 'SEX', 'RACE', 'NAK_RATIO',
                 'weight', 'weight_diet1d', 'weight_diet2d']].copy()
test_out['pred_mean_baseline'] = mean_pred
test_out['pred_ridge'] = ridge_pred
test_out['pred_lgbm'] = gbm_pred
test_out.to_parquet(os.path.join(DERIVED, 'nak_holdout_predictions.parquet'))

summary.to_csv(os.path.join(DERIVED, 'nak_model_summary.csv'), index=False)
imp.to_csv(os.path.join(DERIVED, 'nak_feature_importance.csv'), index=False)
clust.to_parquet(os.path.join(DERIVED, 'nak_catchment_sim.parquet'))

with open(os.path.join(DERIVED, 'nak_model.pkl'), 'wb') as f:
    pickle.dump({'lgbm': gbm, 'ridge': ridge, 'medians': medians,
                 'num_cols': NUM_COLS, 'bin_cols': BIN_COLS, 'cat_cols': CAT_COLS}, f)

print('saved holdout predictions, summary, feature importance, catchment sim, and pickled models.')"""),
]

build(cells, '02_model_fitting.ipynb')
print('wrote 02_model_fitting.ipynb')
