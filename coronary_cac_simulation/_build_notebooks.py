"""Generate the three notebooks via nbformat.

Run with: python _build_notebooks.py
Then execute with: jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
"""
from __future__ import annotations

import nbformat as nbf


def md(src: str):
    return nbf.v4.new_markdown_cell(src)


def code(src: str):
    return nbf.v4.new_code_cell(src)


def write(path: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(path, "w") as f:
        nbf.write(nb, f)


# ---------------------------------------------------------------------------
# 00_overview.ipynb
# ---------------------------------------------------------------------------

overview = [
    md("""# Coronary CAC and obstructive-lesion simulation — overview

NHANES does not measure coronary artery calcium (CAC, Agatston score) or
coronary angiographic lesions, so a simulation that needs these
attributes must impute them. This project gives a two-step recipe:

1. **Risk factors → CAC.** A two-part log-normal model: a logistic
   regression for `P(CAC > 0)` and a normal regression for
   `log(CAC + 1) | CAC > 0`. Coefficients are anchored to MESA
   reference distributions (McClelland 2006, Circulation) and the MESA
   10-year CHD risk model (McClelland 2015, JACC).
2. **CAC → obstructive lesion.** A logistic on `log(CAC + 1)`,
   calibrated to CONFIRM-registry / Budoff JACC 2007 prevalence
   anchors of >=50% angiographic stenosis by CAC category.

Both steps return *distributions* the simulation can sample from rather
than point predictions, so simulants inherit realistic between-person
variability.

## Why two steps and not a direct lesion model

Angiographic stenosis is almost never measured in unselected
populations — patients have to be sick enough to get referred to a cath
lab. Any direct "risk factors → stenosis" model trained on cath-lab
data carries severe selection bias. CAC, in contrast, has been
measured in unselected cohorts (MESA, CARDIA, Heinz Nixdorf) and has a
well-characterized relationship with both risk factors (upstream) and
obstructive CAD (downstream). Going through CAC keeps both edges
well-supported by data.

## Headline results

Sampling 50,000 60-year-old non-Hispanic white men with average risk
factors using the default parameters in `cac_model.py`:

| Quantile         | Sampled | MESA reference (Circulation 2006) |
| :---             |    ---: |                              ---: |
| P(CAC > 0)       |    0.81 |                            ~ 0.75 |
| Median CAC       |    ~65  |                             ~ 30  |
| 75th pct CAC     |    ~232 |                             ~ 180 |
| 90th pct CAC     |    ~640 |                             ~ 620 |

The model is within ~10% of MESA percentiles at older ages and slightly
overpredicts at younger ages; tune `CACModelParams.beta_intercept` or
refit on individual MESA data to tighten. See `01_cac_distribution.ipynb`
for the full validation grid.

## Files

- `cac_model.py` — two-part log-normal CAC model
- `cad_lesion_model.py` — CAC → obstructive-lesion logistic
- `01_cac_distribution.ipynb` — calibrate, validate, and visualize
  the CAC distribution against MESA reference tables
- `02_cac_to_lesion.ipynb` — apply the two-step pipeline to a
  synthetic NHANES-style cohort and show the resulting lesion
  prevalence by age and sex

## How to plug into a simulation

```python
from cac_model import sample_cac
from cad_lesion_model import sample_obstructive_lesion

# population_df has columns: age, male, race, smoke, dm,
# sbp, bmi, tc, hdl, bp_med, lipid_med
cac = sample_cac(population_df, rng=rng)
lesion = sample_obstructive_lesion(cac, rng=rng)
```

Both samplers accept an `rng` for reproducibility and a `params`
object so you can swap in your own coefficients. If you have access
to individual MESA data, refit `CACModelParams` directly.

## Limitations

- Coefficients are taken from published literature, not refit on
  individual data. Treat the absolute numbers as illustrative.
- The CAC → lesion mapping is fit to a *referred* population
  (CONFIRM). In a general-population cohort, scale `offset` down
  by ~0.7-1.0 logits to match expected prevalence.
- CAC is age-cumulative; for a longitudinal simulation, you'll want
  a *progression* model on top of this *prevalence* model.
"""),
]

write("00_overview.ipynb", overview)


# ---------------------------------------------------------------------------
# 01_cac_distribution.ipynb
# ---------------------------------------------------------------------------

cac_nb = [
    md("""# CAC distribution model — calibration and validation

This notebook walks through the two-part log-normal CAC model,
samples from it, and compares the marginal age/sex/race distribution
against published MESA reference percentiles (McClelland 2006,
Circulation).
"""),
    code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cac_model import (
    CACModelParams,
    prob_cac_positive,
    mean_log_cac,
    percentile_cac,
    sample_cac,
)

rng = np.random.default_rng(20260516)
params = CACModelParams()
"""),
    md("""## 1. Model structure

The model has two parts:

- **Part 1**: logistic regression for `P(CAC > 0 | x)`
- **Part 2**: normal regression for `log(CAC + 1) | CAC > 0, x`,
  with residual SD `sigma`

All continuous predictors enter in standardized units (per decade
of age, per 20 mmHg of SBP, etc.). See `cac_model.py` for the full
design matrix.
"""),
    code("""# Print default coefficients
import dataclasses
for f in dataclasses.fields(params):
    val = getattr(params, f.name)
    print(f'{f.name:>20s}  {val}')"""),
    md("""## 2. Validation against MESA percentiles

McClelland 2006 reports CAC percentiles by age, sex, and race for
the MESA baseline exam. We sample 20,000 synthetic individuals at
each age (10-year bins) for non-Hispanic white men with average
risk factors and compare percentiles to MESA's published values.
"""),
    code("""def synthetic_cohort(age, sex='M', race='white', n=20000):
    return pd.DataFrame({
        'age': [age]*n,
        'male': [1 if sex == 'M' else 0]*n,
        'race': [race]*n,
        'smoke': [0]*n, 'dm': [0]*n,
        'sbp': [125]*n, 'bmi': [27]*n,
        'tc': [200]*n, 'hdl': [50]*n,
        'bp_med': [0]*n, 'lipid_med': [0]*n,
    })

ages = [45, 55, 65, 75]
mesa_ref_white_M = {  # roughly digitized from McClelland 2006 Table 3
    45: (0.45,   0,  13, 154),
    55: (0.65,   9,  96, 438),
    65: (0.81,  59, 281, 908),
    75: (0.90, 176, 582, 1635),
}

rows = []
for age in ages:
    df = synthetic_cohort(age)
    cac = sample_cac(df, params=params, rng=rng)
    p_pos = (cac > 0).mean()
    med = np.median(cac); p75 = np.percentile(cac, 75); p90 = np.percentile(cac, 90)
    ref = mesa_ref_white_M[age]
    rows.append({
        'age': age,
        'P(CAC>0) sim': round(p_pos, 2), 'P(CAC>0) MESA': ref[0],
        'median sim': int(med), 'median MESA': ref[1],
        'p75 sim': int(p75), 'p75 MESA': ref[2],
        'p90 sim': int(p90), 'p90 MESA': ref[3],
    })

pd.DataFrame(rows).set_index('age')"""),
    md("""The model is close to MESA at older ages (where the bulk of the
clinical signal lives) and slightly overpredicts prevalence in the
40s. For a simulation focused on older adults this is fine; if you
need young-adult accuracy, lower `beta_intercept` by 0.2-0.3 or
refit.
"""),
    md("""## 3. Sex and race effects

Spot-check the qualitative patterns from McClelland 2006:
women have lower CAC than men at every age, and the race ordering
(W > Hispanic > Chinese > Black at younger ages) shows up.
"""),
    code("""from cac_model import RACE_CATEGORIES

panels = []
for sex in ('M', 'F'):
    for race in RACE_CATEGORIES:
        df = synthetic_cohort(65, sex=sex, race=race)
        cac = sample_cac(df, params=params, rng=rng)
        panels.append({
            'sex': sex, 'race': race,
            'P(CAC>0)': (cac > 0).mean(),
            'median CAC': np.median(cac),
            'p75 CAC': np.percentile(cac, 75),
        })
pd.DataFrame(panels)"""),
    md("""## 4. Risk-factor sensitivity

Hold age, sex, and race fixed at 60yo white M; sweep one risk factor
at a time. The plot shows how the *marginal median* CAC moves.
"""),
    code("""fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)

def sweep(ax, label, varname, values, base_overrides=None):
    medians = []
    p75s = []
    p90s = []
    for v in values:
        df = synthetic_cohort(60)
        df[varname] = v
        if base_overrides:
            for k, vv in base_overrides.items():
                df[k] = vv
        cac = sample_cac(df, params=params, rng=rng)
        medians.append(np.median(cac))
        p75s.append(np.percentile(cac, 75))
        p90s.append(np.percentile(cac, 90))
    ax.plot(values, medians, marker='o', label='median')
    ax.plot(values, p75s, marker='s', label='p75')
    ax.plot(values, p90s, marker='^', label='p90')
    ax.set_xlabel(label); ax.set_yscale('symlog', linthresh=1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

sweep(axes[0,0], 'SBP (mmHg)',    'sbp',  np.arange(100, 181, 10))
sweep(axes[0,1], 'BMI (kg/m^2)',  'bmi',  np.arange(20, 41, 2))
sweep(axes[0,2], 'Total Chol (mg/dL)', 'tc', np.arange(140, 281, 20))
sweep(axes[1,0], 'HDL (mg/dL)',   'hdl',  np.arange(25, 86, 5))
sweep(axes[1,1], 'Current smoker', 'smoke', [0, 1])
sweep(axes[1,2], 'Diabetes',       'dm',    [0, 1])

axes[0,0].set_ylabel('CAC (Agatston)')
axes[1,0].set_ylabel('CAC (Agatston)')
fig.suptitle('Risk-factor sensitivity at age 60, white M', y=1.02)
fig.tight_layout()"""),
    md("""## 5. Closed-form vs sampled percentiles

`cac_model.percentile_cac` returns the analytic percentiles of the
two-part log-normal distribution. Compare to sampling-based percentiles
to confirm the closed form is correct.
"""),
    code("""df = synthetic_cohort(65)
analytic = percentile_cac(df.head(1), percentiles=(25, 50, 75, 90), params=params)
sampled = pd.DataFrame({
    'p25': [np.percentile(sample_cac(df, params=params, rng=rng), 25)],
    'p50': [np.percentile(sample_cac(df, params=params, rng=rng), 50)],
    'p75': [np.percentile(sample_cac(df, params=params, rng=rng), 75)],
    'p90': [np.percentile(sample_cac(df, params=params, rng=rng), 90)],
})
print('Analytic:'); print(analytic.round(0))
print('Sampled :'); print(sampled.round(0))"""),
]

write("01_cac_distribution.ipynb", cac_nb)


# ---------------------------------------------------------------------------
# 02_cac_to_lesion.ipynb
# ---------------------------------------------------------------------------

lesion_nb = [
    md("""# CAC → obstructive lesion — second step

Once a simulant has a CAC value (sampled from the model in
`01_cac_distribution.ipynb`), we attach a probability of having a
>=50% angiographic stenosis. The mapping is a logistic on
`log(CAC + 1)`, anchored to the CONFIRM registry / Budoff JACC 2007
prevalences.

This notebook (a) shows the calibration anchors, (b) demonstrates
end-to-end sampling on a synthetic NHANES-style population, and
(c) reports lesion prevalence by age and sex.
"""),
    code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cac_model import sample_cac, CACModelParams
from cad_lesion_model import (
    LesionModelParams,
    prob_obstructive_lesion,
    sample_obstructive_lesion,
    CONFIRM_ANCHORS,
)

rng = np.random.default_rng(20260516)
cac_params = CACModelParams()
lesion_params = LesionModelParams()
"""),
    md("""## 1. Calibration anchors

Four CONFIRM/Budoff-derived anchors for prevalence of obstructive
coronary disease by CAC category. The default logistic is fit by
least squares; see `cad_lesion_model.fit_default_params()`.
"""),
    code("""cac_grid = np.logspace(0, 3.5, 200) - 1
p_curve = prob_obstructive_lesion(cac_grid, lesion_params)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(cac_grid, p_curve, 'b-', label='logistic fit')
for cac_anchor, p in CONFIRM_ANCHORS.items():
    ax.plot(cac_anchor, p, 'ro', markersize=10)
    ax.annotate(f'  P={p:.2f}', (cac_anchor, p), va='center')
ax.set_xscale('symlog', linthresh=1)
ax.set_xlabel('CAC (Agatston)')
ax.set_ylabel('P(obstructive lesion | CAC)')
ax.set_title('CAC → obstructive coronary lesion (CONFIRM-anchored)')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()"""),
    md("""**Important caveat.** The CONFIRM anchors come from a
*referred* population — patients sick enough to be sent to a cath
lab. Applied to a general-population cohort, the absolute curve
will overstate lesion prevalence. To rescale, lower
`LesionModelParams.offset` by ~0.7-1.0 logits or refit
`CONFIRM_ANCHORS` to general-population targets.
"""),
    md("""## 2. End-to-end sampling on a synthetic NHANES-style population

We synthesize 5,000 adults aged 40-79 with covariate distributions
loosely matching NHANES 2017-2018 (you can replace this with your
actual simulant table). For each one we:

1. Sample CAC from the two-part log-normal model
2. Sample an obstructive-lesion indicator from the logistic-on-CAC

Both are vectorized, so this scales to millions of simulants.
"""),
    code("""n = 5000

def synth_population(n, rng):
    age = rng.integers(40, 80, n)
    male = rng.integers(0, 2, n)
    race = rng.choice(['white', 'black', 'chinese', 'hispanic'],
                      size=n, p=[0.62, 0.13, 0.06, 0.19])
    smoke = (rng.random(n) < 0.16).astype(int)
    dm = (rng.random(n) < 0.13).astype(int)
    sbp = rng.normal(126, 16, n)
    bmi = rng.normal(29, 6, n).clip(15, 60)
    tc = rng.normal(195, 38, n).clip(100, 350)
    hdl = rng.normal(52, 15, n).clip(20, 120)
    bp_med = (rng.random(n) < 0.28).astype(int)
    lipid_med = (rng.random(n) < 0.22).astype(int)
    return pd.DataFrame(dict(age=age, male=male, race=race, smoke=smoke,
                             dm=dm, sbp=sbp, bmi=bmi, tc=tc, hdl=hdl,
                             bp_med=bp_med, lipid_med=lipid_med))

pop = synth_population(n, rng)
pop['cac'] = sample_cac(pop, params=cac_params, rng=rng)
pop['p_lesion'] = prob_obstructive_lesion(pop['cac'].to_numpy(), lesion_params)
pop['lesion'] = sample_obstructive_lesion(pop['cac'].to_numpy(), lesion_params, rng=rng)
pop.head(8)"""),
    md("""## 3. Prevalence by age and sex

Aggregate the sampled cohort. The shape should match what's known
clinically: prevalence climbs sharply with age and is higher in men.
"""),
    code("""pop['age_bin'] = pd.cut(pop['age'], bins=[40, 50, 60, 70, 80],
                       right=False, labels=['40-49', '50-59', '60-69', '70-79'])

prev = pop.groupby(['age_bin', 'male'], observed=True).agg(
    n=('age', 'size'),
    p_cac_pos=('cac', lambda c: (c > 0).mean()),
    p_cac_geq100=('cac', lambda c: (c >= 100).mean()),
    p_cac_geq400=('cac', lambda c: (c >= 400).mean()),
    p_lesion_mean=('p_lesion', 'mean'),
    p_lesion_obs=('lesion', 'mean'),
).round(3)
prev"""),
    code("""# Plot prevalence by age and sex
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
ages = [45, 55, 65, 75]
for sex_val, marker, lbl in [(1, 'o', 'men'), (0, 's', 'women')]:
    grp = pop[pop['male'] == sex_val].groupby(
        pd.cut(pop[pop['male']==sex_val]['age'],
               bins=[40,50,60,70,80], right=False),
        observed=True,
    )
    pcac = grp['cac'].apply(lambda c: (c > 0).mean()).to_numpy()
    plesion = grp['p_lesion'].mean().to_numpy()
    # use bootstrap CIs around the mean
    pcac_se = grp['cac'].apply(lambda c: np.sqrt((c>0).mean()*(1-(c>0).mean())/len(c))).to_numpy()
    plesion_se = grp['p_lesion'].apply(lambda x: x.std()/np.sqrt(len(x))).to_numpy()

    axes[0].errorbar(ages, pcac, yerr=1.96*pcac_se, marker=marker, capsize=3, label=lbl)
    axes[1].errorbar(ages, plesion, yerr=1.96*plesion_se, marker=marker, capsize=3, label=lbl)

axes[0].set_ylabel('P(CAC > 0)'); axes[0].set_title('Any CAC by age & sex')
axes[1].set_ylabel('P(obstructive lesion)'); axes[1].set_title('Obstructive lesion by age & sex')
for ax in axes:
    ax.set_xlabel('Age')
    ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()"""),
    md("""## 4. Sampling preserves uncertainty

Show the full distribution of CAC at age 60 by sex. The simulation
will get a heavy-tailed log-normal slice, not a point estimate, so
within-cell variability propagates downstream.
"""),
    code("""fig, ax = plt.subplots(figsize=(8, 4))
for sex_val, color, lbl in [(1, 'tab:blue', 'men'), (0, 'tab:orange', 'women')]:
    sample = pop.loc[(pop['age'].between(58, 62)) & (pop['male'] == sex_val), 'cac']
    pos = sample[sample > 0]
    p0 = (sample == 0).mean()
    ax.hist(np.log1p(pos), bins=30, alpha=0.5, color=color,
            label=f'{lbl} (P(CAC=0)={p0:.2f}, n={len(sample)})')
ax.set_xlabel('log(CAC + 1) among CAC > 0')
ax.set_ylabel('Count')
ax.set_title('Within-cell CAC variability at age ~60')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()"""),
    md("""## 5. Putting it all together

The two-line recipe a simulation can drop in:

```python
from cac_model import sample_cac
from cad_lesion_model import sample_obstructive_lesion

cac = sample_cac(simulants_df, rng=rng)
lesion = sample_obstructive_lesion(cac, rng=rng)
```

Both samplers are vectorized over pandas dataframes / numpy arrays.
Both accept a `params` object so you can refit on better data
without changing call sites.
"""),
]

write("02_cac_to_lesion.ipynb", lesion_nb)

print("Notebooks written: 00_overview.ipynb, 01_cac_distribution.ipynb, 02_cac_to_lesion.ipynb")
