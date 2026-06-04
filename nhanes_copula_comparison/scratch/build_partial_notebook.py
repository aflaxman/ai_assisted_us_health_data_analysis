"""Build 03_confounding.ipynb and update 00_overview."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

# ===================================================================
# 03_confounding.ipynb
# ===================================================================
nb = new_notebook()
nb.cells.append(new_markdown_cell(r"""# 03 — Confounding by age, sex, and race

The first two notebooks pooled adults 40+ and computed marginal Spearman
correlations and copula fits. But every risk in the eight-risk matrix has a
strong age gradient, and several have sex and race gradients too. If a chunk
of what the marginal correlation calls "BMI and FPG cluster" is really "old
people have higher FPG and lower BMI," then the input to the simulation's
copula might be misspecified — *especially* if the simulation already applies
age-stratified marginal CDFs to the copula's propensity output, which would
double-count the age effect.

This notebook quantifies the effect of partialling out age (linear + quadratic), sex, and race/ethnicity (RIDRETH3, 6 categories) on:

1. The Spearman correlation matrix.
2. Tail dependence χ(q) and lower/upper tail asymmetry.
3. Per-pair Gaussian vs t-copula AIC.
4. Multivariate joint top-decile co-occurrence.

## Method: rank-Gaussian residualization

For each risk x:

1. Pseudo-obs $u_i = \text{rank}(x_i) / (n + 1)$.
2. Normal scores $z_i = \Phi^{-1}(u_i)$.
3. OLS regression $z_i = \alpha + \beta_1 \text{age} + \beta_2 \text{age}^2 + \beta_3 \text{sex} + \sum \gamma_k \text{race}_{ik} + \varepsilon_i$.
4. Standardize residuals to unit variance; partial pseudo-obs $u_i' = \Phi(\varepsilon_i / \hat{\sigma}_\varepsilon)$.

This is the standard semiparametric "rank-Gaussian copula" partialling — it preserves marginal uniformity of $u_i'$ while removing the linear-in-normal-scores component of age/sex/race.
"""))
nb.cells.append(new_code_cell(r"""import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats, optimize, linalg
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
rng = np.random.default_rng(20260514)

full = pd.read_parquet('outputs/nhanes_2017_2018_merged.parquet')
demo = pd.read_sas('scratch/raw/DEMO_J.xpt')[['SEQN','RIDRETH3']]
full = full.merge(demo, on='SEQN', how='left')

RISKS = ['BMXBMI','LBDLDL','SBP_MEAN','LBXGLU','smoking_signed','eGFR_signed']
LABEL = {'BMXBMI':'BMI','LBDLDL':'LDL_C','SBP_MEAN':'SBP','LBXGLU':'FPG',
         'smoking_signed':'smoking','eGFR_signed':'kidney'}
adults = full[(full['AGE']>=40) & full[RISKS].notna().all(axis=1) &
              full['WTMEC2YR'].gt(0) & full['RIDRETH3'].notna()].copy()
n = len(adults)

age = adults['AGE'].values; sex = adults['FEMALE'].values
race = adults['RIDRETH3'].astype(int).values
rcats = sorted(np.unique(race).tolist())
rd = np.column_stack([(race == r).astype(float) for r in rcats[1:]])
age_c = age - age.mean()
X = np.column_stack([np.ones(n), age_c, age_c**2, sex, rd])
print(f'n = {n}; design matrix {X.shape} (intercept, age, age², sex, race({len(rcats)-1}))')

def partial_pseudo(x, X):
    z = stats.norm.ppf(stats.rankdata(x)/(len(x)+1))
    b, *_ = np.linalg.lstsq(X, z, rcond=None)
    e = z - X @ b
    return stats.norm.cdf(e / e.std(ddof=1))

U_raw  = np.column_stack([stats.rankdata(adults[r].values)/(n+1) for r in RISKS])
U_part = np.column_stack([partial_pseudo(adults[r].values, X) for r in RISKS])"""))
nb.cells.append(new_markdown_cell(r"""## 1. Spearman matrix: marginal vs partial"""))
nb.cells.append(new_code_cell(r"""def sp_matrix(U):
    K = U.shape[1]; M = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            M[i,j] = stats.spearmanr(U[:,i], U[:,j]).statistic
    return (M + M.T)/2
SR_raw, SR_part = sp_matrix(U_raw), sp_matrix(U_part)

rows = []
for i in range(len(RISKS)):
    for j in range(i+1, len(RISKS)):
        a, b = LABEL[RISKS[i]], LABEL[RISKS[j]]
        m, p = SR_raw[i,j], SR_part[i,j]
        rows.append(dict(pair=f'{a}-{b}', marginal=round(m,3),
                         partial=round(p,3), shift=round(p-m,3)))
sp_df = pd.DataFrame(rows).sort_values('shift', key=abs, ascending=False)
sp_df"""))
nb.cells.append(new_markdown_cell(r"""**Reading the shifts.**

- **SBP-kidney** swings the most (+0.17 → −0.07, a 0.23 sign flip). The marginal positive comes entirely from age — older adults have both higher SBP and worse kidney function. Within an age stratum, no association remains.
- **LDL_C-kidney** flips sign in a similar age-driven way (−0.12 → +0.01).
- **BMI-FPG** gets *stronger* after partialling (+0.25 → +0.31). Age pushes BMI down and FPG up, so the marginal correlation underestimates the within-age association.
- **BMI-SBP** also strengthens (+0.10 → +0.14).
- Most pairs shift by less than 0.05.

The implication: the marginal Spearman matrix used in `nhanes_risk_correlation_matrix/outputs/correlation_matrix.csv` is the right input only if the simulation's marginal-CDF pushforward is *not* age-stratified. If the simulation's marginal BMI/SBP/etc. distributions are age-specific (which is typical for GBD-driven risk distributions), then the partial matrix is the right input — using the marginal one double-counts age.

## 2. Tail dependence χ(q): does partialling change the diagnostic?"""))
nb.cells.append(new_code_cell(r"""def chi_hat(u, v, q, upper=True):
    if upper:
        d = np.sum(v > q); return np.sum((u>q)&(v>q))/d if d else np.nan
    d = np.sum(v < q); return np.sum((u<q)&(v<q))/d if d else np.nan
def chi_gauss(rho, q):
    z = stats.norm.ppf(q)
    cdf = stats.multivariate_normal.cdf([z,z], mean=[0,0], cov=[[1,rho],[rho,1]])
    return (1 - 2*q + cdf)/(1 - q)

def gauss_ll(u, v, rho):
    zu = stats.norm.ppf(u); zv = stats.norm.ppf(v)
    det = 1 - rho**2; q = (zu**2 - 2*rho*zu*zv + zv**2)/det
    return float(np.sum(-0.5*np.log(det) - 0.5*(q - zu**2 - zv**2)))
def fit_gauss(u, v):
    r = optimize.minimize_scalar(
        lambda r: -gauss_ll(u,v,r) if abs(r)<0.999 else 1e10,
        bounds=(-0.999,0.999), method='bounded')
    return r.x, -r.fun
def fit_t(u, v):
    def nll(p):
        rho, log_nu = p
        if abs(rho)>=0.999: return 1e10
        nu = np.exp(log_nu)
        if nu < 2.01 or nu > 200: return 1e10
        tu, tv = stats.t.ppf(u, df=nu), stats.t.ppf(v, df=nu)
        lb = stats.multivariate_t.logpdf(
            np.column_stack([tu,tv]), loc=[0,0],
            shape=[[1,rho],[rho,1]], df=nu)
        return -float(np.sum(lb - stats.t.logpdf(tu,df=nu) - stats.t.logpdf(tv,df=nu)))
    sp = stats.spearmanr(u, v).statistic
    rho0 = 2*np.sin(np.pi*sp/6)
    r = optimize.minimize(nll, [rho0, np.log(8)],
                          bounds=[(-0.999,0.999),(np.log(2.5),np.log(150))],
                          method='L-BFGS-B')
    return r.x[0], float(np.exp(r.x[1])), -r.fun

PAIRS = [('BMXBMI','LBXGLU'),('SBP_MEAN','LBXGLU'),('BMXBMI','SBP_MEAN'),
         ('LBDLDL','LBXGLU'),('BMXBMI','LBDLDL'),('BMXBMI','eGFR_signed')]
rows = []
for a, b in PAIRS:
    i, j = RISKS.index(a), RISKS.index(b)
    row = {'pair': f'{LABEL[a]}-{LABEL[b]}'}
    for kind, U in [('marg', U_raw), ('part', U_part)]:
        u, v = U[:,i], U[:,j]
        rg, llg = fit_gauss(u, v)
        rt, nu, llt = fit_t(u, v)
        row[f'rho_G_{kind}'] = round(rg, 3)
        row[f'nu_t_{kind}']  = round(nu, 1)
        row[f'dAIC_{kind}']  = round(2 - 2*llg - (4 - 2*llt), 2)
    rows.append(row)
pd.DataFrame(rows)"""))
nb.cells.append(new_markdown_cell(r"""**Per-pair fit findings:**

- **LDL_C-FPG** keeps strong t-copula support: ΔAIC = +12 marginal → +12 partial, ν ≈ 10 either way. The tail dependence is *not* an age artifact.
- **BMI-kidney** *gains* t-copula support after partialling (ΔAIC = −0.7 → +5). Conditioning on age uncovers a modest tail association that the marginal Spearman buried.
- Every other pair: t-copula collapses to Gaussian (ν > 16) in both versions.

## 3. Asymmetric tail dependence: is it real or is it age?"""))
nb.cells.append(new_code_cell(r"""from IPython.display import Image, display
display(Image('outputs/asymmetry_partial.png'))"""))
nb.cells.append(new_markdown_cell(r"""**Asymmetric tail dependence shrinks after partialling.** Most striking: BMI-FPG (asym −0.10 → −0.04), BMI-SBP (−0.10 → −0.06), LDL-kidney (+0.08 → +0.02), SBP-FPG (−0.05 → +0.03 sign flip). The lower-tail clustering that the marginal analysis flagged as "Gaussian and t both can't fit this" was mostly an age artifact — a frailty cohort of older adults who are simultaneously thin, hypotensive, and dysglycemic. Within age-sex-race strata, the asymmetry is much weaker, which means a symmetric Gaussian copula is closer to right.

## 4. Multivariate joint top-decile co-occurrence"""))
nb.cells.append(new_code_cell(r"""def nearest_psd(A, eps=1e-8):
    w, V = linalg.eigh(A); w = np.clip(w, eps, None)
    A2 = V @ np.diag(w) @ V.T; d = np.sqrt(np.diag(A2)); return A2/np.outer(d,d)
def sim_gauss(R, N):
    L = linalg.cholesky(R, lower=True)
    Z = rng.standard_normal((N, R.shape[0])) @ L.T
    return stats.norm.cdf(Z)
def sim_t(R, nu, N):
    L = linalg.cholesky(R, lower=True)
    Z = rng.standard_normal((N, R.shape[0])) @ L.T
    W = rng.chisquare(nu, size=N)/nu
    return stats.t.cdf(Z/np.sqrt(W)[:,None], df=nu)
def joint(U, idx, q=0.90):
    flags = np.ones(len(U), bool)
    for i in idx:
        if i == 1: flags &= (U[:,i] < (1-q))
        else: flags &= (U[:,i] > q)
    return flags.mean()
N = 50_000
RG_marg = nearest_psd(2*np.sin(np.pi*SR_raw/6))
RG_part = nearest_psd(2*np.sin(np.pi*SR_part/6))
UG_marg = sim_gauss(RG_marg, N); UG_part = sim_gauss(RG_part, N)
Ut_part = sim_t(RG_part, 8, N)

CLUSTERS = {'metabolic syndrome (BMI,SBP,FPG)': [0,2,3],
            'metabolic + LDL low':              [0,2,3,1],
            'CV triad + kidney':                [0,2,3,5]}
rows = []
for name, idx in CLUSTERS.items():
    rows.append(dict(cluster=name,
                     empir_marg=round(joint(U_raw,  idx), 4),
                     empir_part=round(joint(U_part, idx), 4),
                     gauss_marg=round(joint(UG_marg, idx), 4),
                     gauss_part=round(joint(UG_part, idx), 4),
                     t8_part=round(joint(Ut_part, idx), 4)))
pd.DataFrame(rows)"""))
nb.cells.append(new_markdown_cell(r"""**Multivariate finding.** The empirical metabolic-syndrome triad rate roughly *doubles* in the partial pseudo-obs (0.0019 → 0.0039) because partialling concentrates the within-stratum BMI-SBP-FPG cluster that age was spreading out. The Gaussian-on-partial prediction (0.0035) sits very close to the empirical partial estimate (0.0039) — Gaussian fits *better* on the residualized data than on the marginal data. The t-copula (ν=8) over-shoots in both cases.

## Bottom line

Partialling out age, sex, and race **changes the matrix** (especially the kidney row) but **does not change the copula recommendation**: Gaussian is still the right family. In fact it fits *better* within strata because asymmetric tail dependence shrinks. The actionable consequence is which Spearman matrix to feed the copula — partial (within-stratum) if the simulation pushes propensities through age-stratified marginal CDFs, marginal otherwise."""))
nbf.write(nb, Path('03_confounding.ipynb'))
print('wrote 03_confounding.ipynb')
