"""Effect of partialling out age, sex, race on:
  - the Spearman matrix
  - tail dependence chi(q)
  - Gaussian vs t-copula AIC
  - multivariate joint extremes

Approach: rank-Gaussian residualization (semiparametric).
  1. u_i = rank(x_i)/(n+1)   pseudo-obs
  2. z_i = Phi^{-1}(u_i)
  3. fit z_i ~ age + age^2 + sex + race_dummies  (OLS)
  4. e_i = residual / sd(residual)
  5. new pseudo-obs u_i' = Phi(e_i)
The new u_i' have approximately uniform marginals with age/sex/race
removed; correlations among them are the partial copula correlations.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats, optimize, linalg
import warnings; warnings.filterwarnings('ignore')

rng = np.random.default_rng(20260514)

full = pd.read_parquet('outputs/nhanes_2017_2018_merged.parquet')

# Merge race (RIDRETH3)
demo = pd.read_sas('scratch/raw/DEMO_J.xpt')[['SEQN','RIDRETH3']]
full = full.merge(demo, on='SEQN', how='left')

RISKS = ['BMXBMI','LBDLDL','SBP_MEAN','LBXGLU','smoking_signed','eGFR_signed']
LABEL = {'BMXBMI':'BMI','LBDLDL':'LDL_C','SBP_MEAN':'SBP','LBXGLU':'FPG',
         'smoking_signed':'smoking','eGFR_signed':'kidney'}

adults = full[(full['AGE']>=40) & full[RISKS].notna().all(axis=1) &
              full['WTMEC2YR'].gt(0) & full['RIDRETH3'].notna()].copy()
n = len(adults)
print(f'n = {n} (adults 40+, all 6 risks, race available)')

# Build design matrix for confounder regression
age = adults['AGE'].values
sex = adults['FEMALE'].values  # 1=female, 0=male
race = adults['RIDRETH3'].astype(int).values
race_categories = sorted(np.unique(race).tolist())
print(f'race categories present: {race_categories}')
race_dummies = np.column_stack([(race == r).astype(float) for r in race_categories[1:]])  # drop ref

age_c = age - age.mean()
X_conf = np.column_stack([
    np.ones(n), age_c, age_c**2, sex, race_dummies
])
print(f'design matrix: {X_conf.shape}, columns = intercept, age, age^2, sex, race(k-1)')

def rank_gaussian(x):
    """Empirical CDF -> standard normal scores."""
    u = stats.rankdata(x) / (len(x) + 1)
    return stats.norm.ppf(u)

def partial_pseudo(x, X_conf):
    """Residualize standard-normal scores against confounders;
    return new pseudo-obs in (0,1)."""
    z = rank_gaussian(x)
    beta, *_ = np.linalg.lstsq(X_conf, z, rcond=None)
    e = z - X_conf @ beta
    e = e / e.std(ddof=1)
    return stats.norm.cdf(e)

U_raw  = np.column_stack([stats.rankdata(adults[r].values)/(n+1) for r in RISKS])
U_part = np.column_stack([partial_pseudo(adults[r].values, X_conf) for r in RISKS])

# ============================================================
# 1. Compare Spearman matrices
# ============================================================
def sp_matrix(U):
    K = U.shape[1]
    M = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            M[i,j] = stats.spearmanr(U[:,i], U[:,j]).statistic
    return (M + M.T)/2

SR_raw  = sp_matrix(U_raw)
SR_part = sp_matrix(U_part)

print('\n=== Spearman matrix: marginal vs partial (age/sex/race) ===')
print('rho_S_marginal (lower) and rho_S_partial (upper)')
M = np.where(np.triu(np.ones_like(SR_raw), k=1).astype(bool), SR_part, SR_raw)
np.fill_diagonal(M, 1.0)
print(pd.DataFrame(np.round(M, 3),
                   index=[LABEL[r] for r in RISKS],
                   columns=[LABEL[r] for r in RISKS]).to_string())

print('\nPair-by-pair shift (partial - marginal):')
print(f"{'pair':<14} {'marg':>7} {'part':>7} {'shift':>7}")
for i in range(len(RISKS)):
    for j in range(i+1, len(RISKS)):
        a, b = LABEL[RISKS[i]], LABEL[RISKS[j]]
        m = SR_raw[i,j]; p = SR_part[i,j]
        print(f'{a+"-"+b:<14} {m:>7.3f} {p:>7.3f} {p-m:>+7.3f}')

# ============================================================
# 2. Recompute tail dependence chi(q) and t-copula fits
# ============================================================
def chi_hat(u, v, q, upper=True):
    if upper:
        d = np.sum(v > q); return np.sum((u>q)&(v>q))/d if d else np.nan
    else:
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
        if abs(rho) >= 0.999: return 1e10
        nu = np.exp(log_nu)
        if nu < 2.01 or nu > 200: return 1e10
        tu = stats.t.ppf(u, df=nu); tv = stats.t.ppf(v, df=nu)
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

print('\n=== Pair-level fits: MARGINAL vs PARTIAL ===')
print(f"{'pair':<14} | {'rho_G':>7} {'nu_t':>6} {'chi_emp_95':>11} {'dAIC':>6} | "
      f"{'rho_G':>7} {'nu_t':>6} {'chi_emp_95':>11} {'dAIC':>6}")
print(f"{'':14}   {'(marginal)':-^32}    {'(partial)':-^32}")
fits = []
for a, b in PAIRS:
    i = RISKS.index(a); j = RISKS.index(b)
    res = {'pair': f'{LABEL[a]}-{LABEL[b]}'}
    for kind, U in [('marg', U_raw), ('part', U_part)]:
        u = U[:,i]; v = U[:,j]
        rg, llg = fit_gauss(u, v)
        rt, nu, llt = fit_t(u, v)
        c95 = chi_hat(u, v, 0.95, upper=stats.spearmanr(u,v).statistic >= 0)
        if stats.spearmanr(u,v).statistic < 0:
            c95 = np.sum((u>0.95)&(v<0.05))/np.sum(v<0.05)  # approx
        daic = 2*1 - 2*llg - (2*2 - 2*llt)
        res[f'rho_G_{kind}'] = rg
        res[f'nu_t_{kind}']  = nu
        res[f'chi_emp_95_{kind}'] = c95
        res[f'dAIC_{kind}']  = daic
    fits.append(res)
    print(f"{res['pair']:<14} | {res['rho_G_marg']:>7.3f} {res['nu_t_marg']:>6.1f} "
          f"{res['chi_emp_95_marg']:>11.3f} {res['dAIC_marg']:>+6.2f} | "
          f"{res['rho_G_part']:>7.3f} {res['nu_t_part']:>6.1f} "
          f"{res['chi_emp_95_part']:>11.3f} {res['dAIC_part']:>+6.2f}")

fits_df = pd.DataFrame(fits)
fits_df.to_parquet('outputs/partial_pair_fits.parquet')

# ============================================================
# 3. Multivariate joint-extreme on partial pseudo-obs
# ============================================================
def nearest_psd(A, eps=1e-8):
    w, V = linalg.eigh(A); w = np.clip(w, eps, None)
    A2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(A2)); return A2/np.outer(d,d)

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

print('\n=== Multivariate joint top-decile co-occurrence ===')
print(f"{'cluster':<35} {'empir(marg)':>11} {'empir(part)':>11} "
      f"{'G(marg)':>9} {'G(part)':>9} {'t8(part)':>9}")
N = 50_000
RG_marg = nearest_psd(2*np.sin(np.pi*SR_raw/6))
RG_part = nearest_psd(2*np.sin(np.pi*SR_part/6))
UG_marg = sim_gauss(RG_marg, N)
UG_part = sim_gauss(RG_part, N)
Ut_part = sim_t(RG_part, 8, N)

CLUSTERS = {
    'metabolic syndrome (BMI,SBP,FPG)':    [0, 2, 3],
    'metabolic + LDL inverse':             [0, 2, 3, 1],
    'CV triad + kidney':                   [0, 2, 3, 5],
}
joint_rows = []
for name, idx in CLUSTERS.items():
    e_m = joint(U_raw, idx); e_p = joint(U_part, idx)
    g_m = joint(UG_marg, idx); g_p = joint(UG_part, idx)
    t_p = joint(Ut_part, idx)
    joint_rows.append(dict(cluster=name, empir_marg=e_m, empir_part=e_p,
                           gauss_marg=g_m, gauss_part=g_p, t8_part=t_p))
    print(f'{name:<35} {e_m:>11.4f} {e_p:>11.4f} '
          f'{g_m:>9.4f} {g_p:>9.4f} {t_p:>9.4f}')
pd.DataFrame(joint_rows).to_parquet('outputs/partial_joint_extreme.parquet')

# ============================================================
# 4. Stratified diagnostic: does the same conclusion hold per stratum?
# ============================================================
print('\n=== Sex-stratified Spearman shift, headline pair (BMI-FPG) ===')
for sex_label, mask in [('female', adults['FEMALE']==1.0), ('male', adults['FEMALE']==0.0)]:
    sub = adults[mask]
    if len(sub) < 100: continue
    rho_m = stats.spearmanr(sub['BMXBMI'], sub['LBXGLU']).statistic
    print(f'  {sex_label} (n={len(sub)}): BMI-FPG Spearman = {rho_m:.3f}')

print('\n=== Age-band Spearman shift, headline pair (BMI-FPG) ===')
for lo, hi in [(40,55),(55,70),(70,85)]:
    sub = adults[adults['AGE'].between(lo,hi-1)]
    rho_m = stats.spearmanr(sub['BMXBMI'], sub['LBXGLU']).statistic
    print(f'  age {lo}-{hi-1} (n={len(sub)}): BMI-FPG Spearman = {rho_m:.3f}')
