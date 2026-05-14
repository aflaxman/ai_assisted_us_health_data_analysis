"""Empirical tail dependence + Gaussian vs t-copula comparison for the
strongest pairs in NHANES 2017-2018.

Key statistic: chi(q) = P(U > q | V > q) where U=F_X(X), V=F_Y(Y).
- Gaussian copula:  chi(q) -> 0  as q -> 1  (asymptotically tail-independent)
- t-copula:         chi(q) -> 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))  > 0
- Empirical:        chi_hat(q) = sum(U>q & V>q) / sum(V>q)
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats, optimize
import warnings; warnings.filterwarnings('ignore')

trial = pd.read_parquet('outputs/trial_band.parquet')

# To get power for tail analysis: pool ages 40+. We'll also report
# trial-band-only as a sensitivity.
full = pd.read_parquet('outputs/nhanes_2017_2018_merged.parquet')
RISKS = ['BMXBMI','LBDLDL','SBP_MEAN','LBXGLU','smoking_signed','eGFR_signed']
adults = full[(full['AGE']>=40) & full[RISKS].notna().all(axis=1) & full['WTMEC2YR'].gt(0)].copy()
print(f'adults 40+ with all 6 risks: n = {len(adults):,}')
print(f'trial band 65-80:            n = {len(trial):,}')

LABEL = {'BMXBMI':'BMI', 'LBDLDL':'LDL_C', 'SBP_MEAN':'SBP', 'LBXGLU':'FPG',
         'smoking_signed':'smoking', 'eGFR_signed':'kidney'}

def empirical_uv(x, y):
    """Pseudo-observations on (0,1): rank/(n+1)."""
    n = len(x)
    u = stats.rankdata(x) / (n + 1)
    v = stats.rankdata(y) / (n + 1)
    return u, v

def chi_hat(u, v, q):
    """P(U > q | V > q) -- upper tail dependence at level q."""
    den = np.sum(v > q)
    if den == 0: return np.nan
    return np.sum((u > q) & (v > q)) / den

def chi_gaussian(rho, q):
    """Closed form for Gaussian copula tail dep coefficient at level q."""
    if abs(rho) >= 1: return np.nan
    z = stats.norm.ppf(q)
    cdf = stats.multivariate_normal.cdf([z, z], mean=[0,0], cov=[[1,rho],[rho,1]])
    return (1 - 2*q + cdf) / (1 - q)

def fit_t_copula_pair(u, v):
    """ML for bivariate t-copula in (rho, nu). Returns (rho_hat, nu_hat, loglik).
    Pseudo-obs are mapped through t_nu^{-1}, then bivariate t density."""
    # negative log-likelihood
    def nll(params):
        rho, log_nu = params
        if abs(rho) >= 0.999: return 1e10
        nu = np.exp(log_nu)
        if nu < 2.01 or nu > 200: return 1e10
        # pseudo-obs -> t_nu quantiles
        tu = stats.t.ppf(u, df=nu)
        tv = stats.t.ppf(v, df=nu)
        # bivariate t log-density at (tu, tv) minus marginal t log-densities
        # bivariate t log-density:
        d = 2
        det = 1 - rho**2
        if det <= 0: return 1e10
        inv = np.array([[1, -rho],[-rho, 1]]) / det
        q_form = inv[0,0]*tu*tu + 2*inv[0,1]*tu*tv + inv[1,1]*tv*tv
        log_biv = (np.log(2) - np.log(2*np.pi) - 0.5*np.log(det)
                   + np.log(stats.gamma(0.5*(nu+d)).pdf if False else 1)  # placeholder
        )
        # use scipy multivariate_t logpdf
        from scipy.stats import multivariate_t
        log_biv = multivariate_t.logpdf(np.column_stack([tu, tv]),
                                        loc=[0,0], shape=[[1,rho],[rho,1]], df=nu)
        log_marg_u = stats.t.logpdf(tu, df=nu)
        log_marg_v = stats.t.logpdf(tv, df=nu)
        ll = np.sum(log_biv - log_marg_u - log_marg_v)
        return -ll
    # initial guess: Pearson on the t-quantiles for some moderate nu
    rho0 = stats.spearmanr(u, v).statistic
    rho0 = 2 * np.sin(np.pi * rho0 / 6)  # Spearman -> Gaussian-rho approximation
    res = optimize.minimize(nll, [rho0, np.log(8)],
                            bounds=[(-0.999, 0.999), (np.log(2.5), np.log(150))],
                            method='L-BFGS-B')
    return res.x[0], float(np.exp(res.x[1])), -res.fun

def gaussian_copula_loglik(u, v, rho):
    """LL of the Gaussian copula at MLE rho."""
    zu = stats.norm.ppf(u)
    zv = stats.norm.ppf(v)
    det = 1 - rho**2
    q = (zu**2 - 2*rho*zu*zv + zv**2) / det
    log_c = -0.5 * np.log(det) - 0.5 * (q - zu**2 - zv**2)
    return np.sum(log_c)

def fit_gaussian_pair(u, v):
    """MLE rho for Gaussian copula on pseudo-obs."""
    res = optimize.minimize_scalar(
        lambda r: -gaussian_copula_loglik(u, v, r) if abs(r) < 0.999 else 1e10,
        bounds=(-0.999, 0.999), method='bounded')
    return res.x, -res.fun

def chi_t_copula(rho, nu, q):
    """Closed form for t-copula upper tail dep at level q."""
    if abs(rho) >= 1: return np.nan
    t_q = stats.t.ppf(q, df=nu)
    arg = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
    # P(U > q, V > q) = 1 - 2q + C(q,q); for t-copula:
    cdf = stats.multivariate_t.cdf([t_q, t_q], loc=[0,0], shape=[[1,rho],[rho,1]], df=nu)
    return (1 - 2*q + cdf) / (1 - q)

PAIRS = [('BMXBMI','LBXGLU'),     # BMI vs FPG -- metabolic syndrome cluster, strongest pair
         ('SBP_MEAN','LBXGLU'),   # SBP vs FPG
         ('BMXBMI','SBP_MEAN'),   # BMI vs SBP
         ('LBDLDL','LBXGLU'),     # LDL vs FPG -- treatment effects
         ('BMXBMI','LBDLDL'),     # BMI vs LDL
         ('BMXBMI','eGFR_signed'),# BMI vs kidney
         ]

print('\n=== Pair-level analysis (adults 40+, n={}) ==='.format(len(adults)))
print()
print(f"{'pair':<20} {'spearman':>10} {'rho_G':>8} {'rho_t':>8} {'nu_t':>7} "
      f"{'chi_emp(.90)':>12} {'chi_emp(.95)':>12} {'chi_G(.95)':>10} {'chi_t(.95)':>10} "
      f"{'dAIC(G-t)':>10}")
rows = []
for a, b in PAIRS:
    sub = adults[[a, b]].dropna()
    x = sub[a].values; y = sub[b].values
    u, v = empirical_uv(x, y)
    sp = stats.spearmanr(x, y).statistic
    rho_g, ll_g = fit_gaussian_pair(u, v)
    rho_t, nu_t, ll_t = fit_t_copula_pair(u, v)
    chi_90 = chi_hat(u, v, 0.90)
    chi_95 = chi_hat(u, v, 0.95)
    cg95   = chi_gaussian(rho_g, 0.95)
    ct95   = chi_t_copula(rho_t, nu_t, 0.95)
    aic_g  = 2*1 - 2*ll_g
    aic_t  = 2*2 - 2*ll_t
    daic   = aic_g - aic_t   # positive = t-copula is better
    rows.append(dict(pair=f'{LABEL[a]}-{LABEL[b]}',
                     spearman=sp, rho_G=rho_g, rho_t=rho_t, nu_t=nu_t,
                     chi_emp_90=chi_90, chi_emp_95=chi_95,
                     chi_G_95=cg95, chi_t_95=ct95, dAIC=daic, n=len(sub)))
    print(f"{LABEL[a]+'-'+LABEL[b]:<20} {sp:>10.3f} {rho_g:>8.3f} {rho_t:>8.3f} {nu_t:>7.1f} "
          f"{chi_90:>12.3f} {chi_95:>12.3f} {cg95:>10.3f} {ct95:>10.3f} {daic:>+10.2f}")

pd.DataFrame(rows).to_parquet('outputs/copula_pair_fits.parquet')
print('\nwrote outputs/copula_pair_fits.parquet')
