"""Multivariate joint-extreme test: how often do 3 or 4 metabolic risks
co-occur in the upper decile? Compare empirical vs Gaussian-copula
simulation vs t-copula simulation, all with the same Spearman correlation
matrix as input.

This is the practical question for the simulation: when initializing
correlated propensities, does the joint behavior at high percentiles
match the data?
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats, linalg
import warnings; warnings.filterwarnings('ignore')

rng = np.random.default_rng(20260514)

full = pd.read_parquet('outputs/nhanes_2017_2018_merged.parquet')
RISKS = ['BMXBMI','LBDLDL','SBP_MEAN','LBXGLU','smoking_signed','eGFR_signed']
LABEL = {'BMXBMI':'BMI', 'LBDLDL':'LDL', 'SBP_MEAN':'SBP', 'LBXGLU':'FPG',
         'smoking_signed':'smk','eGFR_signed':'kid'}
adults = full[(full['AGE']>=40) & full[RISKS].notna().all(axis=1) & full['WTMEC2YR'].gt(0)].copy()
n = len(adults)
print(f'n = {n}')

# Empirical pseudo-obs in [0,1]
U = np.column_stack([stats.rankdata(adults[r].values)/(n+1) for r in RISKS])

# Lower + upper tail chi for every pair
print('\n=== Asymmetric tail dependence diagnostic ===')
print('chi_L = P(U<q | V<q) at q=0.10  // chi_U at q=0.90  // ratio')
print(f"{'pair':<14} {'chi_L(.10)':>10} {'chi_U(.90)':>10} {'asym':>8}")
asym_rows = []
for i in range(len(RISKS)):
    for j in range(i+1, len(RISKS)):
        u, v = U[:, i], U[:, j]
        cL = np.mean((u < 0.10) & (v < 0.10)) / 0.10
        cU = np.mean((u > 0.90) & (v > 0.90)) / 0.10
        # Make sign-consistent: for negatively-correlated pairs we flip v
        sp = stats.spearmanr(u, v).statistic
        if sp < 0:
            # interpret 'concordant tail' as one tail of u with opposite tail of v
            cU2 = np.mean((u > 0.90) & (v < 0.10)) / 0.10
            cL2 = np.mean((u < 0.10) & (v > 0.90)) / 0.10
            cL, cU = cU2, cL2  # so "tail of stronger-association direction"
        pair = f'{LABEL[RISKS[i]]}-{LABEL[RISKS[j]]}'
        asym_rows.append([pair, cL, cU, cU-cL])
        print(f"{pair:<14} {cL:>10.3f} {cU:>10.3f} {cU-cL:>+8.3f}")
pd.DataFrame(asym_rows, columns=['pair','chi_L_10','chi_U_90','asym']).to_parquet('outputs/tail_asymmetry.parquet')

# Spearman correlation matrix
SR = np.zeros((len(RISKS), len(RISKS)))
for i in range(len(RISKS)):
    for j in range(len(RISKS)):
        SR[i,j] = stats.spearmanr(U[:,i], U[:,j]).statistic
SR = (SR + SR.T)/2

# Gaussian-copula Pearson rho via Spearman: rho_P = 2 sin(pi rho_S / 6)
RG = 2*np.sin(np.pi*SR/6)
# Force PSD
def nearest_psd(A, eps=1e-8):
    w, V = linalg.eigh(A); w = np.clip(w, eps, None)
    A2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(A2)); D = np.outer(d,d)
    return A2/D
RG_psd = nearest_psd(RG)

# Sample N from Gaussian copula
N_SIM = 50_000
def sample_gaussian_copula(R, N):
    L = linalg.cholesky(R, lower=True)
    Z = rng.standard_normal((N, R.shape[0])) @ L.T
    return stats.norm.cdf(Z)

def sample_t_copula(R, nu, N):
    L = linalg.cholesky(R, lower=True)
    Z = rng.standard_normal((N, R.shape[0])) @ L.T
    W = rng.chisquare(nu, size=N) / nu
    T = Z / np.sqrt(W)[:, None]
    return stats.t.cdf(T, df=nu)

UG = sample_gaussian_copula(RG_psd, N_SIM)
UT4  = sample_t_copula(RG_psd, 4,  N_SIM)
UT8  = sample_t_copula(RG_psd, 8,  N_SIM)
UT15 = sample_t_copula(RG_psd, 15, N_SIM)

# Now: triple-extreme probabilities. The metabolic-syndrome cluster
# (BMI, SBP, FPG) is the canonical comorbidity profile.
print('\n=== Joint extreme co-occurrence: P(all k risks in top decile) ===')
print('Empirical vs Gaussian copula vs t-copula (same Spearman in)')

CLUSTERS = {
    'metabolic syndrome (BMI,SBP,FPG)':   [0, 2, 3],
    'metabolic + LDL inverse':            [0, 2, 3, 1],   # LDL flipped via low LDL
    'CV triad (BMI,SBP,FPG) + kid':       [0, 2, 3, 5],
}
def joint_extreme(U_mat, idx, q=0.90):
    n = len(U_mat); flags = np.ones(n, bool)
    for k, i in enumerate(idx):
        if i == 1:  # LDL — negatively correlated with metabolic, so "extreme" = low
            flags &= (U_mat[:, i] < (1-q))
        else:
            flags &= (U_mat[:, i] > q)
    return flags.mean()

# Independence baseline = q^k or (q^(k-1))(1-q) etc.
print(f'{"cluster":<35} {"k":>3} {"indep":>8} {"empir":>8} {"gauss":>8} '
      f'{"t_15":>8} {"t_8":>8} {"t_4":>8}')
extreme_rows = []
for name, idx in CLUSTERS.items():
    k = len(idx)
    p_indep = 0.10**k
    p_emp = joint_extreme(U, idx)
    p_g   = joint_extreme(UG, idx)
    p_t15 = joint_extreme(UT15, idx)
    p_t8  = joint_extreme(UT8, idx)
    p_t4  = joint_extreme(UT4, idx)
    row = dict(cluster=name, k=k, indep=p_indep, empirical=p_emp,
               gaussian=p_g, t_nu15=p_t15, t_nu8=p_t8, t_nu4=p_t4)
    extreme_rows.append(row)
    print(f'{name:<35} {k:>3d} {p_indep:>8.4f} {p_emp:>8.4f} {p_g:>8.4f} '
          f'{p_t15:>8.4f} {p_t8:>8.4f} {p_t4:>8.4f}')
pd.DataFrame(extreme_rows).to_parquet('outputs/joint_extreme.parquet')

# Also: how does inflation factor (joint extreme / Gaussian) scale with k?
print('\n=== Inflation factor (joint extreme prob / Gaussian copula prediction) ===')
print('Same clusters; >1.0 means Gaussian underpredicts.')
for r in extreme_rows:
    if r['gaussian'] > 0:
        infl_emp = r['empirical'] / r['gaussian']
        infl_t8  = r['t_nu8']    / r['gaussian']
        print(f"  {r['cluster']:<35} empirical {infl_emp:.2f}x  t(nu=8) {infl_t8:.2f}x")

# Note CI for empirical multi-extreme
print('\n=== Sampling uncertainty on empirical joint-extreme P (Wilson) ===')
for r in extreme_rows:
    p = r['empirical']; cnt = int(round(p*n))
    se = np.sqrt(p*(1-p)/n)
    print(f"  {r['cluster']:<35} empirical p = {p:.4f}  count = {cnt}  se = {se:.4f}")
