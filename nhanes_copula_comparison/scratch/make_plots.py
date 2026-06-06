"""Generate the plots that will be embedded in the notebooks."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats, linalg
import warnings; warnings.filterwarnings('ignore')

OUT = Path('outputs'); OUT.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 100, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

full = pd.read_parquet('outputs/nhanes_2017_2018_merged.parquet')
RISKS = ['BMXBMI','LBDLDL','SBP_MEAN','LBXGLU','smoking_signed','eGFR_signed']
LABEL = {'BMXBMI':'BMI', 'LBDLDL':'LDL-C', 'SBP_MEAN':'SBP', 'LBXGLU':'FPG',
         'smoking_signed':'smoking','eGFR_signed':'kidney (-eGFR)'}
adults = full[(full['AGE']>=40) & full[RISKS].notna().all(axis=1) & full['WTMEC2YR'].gt(0)].copy()
n = len(adults)

# Pseudo-obs
U = np.column_stack([stats.rankdata(adults[r].values)/(n+1) for r in RISKS])

# ============================================================
# Plot 1: chi-plots for the strongest continuous pairs
# ============================================================
PAIRS_PLOT = [('BMXBMI','LBXGLU','BMI vs FPG (ρ_S=+0.25)'),
              ('SBP_MEAN','LBXGLU','SBP vs FPG (ρ_S=+0.12)'),
              ('BMXBMI','SBP_MEAN','BMI vs SBP (ρ_S=+0.10)'),
              ('LBDLDL','LBXGLU','LDL-C vs FPG (ρ_S=-0.14)')]

def chi_emp(u, v, q):
    if q > 0.5:
        d = np.mean(v > q)
        return np.mean((u > q) & (v > q)) / d if d > 0 else np.nan
    else:
        d = np.mean(v < q)
        return np.mean((u < q) & (v < q)) / d if d > 0 else np.nan

def chi_gauss(rho, q):
    if abs(rho) >= 1: return np.nan
    z = stats.norm.ppf(q)
    cdf = stats.multivariate_normal.cdf([z, z], mean=[0,0], cov=[[1,rho],[rho,1]])
    if q > 0.5:
        return (1 - 2*q + cdf) / (1 - q)
    else:
        return cdf / q

def chi_t(rho, nu, q):
    if abs(rho) >= 1: return np.nan
    t_q = stats.t.ppf(q, df=nu)
    cdf = stats.multivariate_t.cdf([t_q, t_q], loc=[0,0], shape=[[1,rho],[rho,1]], df=nu)
    if q > 0.5:
        return (1 - 2*q + cdf) / (1 - q)
    else:
        return cdf / q

fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
qs = np.linspace(0.55, 0.97, 25)
qs_lo = np.linspace(0.03, 0.45, 25)

for ax, (a, b, title) in zip(axes, PAIRS_PLOT):
    i = RISKS.index(a); j = RISKS.index(b)
    u, v = U[:, i], U[:, j]
    sp = stats.spearmanr(u, v).statistic
    # If pair is negatively associated, flip v for tail diagnostic
    if sp < 0:
        v_plot = 1 - v
    else:
        v_plot = v
    rho_g = 2*np.sin(np.pi*sp/6)
    chi_emp_u = [chi_emp(u, v_plot, q) for q in qs]
    chi_emp_l = [chi_emp(u, v_plot, q) for q in qs_lo]
    chi_g_u   = [chi_gauss(abs(rho_g), q) for q in qs]
    chi_g_l   = [chi_gauss(abs(rho_g), q) for q in qs_lo]
    chi_t10_u = [chi_t(abs(rho_g), 10, q) for q in qs]
    chi_t10_l = [chi_t(abs(rho_g), 10, q) for q in qs_lo]

    ax.plot(qs, chi_emp_u, 'o-', color='black', markersize=4, label='empirical')
    ax.plot(qs_lo, chi_emp_l, 'o-', color='black', markersize=4)
    ax.plot(qs, chi_g_u,   color='C0', label='Gaussian (same ρ)')
    ax.plot(qs_lo, chi_g_l, color='C0')
    ax.plot(qs, chi_t10_u, color='C3', linestyle='--', label='t-copula (ν=10)')
    ax.plot(qs_lo, chi_t10_l, color='C3', linestyle='--')
    ax.axvline(0.5, color='gray', alpha=0.4, linestyle=':')
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('quantile q')
    if ax is axes[0]: ax.set_ylabel('χ(q) = P(U≷q | V≷q)')
axes[0].legend(loc='upper center', fontsize=8, frameon=False)
fig.suptitle('Empirical vs Gaussian vs t-copula tail dependence — NHANES 2017–18, adults 40+ (n={})'.format(n), y=1.02, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / 'chi_plots.png', dpi=120, bbox_inches='tight')
print('wrote chi_plots.png')

# ============================================================
# Plot 2: scatter of pseudo-obs with overlays
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, (a, b, title) in zip(axes, PAIRS_PLOT[:3]):
    i = RISKS.index(a); j = RISKS.index(b)
    ax.scatter(U[:, i], U[:, j], s=4, alpha=0.25, color='black')
    ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.9, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.1, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.1, color='gray', linestyle=':', alpha=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f'rank({LABEL[a]})/n')
    ax.set_ylabel(f'rank({LABEL[b]})/n')
    ax.set_aspect('equal'); ax.set_xlim(0,1); ax.set_ylim(0,1)
fig.suptitle('Copula scatter (pseudo-observations) — dotted lines mark the 10th and 90th percentile boxes', y=1.02, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / 'pseudo_obs_scatter.png', dpi=120, bbox_inches='tight')
print('wrote pseudo_obs_scatter.png')

# ============================================================
# Plot 3: joint-extreme inflation bar chart
# ============================================================
fits = pd.read_parquet('outputs/copula_pair_fits.parquet')
extremes = pd.read_parquet('outputs/joint_extreme.parquet')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: AIC delta per pair
fits_sorted = fits.sort_values('dAIC', ascending=True)
colors = ['C3' if v > 4 else ('C1' if v > 0 else 'C0') for v in fits_sorted['dAIC']]
ax1.barh(fits_sorted['pair'], fits_sorted['dAIC'], color=colors)
ax1.axvline(0, color='black', linewidth=0.6)
ax1.axvline(4, color='gray', linestyle=':', label='ΔAIC = 4 (meaningful)')
ax1.set_xlabel('AIC(Gaussian) − AIC(t-copula) ;  >0 favors t')
ax1.set_title('Per-pair AIC: Gaussian vs t-copula\n(adults 40+, n=1544)', fontsize=10)
ax1.legend(loc='lower right', fontsize=8, frameon=False)

# Right: joint extreme rates per cluster
clusters = extremes['cluster'].values
positions = np.arange(len(clusters))
width = 0.18
ax2.bar(positions - 2*width, extremes['indep'],    width, label='independence', color='lightgray')
ax2.bar(positions - 1*width, extremes['empirical'],width, label='empirical', color='black')
ax2.bar(positions + 0*width, extremes['gaussian'], width, label='Gaussian copula', color='C0')
ax2.bar(positions + 1*width, extremes['t_nu15'],   width, label='t-copula ν=15', color='C2')
ax2.bar(positions + 2*width, extremes['t_nu8'],    width, label='t-copula ν=8',  color='C3')
ax2.set_xticks(positions)
ax2.set_xticklabels([c.replace(' ', '\n', 1) for c in clusters], fontsize=8)
ax2.set_ylabel('P(joint top-decile co-occurrence)')
ax2.set_title('Multivariate joint extreme probabilities\n(same Spearman ρ matrix in, different copulas)', fontsize=10)
ax2.legend(fontsize=8, frameon=False, loc='upper right')

fig.tight_layout()
fig.savefig(OUT / 'aic_and_joint_extreme.png', dpi=120, bbox_inches='tight')
print('wrote aic_and_joint_extreme.png')

# ============================================================
# Plot 4: asymmetric tail dependence summary
# ============================================================
asym = pd.read_parquet('outputs/tail_asymmetry.parquet')
# Filter to continuous-continuous pairs (drop those with smk)
cont_only = asym[~asym['pair'].str.contains('smk')].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.5, 4))
y = np.arange(len(cont_only))
ax.barh(y - 0.18, cont_only['chi_L_10'], 0.36, label='lower tail χ(0.10)', color='C0')
ax.barh(y + 0.18, cont_only['chi_U_90'], 0.36, label='upper tail χ(0.90)', color='C3')
ax.set_yticks(y); ax.set_yticklabels(cont_only['pair'])
ax.axvline(0.10, color='gray', linestyle=':', alpha=0.7, label='independence (q=0.10)')
ax.set_xlabel('χ(q) — tail dependence coefficient')
ax.set_title('Asymmetric tail dependence in NHANES metabolic risks\n(continuous-continuous pairs; sign-aligned)', fontsize=10)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / 'tail_asymmetry.png', dpi=120, bbox_inches='tight')
print('wrote tail_asymmetry.png')
