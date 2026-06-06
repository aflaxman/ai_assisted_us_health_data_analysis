"""Heatmap of correlation shifts (partial - marginal) for the overview."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
full = pd.read_parquet('outputs/nhanes_2017_2018_merged.parquet')
demo = pd.read_sas('scratch/raw/DEMO_J.xpt')[['SEQN','RIDRETH3']]
full = full.merge(demo, on='SEQN', how='left')
RISKS = ['BMXBMI','LBDLDL','SBP_MEAN','LBXGLU','smoking_signed','eGFR_signed']
LABEL = {'BMXBMI':'BMI','LBDLDL':'LDL-C','SBP_MEAN':'SBP','LBXGLU':'FPG',
         'smoking_signed':'smk','eGFR_signed':'kid'}
adults = full[(full['AGE']>=40) & full[RISKS].notna().all(axis=1) &
              full['WTMEC2YR'].gt(0) & full['RIDRETH3'].notna()].copy()
n = len(adults)
age=adults['AGE'].values; sex=adults['FEMALE'].values
race=adults['RIDRETH3'].astype(int).values
rcats = sorted(np.unique(race).tolist())
rd = np.column_stack([(race==r).astype(float) for r in rcats[1:]])
age_c = age - age.mean()
X = np.column_stack([np.ones(n), age_c, age_c**2, sex, rd])
def part_pseudo(x):
    z = stats.norm.ppf(stats.rankdata(x)/(len(x)+1))
    b,*_=np.linalg.lstsq(X,z,rcond=None)
    e = z - X@b
    return stats.norm.cdf(e/e.std(ddof=1))
U_r = np.column_stack([stats.rankdata(adults[r].values)/(n+1) for r in RISKS])
U_p = np.column_stack([part_pseudo(adults[r].values) for r in RISKS])

def sp(U):
    K=U.shape[1]; M=np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            M[i,j]=stats.spearmanr(U[:,i],U[:,j]).statistic
    return (M+M.T)/2
SR_r, SR_p = sp(U_r), sp(U_p)
shift = SR_p - SR_r
np.fill_diagonal(shift, 0)
labs=[LABEL[r] for r in RISKS]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, M, title in [
    (axes[0], SR_r, 'Marginal Spearman ρ'),
    (axes[1], SR_p, 'Partial ρ (age/sex/race out)'),
    (axes[2], shift, 'Shift: partial − marginal')]:
    im = ax.imshow(M, vmin=-0.35, vmax=0.35, cmap='RdBu_r')
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=45, ha='right')
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    ax.set_title(title, fontsize=10)
    for i in range(len(labs)):
        for j in range(len(labs)):
            v = M[i,j]
            ax.text(j, i, f'{v:+.2f}' if i!=j else '', ha='center', va='center',
                    fontsize=8, color='white' if abs(v)>0.20 else 'black')
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
plt.suptitle(f'Confounding effect on the Spearman matrix — NHANES 2017-18, adults 40+ (n={n})', y=1.04, fontsize=10)
plt.savefig('outputs/confounding_heatmap.png', dpi=120, bbox_inches='tight')
print('wrote outputs/confounding_heatmap.png')
