"""Does lower-tail asymmetry survive partialling out age/sex/race?"""
import numpy as np, pandas as pd
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

full = pd.read_parquet('outputs/nhanes_2017_2018_merged.parquet')
demo = pd.read_sas('scratch/raw/DEMO_J.xpt')[['SEQN','RIDRETH3']]
full = full.merge(demo, on='SEQN', how='left')

RISKS = ['BMXBMI','LBDLDL','SBP_MEAN','LBXGLU','smoking_signed','eGFR_signed']
LABEL = {'BMXBMI':'BMI','LBDLDL':'LDL','SBP_MEAN':'SBP','LBXGLU':'FPG',
         'smoking_signed':'smk','eGFR_signed':'kid'}
adults = full[(full['AGE']>=40) & full[RISKS].notna().all(axis=1) &
              full['WTMEC2YR'].gt(0) & full['RIDRETH3'].notna()].copy()
n = len(adults)

age = adults['AGE'].values; sex = adults['FEMALE'].values
race = adults['RIDRETH3'].astype(int).values
rcats = sorted(np.unique(race).tolist())
rd = np.column_stack([(race == r).astype(float) for r in rcats[1:]])
age_c = age - age.mean()
X = np.column_stack([np.ones(n), age_c, age_c**2, sex, rd])

def partial_pseudo(x, X):
    z = stats.norm.ppf(stats.rankdata(x)/(len(x)+1))
    b, *_ = np.linalg.lstsq(X, z, rcond=None)
    e = z - X @ b
    return stats.norm.cdf(e / e.std(ddof=1))

U_raw  = np.column_stack([stats.rankdata(adults[r].values)/(n+1) for r in RISKS])
U_part = np.column_stack([partial_pseudo(adults[r].values, X) for r in RISKS])

rows = []
for i in range(len(RISKS)):
    for j in range(i+1, len(RISKS)):
        if LABEL[RISKS[i]] == 'smk' or LABEL[RISKS[j]] == 'smk': continue
        pair = f'{LABEL[RISKS[i]]}-{LABEL[RISKS[j]]}'
        u_m, v_m = U_raw[:,i], U_raw[:,j]
        u_p, v_p = U_part[:,i], U_part[:,j]
        sp_m = stats.spearmanr(u_m, v_m).statistic
        sp_p = stats.spearmanr(u_p, v_p).statistic
        # Sign-align upper/lower tails
        if sp_m < 0:
            cL_m = np.mean((u_m>0.9)&(v_m<0.1))/0.10
            cU_m = np.mean((u_m<0.1)&(v_m>0.9))/0.10
        else:
            cL_m = np.mean((u_m<0.1)&(v_m<0.1))/0.10
            cU_m = np.mean((u_m>0.9)&(v_m>0.9))/0.10
        if sp_p < 0:
            cL_p = np.mean((u_p>0.9)&(v_p<0.1))/0.10
            cU_p = np.mean((u_p<0.1)&(v_p>0.9))/0.10
        else:
            cL_p = np.mean((u_p<0.1)&(v_p<0.1))/0.10
            cU_p = np.mean((u_p>0.9)&(v_p>0.9))/0.10
        rows.append([pair, cL_m, cU_m, cU_m-cL_m, cL_p, cU_p, cU_p-cL_p])

df = pd.DataFrame(rows, columns=['pair','cL_marg','cU_marg','asym_marg',
                                  'cL_part','cU_part','asym_part'])
print('Lower vs upper tail chi: marginal vs partial')
print(f"{'pair':<14}|{'cL':>6}{'cU':>6}{'asym':>7} |{'cL':>6}{'cU':>6}{'asym':>7}")
print(f"{'':<14}|{'(marginal)':-^19}|{'(partial)':-^19}")
for r in rows:
    pair, cLm, cUm, am, cLp, cUp, ap = r
    print(f'{pair:<14}|{cLm:>6.2f}{cUm:>6.2f}{am:>+7.2f} |{cLp:>6.2f}{cUp:>6.2f}{ap:>+7.2f}')
df.to_parquet('outputs/partial_tail_asymmetry.parquet')
