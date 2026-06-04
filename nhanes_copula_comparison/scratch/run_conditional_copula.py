"""Pre-run the conditional-copula analysis to verify everything works
before baking it into a notebook."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

df = pd.read_parquet('outputs/pooled_bmi_fpg_2007_2018.parquet')
adults = df[df['RIDAGEYR'].between(20, 84)].copy()
print(f'adults 20-84: n = {len(adults):,}')

# Weighted Spearman in sliding age windows
def w_rank(x, w):
    o = np.argsort(x, kind='stable')
    cw = np.cumsum(w[o])
    r = np.empty_like(x, dtype=float)
    r[o] = cw - w[o]/2
    return r

def w_spearman(x, y, w):
    rx = w_rank(x, w); ry = w_rank(y, w)
    mx = np.average(rx, weights=w); my = np.average(ry, weights=w)
    cov = np.average((rx-mx)*(ry-my), weights=w)
    sx = np.sqrt(np.average((rx-mx)**2, weights=w))
    sy = np.sqrt(np.average((ry-my)**2, weights=w))
    return cov / (sx * sy)

CENTERS = np.arange(25, 81, 2)
HW = 5  # half-window in years
rho_age = []
n_age = []
for c in CENTERS:
    sub = adults[adults['RIDAGEYR'].between(c-HW, c+HW)]
    if len(sub) < 100:
        rho_age.append(np.nan); n_age.append(len(sub)); continue
    r = w_spearman(sub['BMXBMI'].values, sub['LBXGLU'].values, sub['weight'].values)
    rho_age.append(r); n_age.append(len(sub))
ag = pd.DataFrame({'age': CENTERS, 'rho_S': rho_age, 'n': n_age})
print('Weighted Spearman by age (sliding 10-year window):')
print(ag.head(15).to_string(index=False))
print(' ...')
print(ag.tail(8).to_string(index=False))
