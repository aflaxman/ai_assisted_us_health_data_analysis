"""Compare four ways to model liver fibrosis in NHANES 2017-March 2020 pre-pandemic.

Builds a FibroScan cohort (LSM + CAP) merged with metabolic risk factors, then
computes, under two categorical definitions and two continuous exposures:
  - weighted prevalence of F0-F4
  - weighted Spearman correlation with each metabolic risk factor

Definitions
  A) LSM alone:      F0 <6, F1 6-<8, F2 8-<10, F3 10-<15, F4 >=15 kPa
  B) MASLD-gated:    same LSM stages, but stage counts only if CAP >= 288 dB/m
Continuous exposures: LSM (stiffness) and CAP (steatosis); their rank
dependence is the case for/against a Gaussian-copula joint model.

Writes results.json (consumed by make_figures.py). Run from repo root:
    python -P nhanes_fibrosis_modeling/01_build_and_analyze.py <out_dir>
"""
import json, os, sys
import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "nhanes_fibrosis_modeling/outputs"
os.makedirs(OUT, exist_ok=True)
RAW = "data/raw/nhanes/2017_2020_prepandemic"
CAP_THRESHOLD = 288  # dB/m; steatosis / MASLD gate (alternatives: 248/274/302)


def sas(name, cols):
    d = pd.read_sas(f"{RAW}/{name}.xpt")
    return d[[c for c in cols if c in d.columns]].copy()


# ---- assemble cohort ----
demo = sas("P_DEMO", ["SEQN", "RIAGENDR", "RIDAGEYR", "WTMECPRP", "SDMVPSU", "SDMVSTRA"])
frames = [
    sas("P_LUX", ["SEQN", "LUAXSTAT", "LUXSMED", "LUXCAPM"]),
    sas("P_BMX", ["SEQN", "BMXBMI", "BMXWAIST"]),
    sas("P_GHB", ["SEQN", "LBXGH"]),
    sas("P_GLU", ["SEQN", "LBXGLU"]),
    sas("P_HDL", ["SEQN", "LBDHDD"]),
    sas("P_TRIGLY", ["SEQN", "LBXTR", "LBDLDL"]),
    sas("P_BIOPRO", ["SEQN", "LBXSATSI", "LBXSASSI", "LBXSGTSI"]),
    sas("P_BPXO", ["SEQN", "BPXOSY1", "BPXOSY2", "BPXOSY3"]),
    sas("P_INS", ["SEQN", "LBXIN"]),
]
df = demo
for f in frames:
    df = df.merge(f, on="SEQN", how="left")
df["sysbp"] = df[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)

mask = ((df["RIDAGEYR"] >= 18) & (df["LUAXSTAT"] == 1)
        & df["LUXSMED"].notna() & df["LUXCAPM"].notna()
        & (df["WTMECPRP"].fillna(0) > 0))
coh = df[mask].copy()
w = coh["WTMECPRP"].astype(float)


# ---- stages ----
def lsm_stage(l):
    return 0 if l < 6 else 1 if l < 8 else 2 if l < 10 else 3 if l < 15 else 4


coh["stage_A"] = coh["LUXSMED"].apply(lsm_stage)                     # LSM alone
steat = coh["LUXCAPM"] >= CAP_THRESHOLD
coh["stage_B"] = np.where(steat, coh["stage_A"], 0)                  # MASLD-gated

n_eff = w.sum() ** 2 / (w ** 2).sum()


def wprev(stage):
    W = w.sum()
    out = {}
    for s in range(5):
        p = w[stage == s].sum() / W
        out[f"F{s}"] = [100 * p, 100 * 1.96 * np.sqrt(p * (1 - p) / n_eff)]
    return out


def wpear(x, y, ww):
    ww = ww / ww.sum()
    mx, my = (ww * x).sum(), (ww * y).sum()
    cov = (ww * (x - mx) * (y - my)).sum()
    vx, vy = (ww * (x - mx) ** 2).sum(), (ww * (y - my) ** 2).sum()
    return cov / np.sqrt(vx * vy)


def wspear(x, y):
    m = x.notna() & y.notna() & w.notna()
    xx, yy, ww = x[m], y[m], w[m]
    ne = ww.sum() ** 2 / (ww ** 2).sum()
    r = wpear(xx.rank(), yy.rank(), ww)
    z, se = np.arctanh(np.clip(r, -.999, .999)), 1 / np.sqrt(max(ne - 3, 1))
    return [r, float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)), int(m.sum())]


risks = {"BMI": "BMXBMI", "Waist": "BMXWAIST", "HbA1c": "LBXGH",
         "Fasting glucose": "LBXGLU", "Triglycerides": "LBXTR", "HDL-C": "LBDHDD",
         "ALT": "LBXSATSI", "AST": "LBXSASSI", "GGT": "LBXSGTSI",
         "Systolic BP": "sysbp", "Insulin": "LBXIN"}

res = {"n": int(coh.shape[0]),
       "steatosis_pct": float(100 * w[steat].sum() / w.sum()),
       "cap_threshold": CAP_THRESHOLD,
       "prev": {"A": wprev(coh["stage_A"]), "B": wprev(coh["stage_B"])},
       "lsm_cap": wspear(coh["LUXSMED"], coh["LUXCAPM"]),
       "corr": {}}
for lab, col in risks.items():
    res["corr"][lab] = {
        "LSM": wspear(coh["LUXSMED"], coh[col]),
        "CAP": wspear(coh["LUXCAPM"], coh[col]),
        "stageA": wspear(coh["stage_A"].astype(float), coh[col]),
        "stageB": wspear(coh["stage_B"].astype(float), coh[col]),
    }

json.dump(res, open(f"{OUT}/results.json", "w"), indent=2)
print(f"n={res['n']}  steatosis={res['steatosis_pct']:.1f}%  LSM-CAP rho={res['lsm_cap'][0]:.2f}")
print(f"wrote {OUT}/results.json")
