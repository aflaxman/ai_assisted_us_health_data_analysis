"""Compute CAP (controlled attenuation parameter) correlation pairs.

Mirror of ``03_lsm_pairs.ipynb`` but for CAP (``LUXCAPM``, hepatic
steatosis) instead of LSM (``LUXSMED``, fibrosis). CAP is measured on
the same FibroScan exam as LSM, so it lives in the same P_LUX cycle
(2017 - March 2020 pre-pandemic combined release) and shares the exact
biomarker merge.

Estimates the 7 correlations of CAP against the rest of the matrix:

    CAP <-> {BMI, LDL-C, SBP, FPG, smoking, eGFR, liver_stiffness}

CAP is a continuous "higher = more steatosis = worse" axis, so it takes
no sign flip in value space and is *not* in the low-propensity-bad set
{smoking, kidney_dysfunction}; only CAP<->smoking and CAP<->kidney flip
between value and propensity space (matching notebook 04's convention).
CAP <-> lipoprotein_a has no NHANES overlap (Lp(a) is NHANES III
1991-94, pre-FibroScan) and is carried as a literature 0.0 stand-in,
exactly as LSM <-> Lp(a) is.

Writes:
  outputs/cap_headline.parquet       -- trial-band 65-80 point + jackknife SE
  outputs/cap_age_strat.parquet      -- age-band point estimates
  outputs/cap_pairs_propensity.csv   -- CAP rows in the consuming project's
                                        risk-correlation-matrix CSV format
"""
from __future__ import annotations

import os
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW_PP = DATA / "raw" / "nhanes" / "2017_2020_prepandemic"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RAW_PP.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Download (or load cached) P_LUX cycle XPT files
# ---------------------------------------------------------------------------
FILES = [
    "P_DEMO.xpt", "P_LUX.xpt", "P_BMX.xpt", "P_BPXO.xpt",
    "P_TRIGLY.xpt", "P_GLU.xpt", "P_BIOPRO.xpt", "P_SMQ.xpt",
]
for f in FILES:
    out = RAW_PP / f
    if out.exists():
        print(f"cached: {out.name} ({out.stat().st_size:,} bytes)")
        continue
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{f}"
    print(f"downloading {f} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        out.write_bytes(r.read())
    print(f"  saved {out.stat().st_size:,} bytes")

# ---------------------------------------------------------------------------
# 2. Parse + merge by SEQN  (identical to 03_lsm_pairs, + LUXCAPM)
# ---------------------------------------------------------------------------
DEMO_COLS = ["SEQN", "RIAGENDR", "RIDAGEYR", "WTMECPRP", "SDMVPSU", "SDMVSTRA"]
LUX_COLS = ["SEQN", "LUXSMED", "LUXCAPM", "LUAXSTAT"]
BMX_COLS = ["SEQN", "BMXBMI"]
BPX_COLS = ["SEQN", "BPXOSY1", "BPXOSY2", "BPXOSY3"]
TRIG_COLS = ["SEQN", "LBDLDL"]
GLU_COLS = ["SEQN", "LBXGLU", "WTSAFPRP"]
BIO_COLS = ["SEQN", "LBXSCR"]
SMQ_COLS = ["SEQN", "SMQ020", "SMQ040"]


def load_xpt(name, cols):
    df = pd.read_sas(RAW_PP / name)
    keep = [c for c in cols if c in df.columns]
    return df[keep]


demo = load_xpt("P_DEMO.xpt", DEMO_COLS)
lux = load_xpt("P_LUX.xpt", LUX_COLS)
bmx = load_xpt("P_BMX.xpt", BMX_COLS)
bpx = load_xpt("P_BPXO.xpt", BPX_COLS)
trig = load_xpt("P_TRIGLY.xpt", TRIG_COLS)
glu = load_xpt("P_GLU.xpt", GLU_COLS)
bio = load_xpt("P_BIOPRO.xpt", BIO_COLS)
smq = load_xpt("P_SMQ.xpt", SMQ_COLS)

bpx["SBP_MEAN"] = bpx[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
bpx = bpx[["SEQN", "SBP_MEAN"]]

for f in [demo, lux, bmx, bpx, trig, glu, bio, smq]:
    f["SEQN"] = f["SEQN"].astype(int)

df = (
    demo.merge(lux, on="SEQN", how="left")
    .merge(bmx, on="SEQN", how="left")
    .merge(bpx, on="SEQN", how="left")
    .merge(trig, on="SEQN", how="left")
    .merge(glu, on="SEQN", how="left")
    .merge(bio, on="SEQN", how="left")
    .merge(smq, on="SEQN", how="left")
)
print(f"merged: {len(df):,} rows from P_DEMO")
print(f'  with CAP (LUXCAPM) measured: {df["LUXCAPM"].notna().sum():,}')

df["sex"] = df["RIAGENDR"].map({1.0: "Male", 2.0: "Female"})
df["age_years"] = df["RIDAGEYR"].astype(float)
df["exam_complete"] = df["LUAXSTAT"] == 1.0
df["CAP_DBM"] = df["LUXCAPM"]
df["LSM_KPA"] = df["LUXSMED"]


def smoking_cat(row):
    if row["SMQ020"] == 2.0:
        return 3
    if row["SMQ020"] == 1.0:
        if row["SMQ040"] in (1.0, 2.0):
            return 1
        if row["SMQ040"] == 3.0:
            return 2
    return np.nan


df["smoking_cat"] = df.apply(smoking_cat, axis=1)


def ckd_epi_2021(scr, age, female):
    if pd.isna(scr) or pd.isna(age) or pd.isna(female):
        return np.nan
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    sex_factor = 1.012 if female else 1.0
    ratio = scr / kappa
    return (
        142
        * (min(ratio, 1) ** alpha)
        * (max(ratio, 1) ** -1.200)
        * (0.9938 ** age)
        * sex_factor
    )


df["eGFR"] = df.apply(
    lambda r: ckd_epi_2021(r["LBXSCR"], r["age_years"], r["sex"] == "Female"), axis=1
)

# MEC-examined adults 20+ with a valid CAP exam.
ana = df[
    df["exam_complete"]
    & df["CAP_DBM"].notna()
    & df["WTMECPRP"].fillna(0).gt(0)
    & (df["age_years"] >= 20)
].copy()
trial = ana[(ana["age_years"] >= 65) & (ana["age_years"] <= 80)].copy()
print(f"MEC-examined adults 20+ with CAP:                 n = {len(ana):,}")
print(f"Trial-band 65-80 with CAP:                        n = {len(trial):,}")

# ---------------------------------------------------------------------------
# 3. Weighted Spearman + paired-PSU jackknife  (identical to 03/01)
# ---------------------------------------------------------------------------
def weighted_rank(x, w):
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    r = np.empty_like(x, dtype=float)
    r[o] = cw - w[o] / 2.0
    return r


def w_spearman(x, y, w):
    rx, ry = weighted_rank(x, w), weighted_rank(y, w)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    sx = np.sqrt(np.average((rx - mx) ** 2, weights=w))
    sy = np.sqrt(np.average((ry - my) ** 2, weights=w))
    if sx == 0 or sy == 0:
        return np.nan
    return float(cov / (sx * sy))


def paired_jackknife_spearman(
    df, x_col, y_col, wt="WTMECPRP", psu="SDMVPSU", stratum="SDMVSTRA"
):
    sub = df[[x_col, y_col, wt, psu, stratum]].dropna()
    sub = sub[sub[wt] > 0]
    if len(sub) < 30:
        return np.nan, np.nan, len(sub)
    x = sub[x_col].values.astype(float)
    y = sub[y_col].values.astype(float)
    w = sub[wt].values.astype(float)
    s = sub[stratum].astype(int).values
    p = sub[psu].astype(int).values
    theta = w_spearman(x, y, w)
    var_sum = 0.0
    for st in np.unique(s):
        in_str = s == st
        psus_in = np.unique(p[in_str])
        if len(psus_in) < 2:
            continue
        for psu_id in psus_in:
            wr = w.copy()
            mask_in = in_str & (p == psu_id)
            mask_other = in_str & (p != psu_id)
            wr[mask_in] = 0.0
            wr[mask_other] *= 2.0
            if wr.sum() <= 0:
                continue
            t = w_spearman(x, y, wr)
            if not np.isnan(t):
                var_sum += (t - theta) ** 2
    return theta, np.sqrt(var_sum), len(sub)


# ---------------------------------------------------------------------------
# 4. Sign convention + headline 7 pairs (trial band 65-80)
# CAP (CAP_DBM): higher = more steatosis = worse, no flip (like LSM).
# ---------------------------------------------------------------------------
trial["smoking_signed"] = 4 - trial["smoking_cat"]
trial["eGFR_signed"] = -trial["eGFR"]
ana["smoking_signed"] = 4 - ana["smoking_cat"]
ana["eGFR_signed"] = -ana["eGFR"]

OTHER_RISKS = {
    "BMI": "BMXBMI",
    "LDL_C": "LBDLDL",
    "SBP": "SBP_MEAN",
    "FPG": "LBXGLU",
    "smoking": "smoking_signed",
    "kidney_dysfunction": "eGFR_signed",
    "liver_stiffness": "LSM_KPA",
}
rows = []
for label, col in OTHER_RISKS.items():
    rho, se, n = paired_jackknife_spearman(trial, "CAP_DBM", col)
    rows.append(
        {
            "pair": f"cap <-> {label}",
            "risk_a": "cap",
            "risk_b": label,
            "n": n,
            "rho": rho,
            "se": se,
            "lo": rho - 1.96 * se if not pd.isna(se) else np.nan,
            "hi": rho + 1.96 * se if not pd.isna(se) else np.nan,
        }
    )
cap_headline = pd.DataFrame(rows).round(3)
print("\nTrial-band 65-80, weighted Spearman, 95% CI from paired-PSU jackknife:")
print(cap_headline[["pair", "n", "rho", "se", "lo", "hi"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Age stratification — point estimates only
# ---------------------------------------------------------------------------
AGE_BANDS = {"40-64": (40, 65), "65-80": (65, 81), "80+": (80, 200)}
rows = []
for label, col in OTHER_RISKS.items():
    row = {"pair": f"cap <-> {label}"}
    for ag, (lo, hi) in AGE_BANDS.items():
        sub = ana[ana["age_years"].between(lo, hi - 1)]
        rho, _, n = paired_jackknife_spearman(sub, "CAP_DBM", col)
        row[f"n_{ag}"] = n
        row[ag] = rho
    rows.append(row)
cap_age = pd.DataFrame(rows)
print("\nCAP pair correlations by age band:")
print(cap_age.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Propensity-space conversion (matches 04_assemble_matrix §6)
# ---------------------------------------------------------------------------
LOW_PROP_BAD = {"smoking", "kidney_dysfunction"}
PROJECT_NAME = {
    "BMI": "high_body_mass_index_in_adults",
    "LDL_C": "high_ldl_cholesterol",
    "SBP": "high_systolic_blood_pressure",
    "FPG": "high_fasting_plasma_glucose",
    "smoking": "smoking",
    "lipoprotein_a": "lipoprotein_a",
    "liver_stiffness": "liver_stiffness",
    "kidney_dysfunction": "kidney_dysfunction",
    "cap": "cap",
}

# Add the CAP <-> Lp(a) literature stand-in (no NHANES FibroScan overlap).
long = cap_headline.copy()
long["source"] = "NHANES P_LUX 2017-Mar 2020"
lit = pd.DataFrame(
    [
        {
            "pair": "cap <-> lipoprotein_a",
            "risk_a": "cap",
            "risk_b": "lipoprotein_a",
            "n": 0,
            "rho": 0.0,
            "se": np.nan,
            "lo": np.nan,
            "hi": np.nan,
            "source": "literature (no NHANES overlap)",
        }
    ]
)
long = pd.concat([long, lit], ignore_index=True)

prop_rows = []
for _, r in long.iterrows():
    a, b, rho = r["risk_a"], r["risk_b"], r["rho"]
    flip = (a in LOW_PROP_BAD) ^ (b in LOW_PROP_BAD)
    rho_prop = -rho if flip else rho
    name_a, name_b = sorted([PROJECT_NAME[a], PROJECT_NAME[b]])
    prop_rows.append(
        {
            "project_pair_key": f"{name_a}_AND_{name_b}",
            "risk_a": a,
            "risk_b": b,
            "rho_value_space": round(rho, 3),
            "rho_propensity": round(rho_prop, 3),
            "flipped": flip,
            "n": int(r["n"]),
            "source": r["source"],
        }
    )
prop = pd.DataFrame(prop_rows).sort_values("project_pair_key")
print("\nPropensity-space CAP pairs for the consuming project's matrix CSV:")
print(prop[["project_pair_key", "rho_value_space", "rho_propensity", "flipped", "n"]].to_string(index=False))

cap_headline.to_parquet(OUT / "cap_headline.parquet")
cap_age.to_parquet(OUT / "cap_age_strat.parquet")
prop.to_csv(OUT / "cap_pairs_propensity.csv", index=False)
print(f"\nwrote {OUT}/cap_headline.parquet, cap_age_strat.parquet, cap_pairs_propensity.csv")
