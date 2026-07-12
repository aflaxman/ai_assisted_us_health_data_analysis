"""Shared logic for the NHANES fibrosis exposure-definition comparison.

Cohort: NHANES 2017-March 2020 pre-pandemic, adults 18+ with a complete
FibroScan exam (LUX), MEC-weighted. Provides the labeled CAP-cutoff ladder,
weighted prevalence/correlation helpers, and table builders used by the
notebook and the figure script.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nhanes" / "2017_2020_prepandemic"

# LSM (kPa) -> fibrosis stage, per the thresholds under study
LSM_BANDS = [(6, "F0 <6"), (8, "F1 6-<8"), (10, "F2 8-<10"), (15, "F3 10-<15")]  # else F4 >=15

# CAP steatosis cutoffs (dB/m), labeled by source and steatosis level.
# 248/268/280 : Karlas 2017 individual-patient-data meta-analysis (mixed etiology)
# 302         : de Ledinghen 2014 / Eddowes 2019 (NAFLD-leaning, >=S1)
# 263/288/330 : clinical reference set tabulated across reviews (S1/S2/S3);
#               288 is the baseline used in the first pass of this project.
CUTOFFS = [
    (0, "no gate", "CAP≥0"),
    (248, "Karlas (mixed)", "≥S1 any"),
    (263, "clinical set", "S1 mild"),
    (268, "Karlas (mixed)", "≥S2 mod"),
    (280, "Karlas (mixed)", "≥S3 sev"),
    (288, "clinical / baseline", "S2 mod"),
    (302, "de Ledinghen/Eddowes (NAFLD)", "≥S1 any"),
    (330, "clinical set", "S3 sev"),
]

RISKS = {
    "BMI": "BMXBMI", "Waist": "BMXWAIST", "HbA1c": "LBXGH",
    "Fasting glucose": "LBXGLU", "Triglycerides": "LBXTR", "HDL-C": "LBDHDD",
    "ALT": "LBXSATSI", "AST": "LBXSASSI", "GGT": "LBXSGTSI",
    "Systolic BP": "sysbp", "Insulin": "LBXIN",
}


def collabel(cap, src, lvl):
    if cap == 0:
        return "CAP≥0 (no gate)"
    return f"{cap} {src} {lvl}"


_SRC_SHORT = {
    "Karlas (mixed)": "Karlas", "clinical set": "clin",
    "clinical / baseline": "baseline", "de Ledinghen/Eddowes (NAFLD)": "NAFLD",
}


def shortlabel(cap, src, lvl):
    if cap == 0:
        return "CAP≥0\n(no gate)"
    return f"{cap}\n{_SRC_SHORT.get(src, src)} {lvl}"


def _sas(name, cols):
    d = pd.read_sas(RAW / f"{name}.xpt")
    return d[[c for c in cols if c in d.columns]].copy()


def build_cohort():
    demo = _sas("P_DEMO", ["SEQN", "RIAGENDR", "RIDAGEYR", "WTMECPRP", "SDMVPSU", "SDMVSTRA"])
    frames = [
        _sas("P_LUX", ["SEQN", "LUAXSTAT", "LUXSMED", "LUXCAPM"]),
        _sas("P_BMX", ["SEQN", "BMXBMI", "BMXWAIST"]),
        _sas("P_GHB", ["SEQN", "LBXGH"]),
        _sas("P_GLU", ["SEQN", "LBXGLU"]),
        _sas("P_HDL", ["SEQN", "LBDHDD"]),
        _sas("P_TRIGLY", ["SEQN", "LBXTR", "LBDLDL"]),
        _sas("P_BIOPRO", ["SEQN", "LBXSATSI", "LBXSASSI", "LBXSGTSI"]),
        _sas("P_BPXO", ["SEQN", "BPXOSY1", "BPXOSY2", "BPXOSY3"]),
        _sas("P_INS", ["SEQN", "LBXIN"]),
    ]
    df = demo
    for f in frames:
        df = df.merge(f, on="SEQN", how="left")
    df["sysbp"] = df[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    keep = ((df["RIDAGEYR"] >= 18) & (df["LUAXSTAT"] == 1)
            & df["LUXSMED"].notna() & df["LUXCAPM"].notna()
            & (df["WTMECPRP"].fillna(0) > 0))
    coh = df[keep].copy()
    coh["stage"] = coh["LUXSMED"].apply(lsm_stage)
    return coh


def lsm_stage(l):
    return 0 if l < 6 else 1 if l < 8 else 2 if l < 10 else 3 if l < 15 else 4


def _neff(w):
    return w.sum() ** 2 / (w ** 2).sum()


def wfrac(mask, w):
    """Weighted proportion (%) with approximate 95% CI half-width (ignores clustering)."""
    p = w[mask].sum() / w.sum()
    hw = 1.96 * np.sqrt(p * (1 - p) / _neff(w))
    return 100 * p, 100 * hw


def _wpearson(x, y, w):
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    vx, vy = (w * (x - mx) ** 2).sum(), (w * (y - my) ** 2).sum()
    return cov / np.sqrt(vx * vy)


def wspearman(x, y, w):
    """Weighted Spearman rho with analytic Fisher-z 95% CI (rank then weighted Pearson)."""
    m = x.notna() & y.notna() & w.notna()
    x, y, w = x[m], y[m], w[m]
    r = _wpearson(x.rank(), y.rank(), w)
    z, se = np.arctanh(np.clip(r, -.999, .999)), 1 / np.sqrt(max(_neff(w) - 3, 1))
    return r, float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


# ---- table builders ----
PREV_ROWS = ["No MASLD (no steatosis)", "Steatosis (any MASLD)",
             "MASLD F0", "MASLD F1", "MASLD F2", "MASLD F3", "MASLD F4",
             "MASLD F2+ (significant)", "MASLD F3+ (advanced)"]


def prevalence_table(coh):
    """Rows = categories incl. 'No MASLD'; columns = each labeled cutoff (first = CAP>=0, no gate)."""
    w = coh["WTMECPRP"].astype(float)
    cols = {}
    for cap, src, lvl in CUTOFFS:
        steat = coh["LUXCAPM"] >= cap  # cap=0 -> all true -> no gate
        col = {"No MASLD (no steatosis)": wfrac(~steat, w)[0],
               "Steatosis (any MASLD)": wfrac(steat, w)[0]}
        for s in range(5):
            col[f"MASLD F{s}"] = wfrac(steat & (coh["stage"] == s), w)[0]
        col["MASLD F2+ (significant)"] = wfrac(steat & (coh["stage"] >= 2), w)[0]
        col["MASLD F3+ (advanced)"] = wfrac(steat & (coh["stage"] >= 3), w)[0]
        cols[collabel(cap, src, lvl)] = col
    return pd.DataFrame(cols).reindex(PREV_ROWS).round(1)


def f2plus_sensitivity(coh):
    """MASLD F2+ prevalence (%) with 95% CI half-width for each cutoff (for the line plot)."""
    w = coh["WTMECPRP"].astype(float)
    out = []
    for cap, src, lvl in CUTOFFS:
        steat = coh["LUXCAPM"] >= cap
        p, hw = wfrac(steat & (coh["stage"] >= 2), w)
        out.append({"cap": cap, "src": src, "lvl": lvl, "f2plus": p, "hw": hw,
                    "label": shortlabel(cap, src, lvl)})
    return pd.DataFrame(out)


def correlation_table(coh):
    """Rows = risks; cols = LSM cont, CAP cont, F-stage LSM-alone, then MASLD F-stage per cutoff."""
    w = coh["WTMECPRP"].astype(float)
    rows = {}
    for lab, col in RISKS.items():
        r = {"LSM (continuous)": wspearman(coh["LUXSMED"], coh[col], w)[0],
             "CAP (continuous)": wspearman(coh["LUXCAPM"], coh[col], w)[0]}
        for cap, src, lvl in CUTOFFS:
            gated = np.where(coh["LUXCAPM"] >= cap, coh["stage"], 0).astype(float)
            r[collabel(cap, src, lvl)] = wspearman(pd.Series(gated, index=coh.index), coh[col], w)[0]
        rows[lab] = r
    return pd.DataFrame(rows).T.round(3)
