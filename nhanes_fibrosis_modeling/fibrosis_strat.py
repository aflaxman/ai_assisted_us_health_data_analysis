"""Sex x 10-year age-band stratification of the fibrosis analysis.

Powered for common outcomes only: steatosis/MASLD and F2+ prevalence are stable
per cell; F3+ is marginal (report with caution); F4 is too sparse to stratify.
Stratified correlations use the full-MEC-sample risks only (fasting labs are too
thin once split by cell and steatosis-gated).
"""
import numpy as np
import pandas as pd
import fibrosis_lib as fl

AGE_BANDS = [(18, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 80)]
# full-MEC-sample metabolic risks (exclude fasting-subsample glucose/trig/insulin)
STRAT_RISKS = ["Waist", "BMI", "HbA1c", "HDL-C", "ALT", "AST", "GGT", "Systolic BP"]


def add_strata(coh):
    coh = coh.copy()

    def band(a):
        for lo, hi in AGE_BANDS:
            if lo <= a <= hi:
                return f"{lo}-{hi}"
        return None

    coh["ageband"] = coh["RIDAGEYR"].apply(band)
    coh["sexlab"] = coh["RIAGENDR"].map({1.0: "Male", 2.0: "Female"})
    return coh


# requested reporting categories (LSM-based fibrosis stage, no steatosis gate)
STAGE_GROUPS = [("<F1", {0}), ("F1", {1}), ("F2–3", {2, 3}), (">F3", {4})]


def stage_distribution(coh):
    """Weighted % (+95% CI) of each LSM stage group per sex x age band; sums to 100%/cell."""
    coh = add_strata(coh)
    recs = []
    for (sx, ab), g in coh.groupby(["sexlab", "ageband"]):
        w = g["WTMECPRP"].astype(float)
        for name, ss in STAGE_GROUPS:
            mask = g["stage"].isin(ss)
            p, hw = fl.wfrac(mask, w)
            recs.append(dict(sex=sx, age=ab, category=name, pct=round(p, 1),
                             ci_lo=round(max(p - hw, 0), 1), ci_hi=round(p + hw, 1),
                             n_event=int(mask.sum()), N=len(g)))
    return pd.DataFrame(recs)


def stage_cell_counts(coh):
    """Unweighted event counts per stratum for the requested LSM stage groups (feasibility)."""
    coh = add_strata(coh)
    recs = []
    for (sx, ab), g in coh.groupby(["sexlab", "ageband"]):
        row = dict(sex=sx, age=ab, N=len(g))
        for name, ss in STAGE_GROUPS:
            row[name] = int(g["stage"].isin(ss).sum())
        recs.append(row)
    return pd.DataFrame(recs).sort_values(["sex", "age"]).reset_index(drop=True)


def cell_counts(coh, cut=288):
    """Unweighted N and event counts per sex x age band (feasibility table)."""
    coh = add_strata(coh)
    recs = []
    for (sx, ab), g in coh.groupby(["sexlab", "ageband"]):
        m = g["LUXCAPM"] >= cut
        recs.append(dict(
            sex=sx, age=ab, N=len(g),
            steatosis=int(m.sum()),
            MASLD_F2plus=int((m & (g["stage"] >= 2)).sum()),
            MASLD_F3plus=int((m & (g["stage"] >= 3)).sum()),
            MASLD_F4=int((m & (g["stage"] == 4)).sum()),
            n_fasting=int(g["LBXGLU"].notna().sum())))
    return pd.DataFrame(recs).sort_values(["sex", "age"]).reset_index(drop=True)


def stratified_prevalence(coh, cut=288):
    """Weighted prevalence (%) + 95% CI half-width per stratum for key outcomes."""
    coh = add_strata(coh)
    recs = []
    for (sx, ab), g in coh.groupby(["sexlab", "ageband"]):
        w = g["WTMECPRP"].astype(float)
        m = g["LUXCAPM"] >= cut
        outcomes = {
            "Steatosis (MASLD)": m,
            "MASLD F2+ (significant)": m & (g["stage"] >= 2),
            "MASLD F3+ (advanced)": m & (g["stage"] >= 3),
        }
        for name, mask in outcomes.items():
            p, hw = fl.wfrac(mask, w)
            recs.append(dict(sex=sx, age=ab, outcome=name, pct=round(p, 1),
                             ci_lo=round(max(p - hw, 0), 1), ci_hi=round(p + hw, 1),
                             n_event=int(mask.sum()), N=len(g)))
    return pd.DataFrame(recs)


def stratified_correlation(coh, cut=288, risks=None):
    """Weighted Spearman rho of the MASLD F-stage (gated at `cut`) with each risk, per stratum."""
    risks = risks or STRAT_RISKS
    coh = add_strata(coh)
    rows = {}
    for (sx, ab), g in coh.groupby(["sexlab", "ageband"]):
        w = g["WTMECPRP"].astype(float)
        gated = pd.Series(np.where(g["LUXCAPM"] >= cut, g["stage"], 0).astype(float), index=g.index)
        rows[f"{sx} {ab}"] = {r: fl.wspearman(gated, g[fl.RISKS[r]], w)[0] for r in risks}
    return pd.DataFrame(rows).T[risks].round(3)
