"""Shared fitting/calibration logic for the NHANES LSM + CAP distribution project.

Pooled cohort: NHANES ``P_LUX`` (2017 - March 2020, weight ``WTMECPRP``) plus
``LUX_L`` (2021 - August 2023, weight ``WTMEC2YR``), each cycle's MEC weight
halved. Records kept: complete FibroScan exam (``LUAXSTAT == 1``), positive MEC
weight, and a non-missing measurement for the variable under study
(``LUXSMED`` for LSM, ``LUXCAPM`` for CAP).

This module centralises everything the numbered notebooks share:

* pooling / parsing of the two cycles (``build_pooled``, ``analysis_frame``)
* survey-weighted statistics (``w_mean`` ... ``weighted_ks``)
* the fibrosis-stage cutoff ladder (6/8/10/15 kPa) and its calibration weights
* the multi-cutoff lognormal calibration that prioritises F1/F2/F3
  (``fit_lognorm_multicut``)
* logit-space smoothing of small-cell calibration targets
  (``smooth_targets_logit``)
* the CAP moment-match fit and a family-selection helper (``cap_moment_fit``,
  ``cap_family_ks``)
* the single-anchor and alternative fits kept for the method-comparison notebook

The downstream consumer (``vivarium_csu_mace_rct``) rebuilds the LSM exposure
from ``(mean_kpa, sd_kpa)`` with a hard-coded ``"lognormal"`` type and routes
simulants purely by threshold, so the realised stage-share vector equals the
fitted lognormal's CDF differences at the ladder cutoffs. That is why LSM stays
a two-parameter lognormal and we calibrate its cumulative shares directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
RAW = DATA / "raw" / "nhanes"
DERIVED = DATA / "derived"

RAW_P = RAW / "2017_2020_prepandemic"  # P_DEMO, P_LUX  (WTMECPRP)
RAW_L = RAW / "2021_2023"              # DEMO_L, LUX_L  (WTMEC2YR)

POOLED_PARQUET = DERIVED / "nhanes_p_lux_plus_l.parquet"

# ---------------------------------------------------------------------------
# Fibrosis-stage ladder (VCTE LSM cutoffs, kPa) -- matches the repo's staging
# authority, nhanes_fibrosis_modeling/fibrosis_lib.py:
#   F0 <6, F1 6-<8, F2 8-<10, F3 10-<15, F4 >=15
# Calibration weights favour the F0/F1, F1/F2, F2/F3 boundaries over F3/F4 so
# the fit spends its two degrees of freedom where accuracy matters most.
# ---------------------------------------------------------------------------
LSM_CUTOFFS = [6.0, 8.0, 10.0, 15.0]
LSM_STAGES = ["F0", "F1", "F2", "F3", "F4"]
# Per-stage calibration weights (F0, F1, F2, F3, F4). The fit minimises the
# weighted squared error of the *stage shares* -- the routing fractions the
# simulation actually reads -- so F1/F2/F3 are weighted above F0/F4.
LSM_STAGE_WEIGHTS = [1.0, 2.0, 2.0, 2.0, 0.5]

# CAP steatosis-grade cutoffs (dB/m), Karlas 2017 mixed-etiology set
# (>=S1 / >=S2 / >=S3). Informational only: CAP is moment-matched, not
# grade-calibrated. 288 is the sibling project's "any steatosis" baseline gate.
CAP_CUTOFFS = [248.0, 268.0, 280.0]
CAP_GRADES = ["S0", "S1", "S2", "S3"]
CAP_STEATOSIS_GATE = 288.0

# Legacy single-anchor cutoff kept only for the method-comparison notebook.
F4_CUTOFF = 12.5
# FibroScan dynamic range the downstream sampler clips LSM to.
LSM_FLOOR, LSM_CEIL = 1.5, 75.0


# ---------------------------------------------------------------------------
# Survey-weighted statistics
# ---------------------------------------------------------------------------
def w_mean(y, w):
    return float(np.average(y, weights=w))


def w_var(y, w):
    m = w_mean(y, w)
    return float(np.average((y - m) ** 2, weights=w))


def w_sd(y, w):
    return float(np.sqrt(w_var(y, w)))


def w_quantile(y, w, q):
    o = np.argsort(y)
    ys, ws = np.asarray(y)[o], np.asarray(w)[o]
    cw = np.cumsum(ws) / ws.sum()
    return float(np.interp(q, cw, ys))


def w_share_above(y, w, cutoff):
    return float(np.average((np.asarray(y) >= cutoff).astype(float), weights=w))


def w_log_mean(y, w):
    return float(np.average(np.log(y), weights=w))


def w_log_sd(y, w):
    m = w_log_mean(y, w)
    return float(np.sqrt(np.average((np.log(y) - m) ** 2, weights=w)))


def n_eff(w):
    """Kish effective sample size."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def weighted_ks(y, w, cdf):
    """Survey-weighted Kolmogorov-Smirnov distance to a CDF callable."""
    o = np.argsort(y)
    ys, ws = np.asarray(y)[o], np.asarray(w)[o]
    emp = np.cumsum(ws) / ws.sum()
    return float(np.max(np.abs(emp - cdf(ys))))


# ---------------------------------------------------------------------------
# Stage / grade shares
# ---------------------------------------------------------------------------
def empirical_cum_at(y, w, cutoffs):
    """Weighted CDF value F(c) = P[X < c] at each cutoff."""
    return np.array([1.0 - w_share_above(y, w, c) for c in cutoffs], dtype=float)


def shares_from_cum(cum):
    """Per-bin shares from cumulative values at ordered cutoffs."""
    edges = np.concatenate([[0.0], np.asarray(cum, float), [1.0]])
    return np.diff(edges)


def empirical_stage_shares(y, w, cutoffs):
    return shares_from_cum(empirical_cum_at(y, w, cutoffs))


def lognorm_stage_shares(mu, sigma, cutoffs):
    cum = stats.lognorm.cdf(np.asarray(cutoffs, float), sigma, scale=np.exp(mu))
    return shares_from_cum(cum)


def normal_grade_shares(mean, sd, cutoffs):
    cum = stats.norm.cdf(np.asarray(cutoffs, float), loc=mean, scale=sd)
    return shares_from_cum(cum)


# ---------------------------------------------------------------------------
# Lognormal <-> arithmetic moment conversions
# ---------------------------------------------------------------------------
def arith_to_lognorm(mean, sd):
    """(mean, sd) -> (mu, sigma) of the lognormal with those moments."""
    sigma2 = np.log(1.0 + (sd / mean) ** 2)
    mu = np.log(mean ** 2 / np.sqrt(mean ** 2 + sd ** 2))
    return float(mu), float(np.sqrt(sigma2))


def lognorm_to_arith(mu, sigma):
    """(mu, sigma) -> (mean, sd) of the lognormal."""
    mean = float(np.exp(mu + sigma ** 2 / 2.0))
    sd = float(mean * np.sqrt(np.exp(sigma ** 2) - 1.0))
    return mean, sd


# ---------------------------------------------------------------------------
# The core: weighted multi-cutoff lognormal calibration
# ---------------------------------------------------------------------------
def fit_lognorm_multicut(y, w, cutoffs=LSM_CUTOFFS, stage_weights=LSM_STAGE_WEIGHTS,
                         target_cum=None):
    """Fit a lognormal ``(mu, sigma)`` by minimising the weighted squared error
    between its implied *stage shares* and the target shares at ``cutoffs``.

    The stage shares (per-bin masses) are exactly the routing fractions the
    downstream simulation reads, so the objective targets them directly rather
    than the cumulative CDF. ``stage_weights`` is a per-stage policy-priority
    vector (F0..F4); it up-weights F1/F2/F3 -- not an inverse-variance weighting,
    which would up-weight the tiny F4 tail (the opposite of the goal).

    ``target_cum`` (optional) supplies cumulative shares F(c) to calibrate to --
    e.g. smoothed targets from :func:`smooth_targets_logit`; the objective uses
    their differenced stage shares. When omitted the raw empirical shares are used.

    Over-determined by design: four cutoffs, two parameters. The residuals are
    the honest report of how well a two-parameter family can hit the ladder --
    a unimodal lognormal cannot reproduce a non-monotone stage profile (e.g. an
    empirical F2 dip below F3).
    """
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    cutoffs = np.asarray(cutoffs, float)
    sw = np.asarray(stage_weights, float)
    if target_cum is None:
        target_shares = empirical_stage_shares(y, w, cutoffs)
    else:
        target_shares = shares_from_cum(np.asarray(target_cum, float))

    def obj(p):
        mu, log_sg = p
        sg = np.exp(log_sg)
        pred = shares_from_cum(stats.lognorm.cdf(cutoffs, sg, scale=np.exp(mu)))
        return float(np.sum(sw * (pred - target_shares) ** 2))

    # warm start from the log-moment match
    mu0 = w_log_mean(y, w)
    sg0 = max(w_log_sd(y, w), 1e-2)

    # coarse grid guard against local minima
    grid_start, grid_val = (mu0, np.log(sg0)), np.inf
    for gm in np.linspace(mu0 - 1.0, mu0 + 1.0, 21):
        for gs in np.linspace(0.05, 1.3, 26):
            v = obj((gm, np.log(gs)))
            if v < grid_val:
                grid_val, grid_start = v, (gm, np.log(gs))

    best = None
    for start in ([mu0, np.log(sg0)], list(grid_start)):
        r = minimize(obj, start, method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 5000})
        if best is None or r.fun < best.fun:
            best = r
    return float(best.x[0]), float(np.exp(best.x[1]))


def smooth_targets_logit(ages, cum_by_cut, neff, order=2):
    """Smooth per-cutoff cumulative shares across age within a sex.

    ``cum_by_cut`` is an ``(n_ages, n_cut)`` array of F(c) values; ``neff`` the
    per-band effective sample sizes used as fit weights. Smoothing happens in
    logit space with a low-order (``order``, default quadratic) polynomial in
    age, so populous bands dominate and the real age gradient in F3/F4 is
    preserved rather than flattened. Returns a same-shaped array; each row is
    forced non-decreasing across cutoffs so the implied stage shares stay
    non-negative.
    """
    ages = np.asarray(ages, float)
    neff = np.asarray(neff, float)
    cum = np.asarray(cum_by_cut, float)
    out = np.empty_like(cum)
    eps = 1e-3
    eff_order = min(order, len(ages) - 1)
    for j in range(cum.shape[1]):
        p = np.clip(cum[:, j], eps, 1.0 - eps)
        z = np.log(p / (1.0 - p))  # logit
        if eff_order >= 1:
            X = np.vstack([ages ** k for k in range(eff_order + 1)]).T
            W = np.diag(neff)
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ z)
            zs = X @ beta
        else:
            zs = np.full_like(z, np.average(z, weights=neff))
        out[:, j] = 1.0 / (1.0 + np.exp(-zs))
    return np.maximum.accumulate(out, axis=1)


# ---------------------------------------------------------------------------
# CAP: moment-matched distribution (family selection is informational)
# ---------------------------------------------------------------------------
def cap_moment_fit(y, w):
    """Weighted (mean, sd) of CAP (dB/m). Default family is Normal."""
    return w_mean(y, w), w_sd(y, w)


def cap_family_ks(y, w):
    """Weighted KS distance of moment-matched Normal / lognormal / scaled-Beta
    fits to the empirical CAP CDF. Justifies the Normal default."""
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    m, s = cap_moment_fit(y, w)
    out = {"normal": weighted_ks(y, w, lambda v: stats.norm.cdf(v, m, s))}
    mu, sg = arith_to_lognorm(m, s)
    out["lognormal"] = weighted_ks(y, w, lambda v: stats.lognorm.cdf(v, sg, scale=np.exp(mu)))
    lo, hi = float(y.min()), float(y.max())
    rng = hi - lo
    if rng > 0:
        mm = (m - lo) / rng
        vv = (s / rng) ** 2
        common = mm * (1.0 - mm) / vv - 1.0
        a, b = mm * common, (1.0 - mm) * common
        if a > 0 and b > 0:
            out["beta"] = weighted_ks(
                y, w, lambda v: stats.beta.cdf(np.clip((v - lo) / rng, 0, 1), a, b))
    return out


# ---------------------------------------------------------------------------
# Two-level model helpers (categorical stage + within-stage continuous)
# ---------------------------------------------------------------------------
# For the alternative architecture evaluated in notebook 07: assign a fibrosis /
# steatosis category from the empirical joint, then draw a measurement from a
# within-category truncated distribution. This reproduces the stage shares
# exactly and represents the wide within-F4 tail (LSM ~15-75 kPa) that a single
# lognormal cannot. These are shared so future production code can reuse them.
def stage_edges(cutoffs, floor, ceil):
    """Full bin edges ``(floor, *cutoffs, ceil)`` for a stage ladder."""
    return [float(floor)] + [float(c) for c in cutoffs] + [float(ceil)]


def fit_truncated_lognorm(y, w, lo, hi):
    """Weighted MLE of a lognormal truncated to ``[lo, hi)``. Falls back to a
    moment-based guess when the cell is too thin (< 5 obs) to optimise."""
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    if len(y) < 5:
        c = np.average(y, weights=w) if len(y) else (lo + hi) / 2.0
        return float(np.log(np.clip(c, lo + 1e-3, hi - 1e-3))), 0.3
    mu0 = float(np.average(np.log(y), weights=w))
    sg0 = max(float(np.sqrt(np.average((np.log(y) - mu0) ** 2, weights=w))), 0.05)

    def nll(p):
        mu, ls = p
        sg = np.exp(ls)
        num = stats.lognorm.logpdf(y, sg, scale=np.exp(mu))
        den = np.log(max(stats.lognorm.cdf(hi, sg, scale=np.exp(mu))
                         - stats.lognorm.cdf(lo, sg, scale=np.exp(mu)), 1e-12))
        return -np.sum(w * (num - den))

    r = minimize(nll, [mu0, np.log(sg0)], method="Nelder-Mead")
    return float(r.x[0]), float(np.exp(r.x[1]))


def truncated_lognorm_cdf(x, mu, sigma, lo, hi):
    """Conditional CDF of a lognormal truncated to ``[lo, hi)``."""
    x = np.atleast_1d(np.asarray(x, float))
    flo = stats.lognorm.cdf(lo, sigma, scale=np.exp(mu))
    fhi = stats.lognorm.cdf(hi, sigma, scale=np.exp(mu))
    return np.clip((stats.lognorm.cdf(x, sigma, scale=np.exp(mu)) - flo)
                   / max(fhi - flo, 1e-12), 0.0, 1.0)


def truncated_lognorm_rvs(mu, sigma, lo, hi, size, rng):
    """Draw ``size`` samples from a lognormal truncated to ``[lo, hi)`` via
    inverse-CDF sampling (``rng`` is a numpy Generator)."""
    flo = stats.lognorm.cdf(lo, sigma, scale=np.exp(mu))
    fhi = stats.lognorm.cdf(hi, sigma, scale=np.exp(mu))
    u = rng.uniform(flo, fhi, size=size)
    return stats.lognorm.ppf(u, sigma, scale=np.exp(mu))


def stage_mixture_cdf(x, shares, params, edges):
    """Mixture CDF: ``sum_s shares[s] * truncated_lognorm_cdf`` over bin s.
    ``params`` is a list of ``(mu, sigma)`` per stage; ``edges`` has
    ``len(shares) + 1`` entries."""
    x = np.atleast_1d(np.asarray(x, float))
    out = np.zeros_like(x)
    for s, (mu, sg) in enumerate(params):
        out += shares[s] * truncated_lognorm_cdf(x, mu, sg, edges[s], edges[s + 1])
    return out


# ---------------------------------------------------------------------------
# Legacy single-anchor + alternative fits (method-comparison notebook only)
# ---------------------------------------------------------------------------
def fit_log_mm(y, w):
    """Log-moment match: mu, sigma = weighted mean, sd of log(y)."""
    return w_log_mean(y, w), w_log_sd(y, w)


def fit_arith_mm(mean, sd):
    return arith_to_lognorm(mean, sd)


def fit_gamma_mom(mean, sd):
    k = (mean / sd) ** 2
    theta = sd ** 2 / mean
    return k, theta


def fit_weibull_mle(y, w):
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    init = stats.weibull_min.fit(y, floc=0)

    def negll(p):
        c, s = p
        if c <= 0 or s <= 0:
            return 1e12
        return -np.sum(w * stats.weibull_min.logpdf(y, c, scale=s))

    r = minimize(negll, [init[0], init[2]], method="Nelder-Mead")
    return float(r.x[0]), float(r.x[1])


def fit_lognorm_f4(median, target_f4, cutoff=F4_CUTOFF):
    """Single-anchor calibration: mu = log(median); sigma so P[X>=cutoff]=target."""
    mu = np.log(median)
    if not (0.0 < target_f4 < 1.0):
        return mu, np.nan
    z = stats.norm.ppf(1.0 - target_f4)
    if z <= 0:
        return mu, np.nan
    return mu, max(0.0, (np.log(cutoff) - mu) / z)


# ---------------------------------------------------------------------------
# Pooling / parsing
# ---------------------------------------------------------------------------
def _parse_cycle(demo_path, lux_path, wt_col, cycle):
    demo = pd.read_sas(demo_path)[
        ["SEQN", "RIAGENDR", "RIDAGEYR", wt_col, "SDMVPSU", "SDMVSTRA"]]
    lux = pd.read_sas(lux_path)[
        ["SEQN", "LUXSMED", "LUXSIQR", "LUXCAPM", "LUXCPIQR", "LUAXSTAT"]]
    df = demo.merge(lux, on="SEQN", how="left")
    out = pd.DataFrame({
        "SEQN": df["SEQN"].astype("int64").astype(str),
        "cycle": cycle,
        "sex": df["RIAGENDR"].map({1.0: "Male", 2.0: "Female"}),
        "age_years": df["RIDAGEYR"].astype(float),
        "MEC_WT": df[wt_col].astype(float),
        "SDMVPSU": df["SDMVPSU"].astype("Int64"),
        "SDMVSTRA": df["SDMVSTRA"].astype("Int64"),
        "LSM_KPA": df["LUXSMED"].astype(float),
        "LSM_IQR": df["LUXSIQR"].astype(float),
        "CAP_DBM": df["LUXCAPM"].astype(float),
        "CAP_IQR": df["LUXCPIQR"].astype(float),
        "exam_complete": df["LUAXSTAT"] == 1.0,
    })
    return out


def build_pooled():
    """Parse and pool both cycles; halve each cycle's MEC weight for pooling."""
    p = _parse_cycle(RAW_P / "P_DEMO.xpt", RAW_P / "P_LUX.xpt", "WTMECPRP", "2017_2020")
    l = _parse_cycle(RAW_L / "DEMO_L.xpt", RAW_L / "LUX_L.xpt", "WTMEC2YR", "2021_2023")
    pool = pd.concat([p, l], ignore_index=True)
    pool["MEC_WT_POOL"] = pool["MEC_WT"] * 0.5
    return pool


def analysis_frame(pool, variable="LSM"):
    """Complete-exam, positive-weight records with the chosen variable present.

    ``variable`` is ``"LSM"``, ``"CAP"``, or ``"both"``.
    """
    m = pool["exam_complete"].fillna(False) & pool["MEC_WT"].fillna(0).gt(0)
    if variable in ("LSM", "both"):
        m = m & pool["LSM_KPA"].notna()
    if variable in ("CAP", "both"):
        m = m & pool["CAP_DBM"].notna()
    return pool[m].copy()


def load_pooled(variable="LSM"):
    """Convenience reader: pooled parquet -> analysis frame for a variable."""
    pool = pd.read_parquet(POOLED_PARQUET)
    return analysis_frame(pool, variable=variable)


# ---------------------------------------------------------------------------
# Age banding (top-code aware)
# ---------------------------------------------------------------------------
# NHANES top-codes age at 80 in both cycles, so the terminal band is the entire
# 80+ mixture, not a 5-year bin. Bands below it are ordinary 5-year bins.
def age_band(age, start=60, top=80):
    """Return (age_start, age_end, label) for a 5-year band, open-ended at top."""
    if age >= top:
        return float(top), 125.0, f"{top}+"
    lo = start + int((age - start) // 5) * 5
    return float(lo), float(lo + 5), f"{lo}-{lo + 4}"


def band_edges(start=60, top=80):
    """List of (age_start, age_end, label) for the fitted bands, 80+ terminal."""
    bands = []
    a = start
    while a < top:
        bands.append((float(a), float(a + 5), f"{a}-{a + 4}"))
        a += 5
    bands.append((float(top), 125.0, f"{top}+"))
    return bands


def assign_band(df, start=60, top=80):
    """Add age_start / age_end / age_group columns (top-code aware)."""
    out = df.copy()
    bands = df["age_years"].apply(lambda a: age_band(a, start=start, top=top))
    out["age_start"] = [b[0] for b in bands]
    out["age_end"] = [b[1] for b in bands]
    out["age_group"] = [b[2] for b in bands]
    return out


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------
def calibration_meta(extra=None):
    meta = {
        "lsm_dist_family": "lognormal",
        "lsm_cutoffs_kpa": LSM_CUTOFFS,
        "lsm_stage_labels": LSM_STAGES,
        "lsm_stage_weights": LSM_STAGE_WEIGHTS,
        "lsm_calibration_objective": (
            "minimise sum_s stage_weight_s * (fit_share_s - target_share_s)^2 "
            "over stages F0..F4; target shares optionally smoothed across age"
        ),
        "lsm_clip_kpa": [LSM_FLOOR, LSM_CEIL],
        "cap_dist_family": "normal",
        "cap_cutoffs_dbm": CAP_CUTOFFS,
        "cap_grade_labels": CAP_GRADES,
        "cap_steatosis_gate_dbm": CAP_STEATOSIS_GATE,
        "pooling": (
            "P_LUX (2017-Mar 2020, WTMECPRP) + LUX_L (2021-Aug 2023, WTMEC2YR); "
            "each cycle MEC weight halved"
        ),
    }
    if extra:
        meta.update(extra)
    return meta


def write_meta(path, extra=None):
    meta = calibration_meta(extra)
    Path(path).write_text(json.dumps(meta, indent=2) + "\n")
    return meta
