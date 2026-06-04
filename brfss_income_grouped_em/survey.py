"""Design-based variance for the income-health gradient.

BRFSS LLCP is a stratified, weighted sample. ``_PSU`` is unique within a state-year,
so the ultimate sampling unit is effectively the individual and the design reduces
to stratified sampling with unequal weights (``_STSTR`` = stratum, ``_LLCPWT`` =
weight). Two design-based variance estimators are provided:

* ``bootstrap_gradients`` (primary) -- a stratified resampling bootstrap (the
  Rao-Wu rescaling bootstrap; with BRFSS's large strata the finite-population
  ``n_h/(n_h-1)`` correction is negligible). Within each stratum we resample
  respondents with replacement and refit the *entire* pipeline (income model +
  outcome model), so the interval propagates all sources of estimation error.

* ``linearized_se`` (cross-check) -- a stratified Taylor-linearised sandwich on the
  outcome-model estimating equations, holding the income measurement model fixed.

No naive IID standard errors are produced anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from income_model import (Recoded, marginal_information, expit)


# ---------------------------------------------------------------------------
# stratified resampling
# ---------------------------------------------------------------------------
def make_resampler(strata):
    """Return ``sample(rng) -> index array`` that resamples respondents with
    replacement *within* each stratum, preserving per-stratum sample sizes."""
    strata = np.asarray(strata)
    order = np.argsort(strata, kind="stable")
    ss = strata[order]
    _, first, counts = np.unique(ss, return_index=True, return_counts=True)
    start_per = np.repeat(first, counts)        # (n,) sorted-space group start
    count_per = np.repeat(counts, counts)       # (n,) sorted-space group size
    n = strata.size

    def sample(rng):
        u = rng.random(n)
        off = np.minimum((u * count_per).astype(np.int64), count_per - 1)
        return order[start_per + off]

    return sample


def subset(d: Recoded, idx):
    return replace(d, X=d.X[idx], y=d.y[idx], w=d.w[idx], kind=d.kind[idx],
                   log_lo=d.log_lo[idx], log_hi=d.log_hi[idx], log_mid=d.log_mid[idx])


@dataclass
class CI:
    point: float
    se: float
    lo: float
    hi: float
    boot: np.ndarray

    def as_or(self):
        """Exponentiate (gradient is on the log-OR scale) -> OR CI."""
        return (float(np.exp(self.point)), float(np.exp(self.lo)), float(np.exp(self.hi)))


def bootstrap_gradients(d: Recoded, strata, fitters, B=200, seed=0,
                        points=None, log=None):
    """Run the stratified bootstrap.

    Parameters
    ----------
    fitters : dict name -> callable(Recoded) -> object with
        ``.gradient_log_or_per_doubling``.
    points : dict name -> full-sample point estimate (log-OR per doubling). If
        omitted it is computed from ``fitters`` on the full sample.

    Returns dict name -> :class:`CI` (point + 95% percentile interval on the
    log-OR-per-doubling scale; ``.as_or()`` converts to the OR scale).
    """
    resampler = make_resampler(strata)
    rng = np.random.default_rng(seed)
    if points is None:
        points = {name: fn(d).gradient_log_or_per_doubling for name, fn in fitters.items()}

    boot = {name: np.full(B, np.nan) for name in fitters}
    for b in range(B):
        idx = resampler(rng)
        db = subset(d, idx)
        for name, fn in fitters.items():
            try:
                boot[name][b] = fn(db).gradient_log_or_per_doubling
            except Exception:
                boot[name][b] = np.nan
        if log is not None and (b + 1) % max(1, B // 10) == 0:
            log(f"  bootstrap {b + 1}/{B}")

    out = {}
    for name in fitters:
        vals = boot[name][np.isfinite(boot[name])]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[name] = CI(point=float(points[name]), se=float(vals.std(ddof=1)),
                       lo=float(lo), hi=float(hi), boot=vals)
    return out


# ---------------------------------------------------------------------------
# linearized sandwich cross-check (outcome model, income model held fixed)
# ---------------------------------------------------------------------------
def _per_obs_score(params, X, y, w, I_nodes):
    """Per-respondent weighted score rows (n, p+1) of the marginal logit."""
    theta, eta = params[0], params[1:]
    lin = theta * I_nodes + (X @ eta)[:, None]
    logp = np.where(y[:, None] == 1, -np.logaddexp(0.0, -lin), -np.logaddexp(0.0, lin))
    mmax = logp.max(axis=1, keepdims=True)
    lse = mmax[:, 0] + np.log(np.exp(logp - mmax).mean(axis=1))
    post = np.exp(logp - lse[:, None])
    post = post / post.sum(axis=1, keepdims=True)
    resid = y[:, None] - expit(lin)
    wr = post * resid
    r_i = wr.sum(axis=1)
    s_theta = (wr * I_nodes).sum(axis=1)
    s = np.column_stack([s_theta, X * r_i[:, None]])
    return w[:, None] * s


def linearized_se(params, X, y, w, I_nodes, strata):
    """Stratified Taylor-linearised SE of the gradient (theta), income model fixed.

    Returns SE of ``theta`` on the log-OR-per-unit-log-income scale; multiply by
    ln 2 for the per-doubling scale.
    """
    strata = np.asarray(strata)
    H = marginal_information(params, X, y, w, I_nodes)   # observed information (Louis)
    Hinv = np.linalg.inv(H)
    s = _per_obs_score(params, X, y, w, I_nodes)       # (n, p+1)

    # stratified design variance of the total score
    order = np.argsort(strata, kind="stable")
    ss = strata[order]
    s_ord = s[order]
    uniq, first, counts = np.unique(ss, return_index=True, return_counts=True)
    V = np.zeros((s.shape[1], s.shape[1]))
    for g in range(uniq.size):
        sl = slice(first[g], first[g] + counts[g])
        nh = counts[g]
        if nh < 2:
            continue
        sh = s_ord[sl]
        sbar = sh.mean(axis=0)
        dev = sh - sbar
        V += (nh / (nh - 1)) * (dev.T @ dev)
    cov = Hinv @ V @ Hinv
    return float(np.sqrt(cov[0, 0]))
