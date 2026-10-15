"""Small survey-analysis helpers for NHANES-style complex designs.

Implements the three things this project needs and nothing more:

* ``svy_mean``  -- Taylor-linearised mean / proportion with stratified,
  clustered (PSU) variance and optional domain (subgroup) estimation.
* ``rake``      -- iterative proportional fitting of base weights to
  external population margins (poststratification calibration).
* ``svy_logit`` -- survey-weighted logistic regression with the usual
  stratified-cluster sandwich variance.

Confidence intervals use a t distribution with (PSUs - strata) degrees
of freedom, the convention in SUDAAN / R ``survey`` / Stata ``svy``.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _cluster_var(z, strata, psu):
    """Between-PSU, within-stratum variance of totals of ``z`` (n x k)."""
    z = np.atleast_2d(np.asarray(z, float))
    if z.shape[0] != len(strata):
        z = z.T
    d = pd.DataFrame(z)
    d['_s'] = np.asarray(strata)
    d['_p'] = np.asarray(psu)
    tot = d.groupby(['_s', '_p']).sum()
    k = z.shape[1]
    var = np.zeros((k, k))
    for _, g in tot.groupby(level='_s'):
        n = len(g)
        if n > 1:
            dev = g.values - g.values.mean(axis=0)
            var += n / (n - 1) * dev.T @ dev
    dof = len(tot) - tot.index.get_level_values('_s').nunique()
    return var, dof


def svy_mean(y, w, strata, psu, domain=None, level=0.95):
    """Weighted mean (or proportion if ``y`` is boolean) with design-based CI.

    ``domain`` is an optional boolean mask; rows outside the domain stay
    in the computation with zero weight so that PSU structure is kept
    (correct subgroup variance).
    Returns dict(est, se, lo, hi, n, dof).
    """
    y = np.asarray(y, float)
    w = np.asarray(w, float).copy()
    if domain is not None:
        dmask = np.asarray(domain, bool)
        w = np.where(dmask, w, 0.0)
        n = int(dmask.sum())
    else:
        n = len(y)
    y = np.where(np.isnan(y), 0.0, y)  # zero-weight rows may carry NaN
    W = w.sum()
    est = np.sum(w * y) / W
    z = w * (y - est) / W
    var, dof = _cluster_var(z, strata, psu)
    se = float(np.sqrt(var[0, 0]))
    t = stats.t.ppf(0.5 + level / 2, dof)
    return dict(est=est, se=se, lo=est - t * se, hi=est + t * se, n=n, dof=dof)


def rake(w, df, margins, tol=1e-10, maxit=200):
    """Rake base weights ``w`` so that the weighted distribution of each
    column named in ``margins`` matches the supplied population counts.

    ``margins`` maps column name -> dict(category -> population count).
    Returns calibrated weights scaled to the total population implied by
    the first margin.
    """
    w = np.asarray(w, float).copy()
    for _ in range(maxit):
        maxchg = 0.0
        for col, targ in margins.items():
            targ = pd.Series(targ, dtype=float)
            targ = targ / targ.sum()
            cur = pd.Series(w).groupby(df[col].astype(str).values).sum()
            cur = cur / cur.sum()
            factor = (targ.reindex(cur.index) / cur)
            f = factor.reindex(df[col].astype(str).values).values
            w = w * f
            maxchg = max(maxchg, float(np.abs(np.log(f)).max()))
        if maxchg < tol:
            break
    first = next(iter(margins.values()))
    return w * (float(sum(first.values())) / w.sum())


def svy_logit(X, y, w, strata, psu, names=None, maxit=50):
    """Survey-weighted logistic regression (pseudo-maximum likelihood).

    ``X`` must already contain an intercept column. Returns a DataFrame
    with coef, se, OR and 95% CI (t with design df).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    beta = np.zeros(X.shape[1])
    for _ in range(maxit):
        p = 1 / (1 + np.exp(-X @ beta))
        grad = X.T @ (w * (y - p))
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    H = (X * (w * p * (1 - p))[:, None]).T @ X
    scores = X * (w * (y - p))[:, None]
    G, dof = _cluster_var(scores, strata, psu)
    Hinv = np.linalg.inv(H)
    V = Hinv @ G @ Hinv
    se = np.sqrt(np.diag(V))
    t = stats.t.ppf(0.975, dof)
    out = pd.DataFrame({'coef': beta, 'se': se,
                        'OR': np.exp(beta),
                        'OR_lo': np.exp(beta - t * se),
                        'OR_hi': np.exp(beta + t * se)},
                       index=names if names is not None else range(X.shape[1]))
    out.attrs['dof'] = dof
    return out


def vermeulen_free_t(tt_ngdl, shbg_nmol, albumin_gdl):
    """Calculated free testosterone (ng/dL) by the Vermeulen (1999) method.

    Association constants: K_albumin = 3.6e4 L/mol, K_SHBG = 1e9 L/mol.
    """
    tt = np.asarray(tt_ngdl, float) * 0.0347 * 1e-9   # mol/L
    shbg = np.asarray(shbg_nmol, float) * 1e-9
    alb = np.asarray(albumin_gdl, float) * 10 / 69000  # g/dL -> mol/L
    N = 1 + 3.6e4 * alb
    Kt = 1e9
    a = N * Kt
    b = N + Kt * (shbg - tt)
    ft = (-b + np.sqrt(b ** 2 + 4 * a * tt)) / (2 * a)  # mol/L
    return ft * 1e9 / 0.0347  # ng/dL
