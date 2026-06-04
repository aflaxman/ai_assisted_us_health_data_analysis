"""Grouped-data EM income model + latent-income logistic regression.

This module is data-agnostic: it operates on numeric arrays, so the same engine
serves models A, B, and C and the survey bootstrap. BRFSS-specific recoding
(bracket boundaries, missingness codes) lives in ``recode.py``.

The two pieces
--------------
1. ``fit_grouped_lognormal`` -- the measurement model. Log-income ``I`` is
   Normal with a covariate-dependent mean, ``I | X ~ N(X beta, sigma^2)``, and we
   observe only the interval ``(lo, hi]`` each respondent falls in (bracketed
   income). This is the grouped-continuous-data likelihood (Heitjan 1989). We
   maximise it with EM: the E-step takes truncated-normal moments within each
   bracket; the M-step is a weighted least-squares update of ``beta`` plus a
   variance update. The open-ended top bracket is ``(ln 200k, +inf)`` -- a proper
   right-censored term, never an arbitrary midpoint.

2. ``fit_marginal_logit`` -- the outcome model. The binary outcome follows
   ``logit P(Y=1) = theta * I + X eta`` with ``I`` latent. We integrate ``I`` out
   against its conditional distribution using an equal-probability quadrature grid
   (``make_nodes``) and maximise the resulting marginal likelihood directly by
   L-BFGS with the exact (Fisher-identity) gradient. ``theta`` is the
   income-health gradient; ``exp(theta * ln 2)`` is the odds ratio per doubling of
   income.

Models A/B/C differ only in how each respondent's latent income is represented
(see ``fit_A``/``fit_B``/``fit_C``):

* A -- a single node at the bracket midpoint (no integration), bracketed only.
* B -- bracket-truncated quadrature, bracketed only.
* C -- bracket-truncated for bracketed; full conditional for don't-know (MAR);
  exponentially tilted conditional for refused (MNAR sensitivity parameter gamma).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, log_ndtr, ndtri

LN2 = np.log(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _phi(x):
    """Standard-normal pdf, with 0 at +-inf endpoints."""
    out = np.zeros_like(np.asarray(x, float))
    fin = np.isfinite(x)
    out[fin] = _INV_SQRT_2PI * np.exp(-0.5 * np.asarray(x, float)[fin] ** 2)
    return out


def _truncnorm_moments(a, b, loc, scale):
    """Mean and variance of N(loc, scale^2) truncated to (loc+a*scale, loc+b*scale],
    where ``a``/``b`` are standardised bounds (``+-inf`` allowed). Fast analytic
    formulas (avoids scipy.stats overhead in the EM inner loop)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    Phi_a = _norm_cdf(a)
    Phi_b = _norm_cdf(b)
    Z = np.clip(Phi_b - Phi_a, 1e-300, None)
    pa, pb = _phi(a), _phi(b)
    ratio = (pa - pb) / Z
    mean = loc + scale * ratio
    # a*phi(a) and b*phi(b): 0 at infinite endpoints (limit x*phi(x) -> 0).
    # phi already returns 0 there; zero the bound too so we never form inf*0.
    afa = np.where(np.isfinite(a), a, 0.0) * pa
    bfb = np.where(np.isfinite(b), b, 0.0) * pb
    var = scale ** 2 * (1.0 + (afa - bfb) / Z - ratio ** 2)
    var = np.clip(var, 1e-12, None)
    return mean, var


# ---------------------------------------------------------------------------
# numerically stable helpers
# ---------------------------------------------------------------------------
def _log1pexp(x):
    """Stable log(1 + exp(x))."""
    out = np.logaddexp(0.0, x)
    return out


def _bernoulli_loglik(lin, y):
    """log P(Y=y | linear predictor lin) for the logistic model, elementwise.

    log p1 = -log1pexp(-lin); log p0 = -log1pexp(lin).
    """
    # y broadcast against lin (which may be 2-D over quadrature nodes)
    return np.where(y == 1, -_log1pexp(-lin), -_log1pexp(lin))


# ---------------------------------------------------------------------------
# 1. grouped-data lognormal measurement model (EM)
# ---------------------------------------------------------------------------
@dataclass
class IncomeModel:
    beta: np.ndarray      # coefficients for E[log-income] = X beta
    sigma: float          # residual SD of log-income
    loglik: float         # final weighted grouped-data log-likelihood
    n_iter: int
    converged: bool


def _grouped_loglik(X, log_lo, log_hi, w, beta, sigma):
    """Weighted observed-data log-likelihood of the interval (grouped) model."""
    m = X @ beta
    a = (log_lo - m) / sigma
    b = (log_hi - m) / sigma
    # log[ Phi(b) - Phi(a) ] via stable difference of log_ndtr
    log_hi_cdf = log_ndtr(b)
    log_lo_cdf = log_ndtr(a)
    # Phi(b) - Phi(a) = exp(log_ndtr(b)) - exp(log_ndtr(a)); stable via log-sum-exp
    diff = np.exp(log_hi_cdf) - np.exp(log_lo_cdf)
    diff = np.clip(diff, 1e-300, None)
    return float(np.sum(w * np.log(diff)))


def fit_grouped_lognormal(X, log_lo, log_hi, w, max_iter=500, tol=1e-8,
                          beta0=None, sigma0=None):
    """EM for ``I | X ~ N(X beta, sigma^2)`` observed only as the interval (lo, hi].

    Parameters
    ----------
    X : (n, p) design matrix (include an intercept column).
    log_lo, log_hi : (n,) interval endpoints on the log-income scale. Use
        ``-np.inf`` / ``np.inf`` for open ends (bottom and top brackets).
    w : (n,) survey weights.

    Returns
    -------
    IncomeModel
    """
    X = np.asarray(X, float)
    log_lo = np.asarray(log_lo, float)
    log_hi = np.asarray(log_hi, float)
    w = np.asarray(w, float)
    n, p = X.shape

    # --- initialise from interval midpoints (finite proxy) ---
    if beta0 is None or sigma0 is None:
        finite_lo = np.where(np.isfinite(log_lo), log_lo, log_hi - 1.0)
        finite_hi = np.where(np.isfinite(log_hi), log_hi, log_lo + 1.0)
        mid = 0.5 * (finite_lo + finite_hi)
        WX = X * w[:, None]
        beta = np.linalg.solve(WX.T @ X, WX.T @ mid)
        resid = mid - X @ beta
        sigma = np.sqrt(np.sum(w * resid ** 2) / np.sum(w))
    else:
        beta, sigma = np.asarray(beta0, float).copy(), float(sigma0)

    ll_old = -np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        m = X @ beta
        a = (log_lo - m) / sigma
        b = (log_hi - m) / sigma
        # truncated-normal conditional moments within each bracket
        mean_t, var_t = _truncnorm_moments(a, b, m, sigma)

        # M-step: weighted least squares of E[I] on X, plus variance update
        WX = X * w[:, None]
        beta = np.linalg.solve(WX.T @ X, WX.T @ mean_t)
        resid = mean_t - X @ beta
        sigma = np.sqrt(np.sum(w * (var_t + resid ** 2)) / np.sum(w))

        ll = _grouped_loglik(X, log_lo, log_hi, w, beta, sigma)
        if ll - ll_old < tol * (abs(ll_old) + tol):
            converged = True
            ll_old = ll
            break
        ll_old = ll
    return IncomeModel(beta=beta, sigma=float(sigma), loglik=ll_old,
                       n_iter=it, converged=converged)


# ---------------------------------------------------------------------------
# 2. quadrature nodes for latent income
# ---------------------------------------------------------------------------
def make_nodes(m, sigma, log_lo, log_hi, K):
    """Equal-probability quadrature nodes for ``I ~ N(m, sigma^2)`` truncated to
    ``(log_lo, log_hi]``.

    Each respondent gets ``K`` nodes at the truncated-normal quantiles
    ``u = (k - 0.5) / K``; the prior weight on every node is ``1/K``. This handles
    open tails (``+-inf`` bounds) and arbitrary truncation exactly and smoothly.

    Returns ``nodes`` of shape ``(n, K)`` on the log-income scale.
    """
    m = np.asarray(m, float)
    n = m.shape[0]
    sigma = float(sigma)
    log_lo = np.broadcast_to(np.asarray(log_lo, float), (n,))
    log_hi = np.broadcast_to(np.asarray(log_hi, float), (n,))

    lo_cdf = _norm_cdf((log_lo - m) / sigma)          # (n,)
    hi_cdf = _norm_cdf((log_hi - m) / sigma)
    u = (np.arange(K) + 0.5) / K                       # (K,)
    # map probability grid into the truncated interval
    p = lo_cdf[:, None] + u[None, :] * (hi_cdf - lo_cdf)[:, None]   # (n, K)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    nodes = m[:, None] + sigma * ndtri(p)
    return nodes


def _norm_cdf(z):
    return np.exp(log_ndtr(z))


# ---------------------------------------------------------------------------
# 3. latent-income logistic regression (marginal MLE by Newton)
# ---------------------------------------------------------------------------
@dataclass
class OutcomeFit:
    theta: float                 # log-OR per unit log-income
    eta: np.ndarray              # remaining coefficients (incl. intercept)
    loglik: float
    or_per_doubling: float       # exp(theta * ln 2)
    log_or_per_doubling: float   # theta * ln 2
    success: bool
    params: np.ndarray           # [theta, *eta]


def _marginal_obj(params, X, y, w, I_nodes, want_hess=False):
    """Weighted marginal log-likelihood, score, and (optionally) observed
    information for the latent-income logistic model.

    Marginal likelihood per respondent: ``L_i = (1/K) sum_k P(y_i | theta I_ik + X_i eta)``.
    The score is the posterior-weighted complete-data score (Fisher identity); the
    observed information uses Louis's formula
    ``J = E_post[complete info] - Var_post[complete score]`` -- both obtained from one
    sweep over the (n, K) grid, so Newton converges in a handful of iterations.

    Returns ``(loglik, score, J)`` where ``J`` is ``None`` unless ``want_hess``.
    Score/J are on the maximisation (log-likelihood) scale.
    """
    theta = params[0]
    eta = params[1:]
    lin = theta * I_nodes + (X @ eta)[:, None]          # (n, K)
    sgn = (2.0 * y - 1.0)[:, None]
    logpy = -np.logaddexp(0.0, -sgn * lin)              # log P(y | node), one call
    M = logpy.max(axis=1, keepdims=True)
    ex = np.exp(logpy - M)
    sumex = ex.sum(axis=1)
    loglik = float(np.sum(w * (M[:, 0] + np.log(sumex) - np.log(I_nodes.shape[1]))))
    post = ex / sumex[:, None]                          # (n, K), sums to 1 over k

    p = expit(lin)
    r = y[:, None] - p                                  # residual
    gr = post * r                                       # (n, K)
    c0 = gr.sum(axis=1)                                 # marginal residual r_i
    c1 = (gr * I_nodes).sum(axis=1)                     # theta-direction
    g_theta = float(np.sum(w * c1))
    g_eta = X.T @ (w * c0)
    score = np.concatenate([[g_theta], g_eta])

    J = None
    if want_hess:
        V = p * (1.0 - p)
        a = post * V
        A0 = a.sum(axis=1); A1 = (a * I_nodes).sum(axis=1); A2 = (a * I_nodes ** 2).sum(axis=1)
        b = post * r ** 2
        B0 = b.sum(axis=1); B1 = (b * I_nodes).sum(axis=1); B2 = (b * I_nodes ** 2).sum(axis=1)
        wv = w
        # E[complete info]
        T1_00 = float(np.sum(wv * A2))
        T1_0x = X.T @ (wv * A1)
        T1_xx = X.T @ (X * (wv * A0)[:, None])
        # Var[complete score] = E[uu'] - ubar ubar'
        E_00 = float(np.sum(wv * B2)); E_0x = X.T @ (wv * B1); E_xx = X.T @ (X * (wv * B0)[:, None])
        U_00 = float(np.sum(wv * c1 * c1)); U_0x = X.T @ (wv * c1 * c0)
        U_xx = X.T @ (X * (wv * c0 * c0)[:, None])
        J_00 = T1_00 - (E_00 - U_00)
        J_0x = T1_0x - (E_0x - U_0x)
        J_xx = T1_xx - (E_xx - U_xx)
        m = params.size
        J = np.empty((m, m))
        J[0, 0] = J_00
        J[0, 1:] = J_0x
        J[1:, 0] = J_0x
        J[1:, 1:] = J_xx
    return loglik, score, J


def _neg_loglik_grad(params, X, y, w, I_nodes):
    """Negative marginal log-likelihood and its gradient (for external callers)."""
    ll, score, _ = _marginal_obj(params, X, y, w, I_nodes, want_hess=False)
    return -ll, -score


def marginal_information(params, X, y, w, I_nodes):
    """Observed information matrix (Louis) at ``params`` -- the sandwich 'bread'."""
    return _marginal_obj(params, X, y, w, I_nodes, want_hess=True)[2]


def fit_marginal_logit(X, y, w, I_nodes, init=None, max_iter=50, tol=1e-8):
    """Maximise the marginal logistic likelihood with latent income, by damped
    Newton with the analytic observed-information Hessian.

    ``I_nodes`` is ``(n, K)`` (use ``K=1`` for a point proxy, e.g. model A).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    I_nodes = np.asarray(I_nodes, float)
    if I_nodes.ndim == 1:
        I_nodes = I_nodes[:, None]

    if init is None:
        Ibar = I_nodes.mean(axis=1)
        Z = np.column_stack([Ibar, X])
        params = _irls_logit(Z, y, w, n_steps=8)
    else:
        params = np.asarray(init, float).copy()

    ll, score, J = _marginal_obj(params, X, y, w, I_nodes, want_hess=True)
    success = False
    for _ in range(max_iter):
        try:
            step = np.linalg.solve(J, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, score, rcond=None)[0]
        # damped line search using the cheap (loglik+score) eval; the expensive
        # Hessian is computed only once, at the accepted iterate.
        t = 1.0
        improved = False
        for _ls in range(30):
            ll_c, score_c, _ = _marginal_obj(params + t * step, X, y, w, I_nodes)
            if np.isfinite(ll_c) and ll_c >= ll - 1e-12:
                params = params + t * step
                ll, score = ll_c, score_c
                improved = True
                break
            t *= 0.5
        if not improved:
            break
        if np.max(np.abs(step * t)) < tol:
            success = True
            break
        _, _, J = _marginal_obj(params, X, y, w, I_nodes, want_hess=True)
    theta = float(params[0])
    return OutcomeFit(theta=theta, eta=params[1:], loglik=float(ll),
                      or_per_doubling=float(np.exp(theta * LN2)),
                      log_or_per_doubling=float(theta * LN2),
                      success=bool(success), params=params)


def _irls_logit(Z, y, w, n_steps=25, tol=1e-10):
    """Plain weighted logistic regression via IRLS (used for warm starts and A)."""
    Z = np.asarray(Z, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    beta = np.zeros(Z.shape[1])
    for _ in range(n_steps):
        eta = Z @ beta
        mu = expit(eta)
        s = np.clip(mu * (1 - mu), 1e-9, None)
        WZ = Z * (w * s)[:, None]
        H = WZ.T @ Z
        g = Z.T @ (w * (y - mu))
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


# ---------------------------------------------------------------------------
# 4. model A / B / C orchestration
# ---------------------------------------------------------------------------
@dataclass
class Recoded:
    """Numeric arrays for one analysis sample (one outcome, one year)."""
    X: np.ndarray          # (n, p) design matrix incl. intercept
    y: np.ndarray          # (n,) binary outcome
    w: np.ndarray          # (n,) survey weight
    kind: np.ndarray       # (n,) one of 'bracket' / 'dk' / 'refused'
    log_lo: np.ndarray     # (n,) bracket lower bound on log scale (-inf ok); nan if not bracket
    log_hi: np.ndarray     # (n,) bracket upper bound on log scale (+inf ok); nan if not bracket
    log_mid: np.ndarray    # (n,) bracket-midpoint log income (model A); nan if not bracket

    def mask(self, *kinds):
        return np.isin(self.kind, kinds)


@dataclass
class ModelResult:
    name: str
    gradient_log_or_per_doubling: float
    or_per_doubling: float
    theta: float
    outcome: OutcomeFit
    income: IncomeModel | None
    n_used: int
    gamma: float | None = None

    @property
    def log_or_per_doubling(self):
        return self.gradient_log_or_per_doubling


def latent_nodes(d: Recoded, income, K, gamma=0.0, two_mechanism=False):
    """Build the latent-income quadrature nodes for model B (bracketed only) or
    model C (bracketed + don't-know MAR + refused MNAR-tilt). Returns
    ``(use_mask, nodes)`` where ``nodes`` is ``(use_mask.sum(), K)`` — exactly the
    nodes the corresponding ``fit_*`` used, so design SEs can reuse them."""
    mu_all = d.X @ income.beta
    bm = d.mask("bracket")
    if not two_mechanism:
        nodes = make_nodes(mu_all[bm], income.sigma, d.log_lo[bm], d.log_hi[bm], K)
        return bm, nodes
    dkm = d.mask("dk"); rm = d.mask("refused")
    full = np.empty((d.X.shape[0], K))
    full[bm] = make_nodes(mu_all[bm], income.sigma, d.log_lo[bm], d.log_hi[bm], K)
    full[dkm] = make_nodes(mu_all[dkm], income.sigma,
                           np.full(dkm.sum(), -np.inf), np.full(dkm.sum(), np.inf), K)
    full[rm] = make_nodes(mu_all[rm] + gamma * income.sigma ** 2, income.sigma,
                          np.full(rm.sum(), -np.inf), np.full(rm.sum(), np.inf), K)
    use = bm | dkm | rm
    return use, full[use]


def fit_A(d: Recoded):
    """Midpoint + listwise deletion: weighted logistic on the bracket midpoint."""
    m = d.mask("bracket")
    fit = fit_marginal_logit(d.X[m], d.y[m], d.w[m], d.log_mid[m][:, None])
    return ModelResult("A_midpoint_listwise", fit.log_or_per_doubling,
                       fit.or_per_doubling, fit.theta, fit, None, int(m.sum()))


def _income_for(d, m, income, income_init):
    if income is not None:
        return income
    b0, s0 = (income_init or (None, None))
    return fit_grouped_lognormal(d.X[m], d.log_lo[m], d.log_hi[m], d.w[m],
                                 beta0=b0, sigma0=s0)


def fit_B(d: Recoded, K=40, income=None, income_init=None, outcome_init=None):
    """Grouped-data likelihood, missing dropped (bracketed respondents only).

    ``income_init=(beta0, sigma0)`` and ``outcome_init`` warm-start the two
    optimisers (used by the bootstrap to converge in a few iterations)."""
    m = d.mask("bracket")
    income = _income_for(d, m, income, income_init)
    mu = d.X[m] @ income.beta
    nodes = make_nodes(mu, income.sigma, d.log_lo[m], d.log_hi[m], K)
    fit = fit_marginal_logit(d.X[m], d.y[m], d.w[m], nodes, init=outcome_init)
    return ModelResult("B_grouped_listwise", fit.log_or_per_doubling,
                       fit.or_per_doubling, fit.theta, fit, income, int(m.sum()))


def fit_C(d: Recoded, gamma=0.0, K=40, income=None, income_init=None, outcome_init=None):
    """Grouped-data likelihood + two-mechanism missingness.

    Don't-know respondents are MAR (income ~ full conditional); refused respondents
    are MNAR with an exponential tilt ``exp(gamma * I)`` on their conditional income
    distribution, i.e. mean log-income shifted by ``gamma * sigma^2``. ``gamma`` is
    the (non-identified) sensitivity parameter; ``gamma = 0`` reduces refused to MAR.
    The income measurement model is always identified from the bracketed data alone.
    """
    bm = d.mask("bracket")
    income = _income_for(d, bm, income, income_init)
    mu_all = d.X @ income.beta
    s2 = income.sigma ** 2

    nodes = np.empty((d.X.shape[0], K))
    # bracketed: truncated to their interval
    nodes[bm] = make_nodes(mu_all[bm], income.sigma, d.log_lo[bm], d.log_hi[bm], K)
    # don't-know: MAR, full conditional
    dkm = d.mask("dk")
    nodes[dkm] = make_nodes(mu_all[dkm], income.sigma,
                            np.full(dkm.sum(), -np.inf), np.full(dkm.sum(), np.inf), K)
    # refused: MNAR exponential tilt -> shifted mean
    rm = d.mask("refused")
    nodes[rm] = make_nodes(mu_all[rm] + gamma * s2, income.sigma,
                           np.full(rm.sum(), -np.inf), np.full(rm.sum(), np.inf), K)

    use = bm | dkm | rm
    fit = fit_marginal_logit(d.X[use], d.y[use], d.w[use], nodes[use], init=outcome_init)
    return ModelResult(f"C_two_mechanism(gamma={gamma:g})", fit.log_or_per_doubling,
                       fit.or_per_doubling, fit.theta, fit, income, int(use.sum()),
                       gamma=gamma)
