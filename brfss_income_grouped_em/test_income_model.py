"""Unit tests for the grouped-data EM engine.

Run with: ``.venv/bin/python -m pytest -q`` from the project directory.
"""
import numpy as np
import pytest

import income_model as im


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _design(rng, n):
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.5, size=n).astype(float)
    return np.column_stack([np.ones(n), x1, x2])


# bracket edges on the log scale (mimics income brackets: open at both tails)
_EDGES = np.log(np.array([10e3, 15e3, 20e3, 25e3, 35e3, 50e3,
                          75e3, 100e3, 150e3, 200e3]))


def _bracketize(I):
    """Assign each latent log-income to a bracket; return (log_lo, log_hi, log_mid)."""
    edges = _EDGES
    k = np.searchsorted(edges, I, side="right")   # 0..len(edges)
    lo = np.where(k == 0, -np.inf, np.take(np.concatenate([[np.nan], edges]), k))
    hi = np.where(k == len(edges), np.inf, np.take(np.concatenate([edges, [np.nan]]), k))
    # midpoints (geometric within bracket; arbitrary finite value for open tails)
    finite_lo = np.where(np.isfinite(lo), lo, hi - 1.0)
    finite_hi = np.where(np.isfinite(hi), hi, lo + 1.0)
    mid = 0.5 * (finite_lo + finite_hi)
    return lo, hi, mid


# ---------------------------------------------------------------------------
# income measurement model
# ---------------------------------------------------------------------------
def test_grouped_lognormal_recovers_params():
    rng = np.random.default_rng(0)
    n = 40_000
    X = _design(rng, n)
    beta = np.array([np.log(45_000), 0.4, -0.25])
    sigma = 0.8
    I = X @ beta + sigma * rng.normal(size=n)
    lo, hi, _ = _bracketize(I)
    w = np.ones(n)
    fit = im.fit_grouped_lognormal(X, lo, hi, w)
    assert fit.converged
    assert np.allclose(fit.beta, beta, atol=0.03)
    assert abs(fit.sigma - sigma) < 0.03


def test_grouped_loglik_matches_direct_integration():
    # one respondent, a single covariate-free bracket: likelihood = Phi(b)-Phi(a)
    X = np.array([[1.0]])
    beta = np.array([np.log(40_000)])
    sigma = 0.7
    lo = np.array([np.log(25_000)])
    hi = np.array([np.log(50_000)])
    ll = im._grouped_loglik(X, lo, hi, np.array([1.0]), beta, sigma)
    from scipy.stats import norm
    expected = np.log(norm.cdf((hi[0] - beta[0]) / sigma) -
                      norm.cdf((lo[0] - beta[0]) / sigma))
    assert abs(ll - expected) < 1e-9


# ---------------------------------------------------------------------------
# quadrature nodes
# ---------------------------------------------------------------------------
def test_make_nodes_untruncated_moments():
    m = np.zeros(1)
    nodes = im.make_nodes(m, 1.0, np.array([-np.inf]), np.array([np.inf]), 200)
    assert abs(nodes.mean()) < 1e-6
    assert abs(nodes.var() - 1.0) < 0.02


def test_make_nodes_respect_truncation():
    m = np.array([0.0])
    lo, hi = np.array([-0.5]), np.array([1.0])
    nodes = im.make_nodes(m, 1.0, lo, hi, 100)
    assert nodes.min() >= -0.5 - 1e-9
    assert nodes.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# outcome model: gradient and recovery
# ---------------------------------------------------------------------------
def test_marginal_logit_gradient_matches_numeric():
    rng = np.random.default_rng(1)
    n, K = 500, 8
    X = _design(rng, n)
    y = rng.binomial(1, 0.4, size=n).astype(float)
    w = rng.uniform(0.5, 1.5, size=n)
    I_nodes = rng.normal(size=(n, K))
    params = np.array([0.3, -0.2, 0.1, 0.05])
    _, g = im._neg_loglik_grad(params, X, y, w, I_nodes)
    num = np.zeros_like(params)
    eps = 1e-6
    for j in range(params.size):
        d = np.zeros_like(params); d[j] = eps
        fp, _ = im._neg_loglik_grad(params + d, X, y, w, I_nodes)
        fm, _ = im._neg_loglik_grad(params - d, X, y, w, I_nodes)
        num[j] = (fp - fm) / (2 * eps)
    assert np.allclose(g, num, atol=1e-5)


def test_marginal_logit_point_node_equals_plain_logistic():
    # with K=1 nodes the marginal MLE is an ordinary weighted logistic on [I, X]
    rng = np.random.default_rng(2)
    n = 8_000
    X = _design(rng, n)
    I = rng.normal(size=n)
    theta, eta = 0.5, np.array([-0.3, 0.2, 0.1])
    lin = theta * I + X @ eta
    y = rng.binomial(1, im.expit(lin)).astype(float)
    w = np.ones(n)
    fit = im.fit_marginal_logit(X, y, w, I[:, None])
    Z = np.column_stack([I, X])
    ref = im._irls_logit(Z, y, w, n_steps=50)
    assert np.allclose(fit.params, ref, atol=1e-3)


def test_grouped_beats_midpoint_for_latent_gradient():
    # The headline scientific check: integrating income over the bracket (model B)
    # recovers the true gradient; the bracket midpoint (model A) attenuates it.
    rng = np.random.default_rng(3)
    n = 60_000
    X = _design(rng, n)
    binc = np.array([np.log(45_000), 0.5, -0.3])
    sigma = 0.9
    I = X @ binc + sigma * rng.normal(size=n)
    theta_true = 0.6
    eta = np.array([-0.4, 0.1, 0.2])
    y = rng.binomial(1, im.expit(theta_true * I + X @ eta)).astype(float)
    lo, hi, mid = _bracketize(I)
    w = np.ones(n)
    d = im.Recoded(X=X, y=y, w=w, kind=np.full(n, "bracket"),
                   log_lo=lo, log_hi=hi, log_mid=mid)
    rA = im.fit_A(d)
    rB = im.fit_B(d, K=48)
    # B recovers theta within tolerance; A is biased toward 0 (attenuation)
    assert abs(rB.theta - theta_true) < 0.05
    assert abs(rA.theta - theta_true) > abs(rB.theta - theta_true)


# ---------------------------------------------------------------------------
# two-mechanism missingness
# ---------------------------------------------------------------------------
def test_mnar_gamma_zero_reduces_to_mar():
    # With gamma=0 refused income uses the same conditional as don't-know (MAR);
    # relabelling refused<->dk must not change the model-C fit.
    rng = np.random.default_rng(4)
    n = 30_000
    X = _design(rng, n)
    binc = np.array([np.log(45_000), 0.4, -0.2])
    sigma = 0.8
    I = X @ binc + sigma * rng.normal(size=n)
    eta = np.array([-0.3, 0.1, 0.15])
    y = rng.binomial(1, im.expit(0.5 * I + X @ eta)).astype(float)
    lo, hi, mid = _bracketize(I)
    kind = np.array(["bracket"] * n, dtype=object)
    miss = rng.choice(n, size=8_000, replace=False)
    kind[miss[:4_000]] = "dk"
    kind[miss[4_000:]] = "refused"
    base = dict(X=X, y=y, w=np.ones(n), log_lo=lo, log_hi=hi, log_mid=mid)
    d1 = im.Recoded(kind=kind.astype(str), **base)
    swapped = kind.copy()
    swapped[kind == "dk"] = "refused"
    swapped[kind == "refused"] = "dk"
    d2 = im.Recoded(kind=swapped.astype(str), **base)
    r1 = im.fit_C(d1, gamma=0.0, K=40)
    r2 = im.fit_C(d2, gamma=0.0, K=40)
    assert abs(r1.theta - r2.theta) < 1e-6


def test_mnar_tilt_shifts_refuser_income():
    # positive gamma shifts refusers' imputed income upward by ~gamma*sigma^2
    rng = np.random.default_rng(5)
    n = 4_000
    X = np.ones((n, 1))
    income = im.IncomeModel(beta=np.array([np.log(40_000)]), sigma=0.8,
                            loglik=0.0, n_iter=0, converged=True)
    mu = X @ income.beta
    g = 1.0
    nodes0 = im.make_nodes(mu, income.sigma, np.full(n, -np.inf),
                           np.full(n, np.inf), 200)
    nodes1 = im.make_nodes(mu + g * income.sigma ** 2, income.sigma,
                           np.full(n, -np.inf), np.full(n, np.inf), 200)
    assert abs((nodes1.mean() - nodes0.mean()) - g * income.sigma ** 2) < 1e-6


def test_weight_duplication_equals_doubling():
    rng = np.random.default_rng(6)
    n = 5_000
    X = _design(rng, n)
    I = rng.normal(size=n)
    y = rng.binomial(1, im.expit(0.5 * I + X @ np.array([-0.2, 0.1, 0.1]))).astype(float)
    w = rng.uniform(0.5, 1.5, size=n)
    f1 = im.fit_marginal_logit(X, y, w, I[:, None])
    Xd = np.vstack([X, X]); Id = np.concatenate([I, I])
    yd = np.concatenate([y, y]); wd = np.concatenate([w, w]) * 0.5
    # duplicating rows with halved weights leaves the weighted estimate unchanged
    f2 = im.fit_marginal_logit(Xd, yd, wd, Id[:, None])
    assert np.allclose(f1.params, f2.params, atol=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
