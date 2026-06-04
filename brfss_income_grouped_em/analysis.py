"""Reusable A/B/C estimation and MNAR sweep on a recoded BRFSS sample.

Factors the logic validated in notebook 03/04 so any outcome (FMD, vision, ...) runs
the identical pipeline. All CIs are design-based: a stratified Taylor-linearised
sandwich (primary) with an optional stratified bootstrap for confirmation.
"""
import numpy as np
import pandas as pd

import income_model as im
import survey as sv


def fit_all(d, K=32):
    """Fit models A, B, C(γ=0) on a recoded sample; returns (rA, rB, rC)."""
    rA = im.fit_A(d)
    rB = im.fit_B(d, K=K)
    rC = im.fit_C(d, gamma=0.0, K=K, income=rB.income)
    return rA, rB, rC


def linearized_ci(d, strata, res, K, two_mech):
    """Design-based linearized 95% CI (log-OR-per-doubling scale) for one model."""
    if res.name.startswith("A"):
        m = d.mask("bracket")
        nodes = d.log_mid[m][:, None]
        X, y, w, st = d.X[m], d.y[m], d.w[m], strata[m]
    else:
        use, nodes = im.latent_nodes(d, res.income, K, gamma=(res.gamma or 0.0),
                                     two_mechanism=two_mech)
        X, y, w, st = d.X[use], d.y[use], d.w[use], strata[use]
    se = sv.linearized_se(res.outcome.params, X, y, w, nodes, st) * im.LN2
    return sv.CI(res.log_or_per_doubling, se, res.log_or_per_doubling - 1.96 * se,
                 res.log_or_per_doubling + 1.96 * se, np.array([]))


def abc_table(d, strata, results, K):
    """Build the A/B/C comparison table with linearized CIs; returns (df, lin_dict)."""
    rA, rB, rC = results
    lin = {"A": linearized_ci(d, strata, rA, K, False),
           "B": linearized_ci(d, strata, rB, K, False),
           "C": linearized_ci(d, strata, rC, K, True)}
    rows = []
    for label, res, key in [("A. midpoint + listwise", rA, "A"),
                            ("B. grouped + listwise", rB, "B"),
                            ("C. grouped + two-mech (γ=0)", rC, "C")]:
        ci = lin[key]
        rows.append({"model": label,
                     "OR per doubling": round(np.exp(res.log_or_per_doubling), 4),
                     "95% CI low": round(np.exp(ci.lo), 4),
                     "95% CI high": round(np.exp(ci.hi), 4),
                     "log-OR/doubling": round(res.log_or_per_doubling, 4),
                     "lin SE": round(ci.se, 4), "n used": res.n_used})
    return pd.DataFrame(rows), lin


def mnar_sweep(d, strata, income, K, deltas):
    """Sweep refusers' assumed mean-log-income shift Δ (= γσ²); income model fixed."""
    sigma2 = income.sigma ** 2
    rows = []
    for delta in deltas:
        g = delta / sigma2
        rC = im.fit_C(d, gamma=g, K=K, income=income)
        use, nodes = im.latent_nodes(d, income, K, gamma=g, two_mechanism=True)
        se = sv.linearized_se(rC.outcome.params, d.X[use], d.y[use], d.w[use],
                              nodes, strata[use]) * im.LN2
        rows.append({"delta": delta, "income_ratio": np.exp(delta), "gamma": g,
                     "log_or": rC.log_or_per_doubling, "se": se,
                     "or": rC.or_per_doubling})
    sweep = pd.DataFrame(rows)
    sweep["or_lo"] = np.exp(sweep["log_or"] - 1.96 * sweep["se"])
    sweep["or_hi"] = np.exp(sweep["log_or"] + 1.96 * sweep["se"])
    return sweep


def bootstrap_confirm(d, strata, results, B, K_boot, seed=2023):
    """Warm-started stratified bootstrap of the gradient for A/B/C (confirmation)."""
    rA, rB, rC = results
    fitters = {
        "A": lambda dd: im.fit_A(dd),
        "B": lambda dd: im.fit_B(dd, K=K_boot,
                                 income_init=(rB.income.beta, rB.income.sigma),
                                 outcome_init=rB.outcome.params),
        "C": lambda dd: im.fit_C(dd, gamma=0.0, K=K_boot,
                                 income_init=(rC.income.beta, rC.income.sigma),
                                 outcome_init=rC.outcome.params),
    }
    points = {"A": rA.log_or_per_doubling, "B": rB.log_or_per_doubling,
              "C": rC.log_or_per_doubling}
    return sv.bootstrap_gradients(d, strata, fitters, B=B, seed=seed, points=points)
