"""
Taylor linearization utilities for complex survey data (stratified cluster design).

Implements the standard variance estimator:
  V(T_hat) = sum_h [ n_h/(n_h-1) * sum_i (z_hi - z_bar_h)^2 ]

where z_hi is the PSU-level sum of (weight * linearized_variable).

For means:     z_hi = sum_j w_j * (y_j - Y_hat/N_hat)  [ratio estimator]
For totals:    z_hi = sum_j w_j * y_j
For GLMs:      z_hi = sum_j w_j * x_j * e_j  (score residuals)

Reference: Binder (1983), Skinner et al. (1989), survey package vignette.
"""

import numpy as np
import pandas as pd
from scipy import stats


def _psu_variance(df: pd.DataFrame, z_col: str,
                  stratum_col: str, psu_col: str) -> float:
    """
    Compute sum_h [n_h/(n_h-1) * sum_i (z_hi - z_bar_h)^2] over all strata.

    df must have columns z_col (PSU-level scores), stratum_col, psu_col.
    Returns the scalar variance estimate.
    """
    # Already PSU-level if called with psu_totals
    grouped = df.groupby(stratum_col)[z_col]
    var_total = 0.0
    for stratum, vals in grouped:
        n_h = len(vals)
        if n_h < 2:
            # Singleton PSU: contribute 0 variance (conservative)
            continue
        z_bar = vals.mean()
        var_total += (n_h / (n_h - 1)) * ((vals - z_bar) ** 2).sum()
    return var_total


def _df_approx(df: pd.DataFrame, stratum_col: str) -> float:
    """Degrees of freedom = sum_h (n_h - 1)."""
    n_psus = df.groupby(stratum_col).size()
    return float((n_psus - 1).sum())


def survey_total(df: pd.DataFrame, y: str, wgt: str,
                 stratum: str, psu: str,
                 alpha: float = 0.05) -> dict:
    """Survey-weighted total with Taylor CI."""
    T_hat = (df[wgt] * df[y]).sum()

    # PSU-level weighted sums
    psu_sums = (df.assign(_wy=df[wgt] * df[y])
                  .groupby([stratum, psu])['_wy'].sum()
                  .reset_index(name='z'))

    V = _psu_variance(psu_sums, 'z', stratum, psu)
    SE = np.sqrt(V)
    df_approx = _df_approx(psu_sums, stratum)
    t_crit = stats.t.ppf(1 - alpha / 2, df=df_approx)
    return {'est': T_hat, 'se': SE, 'lci': T_hat - t_crit * SE,
            'uci': T_hat + t_crit * SE, 'df': df_approx}


def survey_mean(df: pd.DataFrame, y: str, wgt: str,
                stratum: str, psu: str,
                alpha: float = 0.05) -> dict:
    """Survey-weighted mean with Taylor CI (ratio estimator)."""
    W = df[wgt].sum()
    Y = (df[wgt] * df[y]).sum()
    Y_bar = Y / W

    # Linearized variable: g_j = w_j * (y_j - Y_bar) / W
    df = df.copy()
    df['_g'] = df[wgt] * (df[y] - Y_bar) / W

    psu_sums = (df.groupby([stratum, psu])['_g'].sum()
                  .reset_index(name='z'))
    V = _psu_variance(psu_sums, 'z', stratum, psu)
    SE = np.sqrt(V)
    df_approx = _df_approx(psu_sums, stratum)
    t_crit = stats.t.ppf(1 - alpha / 2, df=df_approx)
    return {'est': Y_bar, 'se': SE, 'lci': Y_bar - t_crit * SE,
            'uci': Y_bar + t_crit * SE, 'df': df_approx}


def survey_proportion(df: pd.DataFrame, y: str, wgt: str,
                      stratum: str, psu: str,
                      alpha: float = 0.05) -> dict:
    """Survey-weighted proportion (equivalent to survey_mean for 0/1 y)."""
    return survey_mean(df, y, wgt, stratum, psu, alpha)


def survey_count(df: pd.DataFrame, y: str, wgt: str,
                 stratum: str, psu: str,
                 alpha: float = 0.05) -> dict:
    """
    Survey-weighted count of persons where y == 1.
    Equivalent to survey_total(df, y, ...) since total of w*y is the count.
    """
    return survey_total(df, y, wgt, stratum, psu, alpha)


def survey_glm_logit(df: pd.DataFrame, y: str, x_cols: list, wgt: str,
                     stratum: str, psu: str,
                     alpha: float = 0.05) -> pd.DataFrame:
    """
    Survey-weighted logistic regression with Taylor linearization SEs.

    Implements the linearization (score-based) sandwich estimator:
      V(beta) = A^{-1} B A^{-1}
    where
      A = X'W diag(pi*(1-pi)) X   (Hessian of weighted log-likelihood)
      B = sum_h n_h/(n_h-1) * sum_i (s_hi - s_bar_h)(s_hi - s_bar_h)'
      s_hi = sum_j w_j * x_j * (y_j - pi_j)   (score residuals per PSU)

    Returns DataFrame with columns: variable, coef, se, z, p, lci, uci, or_lci, or_uci.
    """
    from scipy.special import expit
    from scipy.optimize import minimize

    sub = df[[y] + x_cols + [wgt, stratum, psu]].dropna().copy()
    Y = sub[y].values.astype(float)
    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in x_cols])
    # Normalize weights to sum to N (stabilizes matrix condition numbers)
    # Point estimates and ORs are unchanged; only scaling of variance matrices.
    W_raw = sub[wgt].values.astype(float)
    W = W_raw / W_raw.mean()
    S = sub[stratum].values
    P = sub[psu].values

    # Weighted log-likelihood for logistic regression
    def neg_log_lik(beta):
        eta = X @ beta
        pi = expit(eta)
        pi = np.clip(pi, 1e-12, 1 - 1e-12)
        return -np.sum(W * (Y * np.log(pi) + (1 - Y) * np.log(1 - pi)))

    def neg_grad(beta):
        eta = X @ beta
        pi = expit(eta)
        return -(X.T @ (W * (Y - pi)))

    beta0 = np.zeros(X.shape[1])
    res = minimize(neg_log_lik, beta0, jac=neg_grad, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    if not res.success:
        print(f'WARNING: optimization did not converge: {res.message}')
    beta_hat = res.x

    # Hessian A = X'W diag(pi*(1-pi)) X
    pi_hat = expit(X @ beta_hat)
    v = pi_hat * (1 - pi_hat)
    A = (X * (W * v)[:, None]).T @ X

    # Score residuals per person: w_j * x_j * (y_j - pi_j)
    score_j = (W * (Y - pi_hat))[:, None] * X   # shape (n, p)

    # PSU-level score sums
    sub_scores = pd.DataFrame(score_j, columns=[f's{i}' for i in range(X.shape[1])])
    sub_scores[stratum] = S
    sub_scores[psu] = P
    s_cols = [f's{i}' for i in range(X.shape[1])]

    psu_scores = sub_scores.groupby([stratum, psu])[s_cols].sum().reset_index()

    # Sandwich B = sum_h n_h/(n_h-1) sum_i (s_hi - s_bar_h)(s_hi - s_bar_h)'
    p = X.shape[1]
    B = np.zeros((p, p))
    for h, grp in psu_scores.groupby(stratum):
        n_h = len(grp)
        if n_h < 2:
            continue
        Z_h = grp[s_cols].values  # (n_h, p)
        Z_bar = Z_h.mean(axis=0)
        devs = Z_h - Z_bar
        B += (n_h / (n_h - 1)) * devs.T @ devs

    A_inv = np.linalg.inv(A)
    V_beta = A_inv @ B @ A_inv
    se_beta = np.sqrt(np.diag(V_beta))

    # Degrees of freedom: sum_h (n_h - 1) - (p-1) for t-test
    n_psus = psu_scores.groupby(stratum).size()
    df_res = float((n_psus - 1).sum()) - (p - 1)

    t_crit = stats.t.ppf(1 - alpha / 2, df=df_res)
    z_stat = beta_hat / se_beta
    p_vals = 2 * stats.t.sf(np.abs(z_stat), df=df_res)

    col_names = ['Intercept'] + x_cols
    results = pd.DataFrame({
        'variable': col_names,
        'coef': beta_hat,
        'se': se_beta,
        'z': z_stat,
        'p': p_vals,
        'lci': beta_hat - t_crit * se_beta,
        'uci': beta_hat + t_crit * se_beta,
        'or': np.exp(beta_hat),
        'or_lci': np.exp(beta_hat - t_crit * se_beta),
        'or_uci': np.exp(beta_hat + t_crit * se_beta),
    })
    return results
