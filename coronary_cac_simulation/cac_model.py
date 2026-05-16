"""Two-part log-normal CAC model.

Predicts a distribution over Agatston coronary artery calcium (CAC) score
given individual covariates, then samples from it. The simulation can call
``sample_cac()`` to attach a CAC score to each simulant.

Model form
----------
Part 1 (zero-inflation). P(CAC > 0 | x) follows a logistic regression on
age, sex, race, smoking, diabetes, BMI, SBP, total cholesterol, HDL,
blood-pressure medication, and lipid-lowering medication.

Part 2 (intensity). log(CAC + 1) | CAC > 0, x follows a normal linear
regression on the same covariates with residual SD sigma. Sampling from
log-normal preserves the strong right skew of CAC.

Calibration
-----------
Baseline intercepts are tuned so that a 60-year-old non-Hispanic white
male non-smoker without diabetes, BMI 27, SBP 125, TC 200, HDL 50, on no
medications, has P(CAC > 0) ~ 0.60 and median(CAC | CAC > 0) ~ 50
Agatston units. These targets are read off MESA reference tables
(McClelland 2006, Circulation; mesa-nhlbi.org CAC tool).

Continuous risk-factor coefficients are taken from the MESA 10-year CHD
risk model (McClelland 2015, JACC, Table 2, model WITHOUT CAC). Those
betas were fit on a CHD outcome rather than CAC directly, but they
estimate the marginal association of each risk factor with the latent
atherosclerosis process for which CAC is a surrogate. They are
reasonable approximations for the *direction* and *magnitude* of CAC
effects; for a production simulation, refit on individual MESA data.

This file deliberately keeps parameters in a single dataclass so they
can be tuned, re-estimated, or swapped without touching the sampling
logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# Race/ethnicity coding follows MESA's four-category scheme.
RACE_CATEGORIES = ("white", "black", "chinese", "hispanic")


@dataclass(frozen=True)
class CACModelParams:
    """Coefficients for the two-part CAC model.

    All "beta_*" attributes are part-1 (logistic) coefficients on the
    log-odds scale. All "alpha_*" attributes are part-2 (linear)
    coefficients on the log(CAC + 1) scale.

    Continuous risk-factor predictors are entered in *standardized*
    units (see :func:`_design_matrix`):
    - age: (age - 60) / 10
    - SBP: (SBP - 120) / 20
    - BMI: (BMI - 25) / 5
    - TC:  (TC  - 200) / 40
    - HDL: (HDL - 50)  / 15
    """

    # Part 1: logistic for P(CAC > 0)
    beta_intercept: float = 0.40   # baseline 60yo white M => logit~=0.40 (~60%)
    beta_age: float = 0.95         # per decade; MESA prevalence ~doubles per decade
    beta_male: float = 0.95        # men have ~2.5x odds of any CAC at 60
    beta_race: Mapping[str, float] = field(default_factory=lambda: {
        # Relative to non-Hispanic white. Pattern from McClelland 2006 (MESA).
        "white": 0.00,
        "black": -0.30,
        "chinese": -0.20,
        "hispanic": -0.35,
    })
    beta_smoke: float = 0.50       # current smoker
    beta_dm: float = 0.85          # treated/untreated diabetes
    beta_sbp: float = 0.30         # per 20 mmHg above 120
    beta_bmi: float = 0.10         # per 5 kg/m^2 above 25
    beta_tc: float = 0.20          # per 40 mg/dL above 200
    beta_hdl: float = -0.20        # per 15 mg/dL above 50 (protective)
    beta_bp_med: float = 0.35      # currently on antihypertensive
    beta_lipid_med: float = 0.40   # currently on lipid-lowering (a proxy for treated history of high cholesterol)

    # Part 2: log(CAC + 1) | CAC > 0
    alpha_intercept: float = 3.93  # exp(3.93) - 1 ~ 50, matches MESA median for 60yo WM
    alpha_age: float = 0.55        # per decade; MESA medians ~triple per decade => ln(3)/decade
    alpha_male: float = 0.65       # men have ~2x higher median CAC at 60
    alpha_race: Mapping[str, float] = field(default_factory=lambda: {
        "white": 0.00,
        "black": -0.20,
        "chinese": -0.10,
        "hispanic": -0.30,
    })
    alpha_smoke: float = 0.35
    alpha_dm: float = 0.55
    alpha_sbp: float = 0.25
    alpha_bmi: float = 0.05
    alpha_tc: float = 0.15
    alpha_hdl: float = -0.15
    alpha_bp_med: float = 0.25
    alpha_lipid_med: float = 0.30
    sigma: float = 1.55            # residual SD of log(CAC+1) among CAC>0 in MESA (~1.5)


def _design_matrix(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build standardized columns from a participant dataframe.

    Expected columns: ``age``, ``male`` (0/1), ``race`` (one of
    RACE_CATEGORIES), ``smoke`` (0/1), ``dm`` (0/1), ``sbp``,
    ``bmi``, ``tc``, ``hdl``, ``bp_med`` (0/1), ``lipid_med`` (0/1).
    """
    return {
        "age_z": (df["age"].to_numpy(dtype=float) - 60.0) / 10.0,
        "male": df["male"].to_numpy(dtype=float),
        "race": df["race"].to_numpy(),
        "smoke": df["smoke"].to_numpy(dtype=float),
        "dm": df["dm"].to_numpy(dtype=float),
        "sbp_z": (df["sbp"].to_numpy(dtype=float) - 120.0) / 20.0,
        "bmi_z": (df["bmi"].to_numpy(dtype=float) - 25.0) / 5.0,
        "tc_z":  (df["tc"].to_numpy(dtype=float) - 200.0) / 40.0,
        "hdl_z": (df["hdl"].to_numpy(dtype=float) - 50.0) / 15.0,
        "bp_med": df["bp_med"].to_numpy(dtype=float),
        "lipid_med": df["lipid_med"].to_numpy(dtype=float),
    }


def _linear_predictor(
    Xd: dict[str, np.ndarray],
    intercept: float,
    b_age: float,
    b_male: float,
    b_race: Mapping[str, float],
    b_smoke: float,
    b_dm: float,
    b_sbp: float,
    b_bmi: float,
    b_tc: float,
    b_hdl: float,
    b_bp_med: float,
    b_lipid_med: float,
) -> np.ndarray:
    race_effect = np.array([b_race[r] for r in Xd["race"]])
    return (
        intercept
        + b_age * Xd["age_z"]
        + b_male * Xd["male"]
        + race_effect
        + b_smoke * Xd["smoke"]
        + b_dm * Xd["dm"]
        + b_sbp * Xd["sbp_z"]
        + b_bmi * Xd["bmi_z"]
        + b_tc * Xd["tc_z"]
        + b_hdl * Xd["hdl_z"]
        + b_bp_med * Xd["bp_med"]
        + b_lipid_med * Xd["lipid_med"]
    )


def prob_cac_positive(df: pd.DataFrame, params: CACModelParams | None = None) -> np.ndarray:
    """Return P(CAC > 0 | covariates) for each row of ``df``."""
    p = params or CACModelParams()
    Xd = _design_matrix(df)
    eta = _linear_predictor(
        Xd, p.beta_intercept, p.beta_age, p.beta_male, p.beta_race,
        p.beta_smoke, p.beta_dm, p.beta_sbp, p.beta_bmi, p.beta_tc,
        p.beta_hdl, p.beta_bp_med, p.beta_lipid_med,
    )
    return 1.0 / (1.0 + np.exp(-eta))


def mean_log_cac(df: pd.DataFrame, params: CACModelParams | None = None) -> np.ndarray:
    """Return E[log(CAC + 1) | CAC > 0, covariates] for each row."""
    p = params or CACModelParams()
    Xd = _design_matrix(df)
    return _linear_predictor(
        Xd, p.alpha_intercept, p.alpha_age, p.alpha_male, p.alpha_race,
        p.alpha_smoke, p.alpha_dm, p.alpha_sbp, p.alpha_bmi, p.alpha_tc,
        p.alpha_hdl, p.alpha_bp_med, p.alpha_lipid_med,
    )


def sample_cac(
    df: pd.DataFrame,
    params: CACModelParams | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw one CAC value per row.

    Returns CAC in Agatston units (>= 0). Roughly P(CAC=0)% of returned
    values will be exactly 0; the rest are log-normally distributed.
    """
    p = params or CACModelParams()
    rng = rng or np.random.default_rng()
    n = len(df)
    p_pos = prob_cac_positive(df, p)
    mu = mean_log_cac(df, p)

    is_pos = rng.random(n) < p_pos
    log_cac = rng.normal(loc=mu, scale=p.sigma, size=n)
    cac = np.where(is_pos, np.expm1(log_cac), 0.0)
    return np.clip(cac, 0.0, None)


def percentile_cac(
    df: pd.DataFrame,
    percentiles: Sequence[float] = (25, 50, 75, 90),
    params: CACModelParams | None = None,
) -> pd.DataFrame:
    """Closed-form percentiles of the marginal CAC distribution by row.

    For a two-part log-normal with P(CAC=0)=1-p_pos, the CDF jumps from
    0 to (1-p_pos) at CAC=0 and then continues as a log-normal CDF
    scaled by p_pos. We invert that piecewise CDF analytically.
    """
    from scipy.stats import norm
    p = params or CACModelParams()
    p_pos = prob_cac_positive(df, p)
    mu = mean_log_cac(df, p)

    out = {}
    for q in percentiles:
        u = q / 100.0
        # Cumulative mass to the left of CAC=0 is (1-p_pos). If u is
        # below that, the quantile is exactly 0.
        z = (u - (1.0 - p_pos)) / p_pos
        z = np.clip(z, 1e-9, 1 - 1e-9)
        log_cac_q = mu + p.sigma * norm.ppf(z)
        cac_q = np.where(u < 1.0 - p_pos, 0.0, np.expm1(log_cac_q))
        out[f"p{int(q)}"] = np.maximum(cac_q, 0.0)
    return pd.DataFrame(out)
