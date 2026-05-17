"""CAC -> obstructive coronary lesion probability.

Maps a sampled CAC score (Agatston units) to the probability that an
individual has an obstructive (>=50% stenosis) lesion on coronary
angiography. This is the second step of the two-step pipeline:
risk factors -> CAC -> lesion.

Calibration anchors come from the CONFIRM registry and Budoff JACC
2007 ("Long-term prognosis associated with CAC"), which report
prevalence of obstructive CAD by CAC category in symptomatic and
mixed-symptom populations:

    CAC = 0           ~5%  obstructive CAD
    CAC = 1-99        ~12%
    CAC = 100-399     ~28%
    CAC >= 400        ~55%

These were measured in patients referred for evaluation, so the
absolute level overstates risk in an unselected general population.
A simulation that needs to apply this to a general-population cohort
should additionally condition on symptoms / clinical referral
status, or scale ``offset`` down to match the general-population
prevalence target.

Functional form: logistic on log(CAC + 1).
    logit(P(lesion | CAC)) = offset + slope * log(CAC + 1)

``offset`` and ``slope`` below are fit by least squares to the four
anchor probabilities at the geometric centers of each bin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LesionModelParams:
    """Logistic-on-log(CAC+1) parameters."""
    offset: float = -3.027  # logit(P) at CAC=0; fit by fit_default_params()
    slope: float = 0.447    # change in logit per +1 in log(CAC+1)


# Anchor values used to fit the default parameters above. Useful for
# refitting if you want to retarget at general-population prevalence.
CONFIRM_ANCHORS = {
    # Geometric center of each CAC bin -> reported obstructive-CAD prevalence.
    0.0:   0.05,
    10.0:  0.12,
    200.0: 0.28,
    800.0: 0.55,
}


def prob_obstructive_lesion(
    cac: np.ndarray,
    params: LesionModelParams | None = None,
) -> np.ndarray:
    """Return P(obstructive lesion | CAC) for each CAC value."""
    p = params or LesionModelParams()
    eta = p.offset + p.slope * np.log1p(cac)
    return 1.0 / (1.0 + np.exp(-eta))


def sample_obstructive_lesion(
    cac: np.ndarray,
    params: LesionModelParams | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bernoulli-sample one obstructive-lesion indicator per CAC value."""
    rng = rng or np.random.default_rng()
    return (rng.random(len(cac)) < prob_obstructive_lesion(cac, params)).astype(int)


def fit_default_params() -> LesionModelParams:
    """Recompute (offset, slope) by least squares on CONFIRM_ANCHORS.

    Run once to regenerate the dataclass defaults; not needed at
    sample time.
    """
    from scipy.optimize import curve_fit
    cac = np.array(list(CONFIRM_ANCHORS.keys()), dtype=float)
    p = np.array(list(CONFIRM_ANCHORS.values()), dtype=float)
    logit_p = np.log(p / (1 - p))

    def model(x, offset, slope):
        return offset + slope * np.log1p(x)

    (offset, slope), _ = curve_fit(model, cac, logit_p, p0=(-3.0, 0.6))
    return LesionModelParams(offset=float(offset), slope=float(slope))
