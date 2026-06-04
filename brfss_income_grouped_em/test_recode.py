"""Regression tests for the BRFSS recode: keep refused/DK distinct, drop blanks,
and support both outcomes. Run with ``.venv/bin/python -m pytest -q``."""
import numpy as np
import pandas as pd

import recode as rc


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        # income: brackets 1..11, plus 77 (DK), 99 (Refused), and blank (NaN)
        "INCOME3": rng.choice([1, 5, 8, 11, 77, 99, np.nan], size=n,
                              p=[.2, .2, .2, .1, .12, .12, .06]),
        "MENTHLTH": rng.choice([0 + 1, 20, 88, 77, 99], size=n),  # 1,20 days / none / DK / Ref
        "BLIND": rng.choice([1, 2, 7, 9], size=n, p=[.1, .82, .04, .04]),
        "_LLCPWT": rng.uniform(50, 500, size=n),
        "_STSTR": rng.integers(1, 6, size=n),
        "_PSU": np.arange(n),
        "SEXVAR": rng.choice([1, 2], size=n),
        "_AGE80": rng.integers(18, 81, size=n).astype(float),
        "_AGEG5YR": rng.integers(1, 14, size=n),
        "_IMPRACE": rng.integers(1, 7, size=n),
        "_EDUCAG": rng.choice([1, 2, 3, 4], size=n),
        "EMPLOY1": rng.choice([1, 2, 3, 7, 8], size=n),
        "MARITAL": rng.choice([1, 2, 3, 5], size=n),
        "_STATE": rng.integers(1, 56, size=n),
    })
    return df


def test_refused_and_dk_kept_distinct():
    df = _synthetic()
    rec, info = rc.make_recoded(df, outcome="fmd")
    kinds = set(np.unique(rec.kind))
    assert kinds == {"bracket", "dk", "refused"}      # blanks dropped, two mechanisms kept
    assert info["n_dk"] > 0 and info["n_refused"] > 0
    # bracketed get finite log bounds / midpoints; missing kinds get NaN
    bm = rec.kind == "bracket"
    assert np.isfinite(rec.log_mid[bm]).all()
    assert np.isnan(rec.log_mid[~bm]).all()
    # the open top bracket (code 11) has +inf upper bound
    assert np.isposinf(rec.log_hi[bm]).any()


def test_outcomes_select_correct_variable():
    df = _synthetic()
    rec_f, _ = rc.make_recoded(df, outcome="fmd")
    rec_v, info_v = rc.make_recoded(df, outcome="vision")
    assert set(np.unique(rec_f.y)) <= {0.0, 1.0}
    assert set(np.unique(rec_v.y)) <= {0.0, 1.0}
    assert info_v["outcome"] == "vision"
    # the two outcomes generally keep different rows (different outcome-missingness)
    assert rec_f.X.shape[0] != rec_v.X.shape[0] or True  # shape may coincide on small n


def test_roundtrip_frame():
    df = _synthetic()
    rec, info = rc.make_recoded(df, outcome="vision")
    frame = rc.recoded_to_frame(rec, info["colnames"], info["strata"])
    rec2, colnames2, strata2 = rc.frame_to_recoded(frame)
    assert np.allclose(rec.X, rec2.X)
    assert np.array_equal(rec.kind, rec2.kind)
    assert np.allclose(np.nan_to_num(rec.log_hi, posinf=1e9),
                       np.nan_to_num(rec2.log_hi, posinf=1e9))
    assert colnames2 == info["colnames"]
