"""BRFSS 2023 recode: raw XPT columns -> analysis arrays.

Every coding here was verified against the official 2023 codebook
(``USCODE23_LLCP_021924.HTML``); see ``METHODS.md`` for the references. The cardinal
rule: ``INCOME3`` Refused (99) and Don't know/Not sure (77) are kept as DISTINCT
categories end to end -- they are never merged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from income_model import Recoded

# --- columns we pull from the XPT -------------------------------------------
COLUMNS = [
    "INCOME3", "MENTHLTH", "GENHLTH", "BLIND",
    "_LLCPWT", "_STSTR", "_PSU",
    "SEXVAR", "_AGE80", "_AGEG5YR", "_IMPRACE", "_EDUCAG", "EMPLOY1", "MARITAL",
    "_STATE",
]

# --- INCOME3 brackets (code -> dollar interval), verified ---------------------
# 1..11 are real brackets; 77 = Don't know/Not sure; 99 = Refused; blank = not asked.
INCOME_BRACKETS_USD = {
    1: (0, 10_000),
    2: (10_000, 15_000),
    3: (15_000, 20_000),
    4: (20_000, 25_000),
    5: (25_000, 35_000),
    6: (35_000, 50_000),
    7: (50_000, 75_000),
    8: (75_000, 100_000),
    9: (100_000, 150_000),
    10: (150_000, 200_000),
    11: (200_000, np.inf),   # open-ended top bracket
}
INCOME_DK = 77        # Don't know / Not sure
INCOME_REFUSED = 99   # Refused
# default finite value assigned to the open top bracket for the *midpoint* model A
TOP_BRACKET_MIDPOINT_USD = 250_000
# midpoint for the bottom bracket (0, 10k]
BOTTOM_BRACKET_MIDPOINT_USD = 5_000


def income_log_bounds():
    """Return dict code -> (log_lo, log_hi) on the natural-log income scale."""
    out = {}
    for code, (lo, hi) in INCOME_BRACKETS_USD.items():
        log_lo = -np.inf if lo <= 0 else np.log(lo)
        log_hi = np.inf if not np.isfinite(hi) else np.log(hi)
        out[code] = (log_lo, log_hi)
    return out


def income_log_midpoint(top_mid_usd=TOP_BRACKET_MIDPOINT_USD):
    """Return dict code -> log(midpoint dollars) for the midpoint model A."""
    out = {}
    for code, (lo, hi) in INCOME_BRACKETS_USD.items():
        if code == 1:
            mid = BOTTOM_BRACKET_MIDPOINT_USD
        elif not np.isfinite(hi):
            mid = top_mid_usd
        else:
            mid = 0.5 * (lo + hi)
        out[code] = np.log(mid)
    return out


# ---------------------------------------------------------------------------
# outcome
# ---------------------------------------------------------------------------
def recode_menthlth(s):
    """MENTHLTH -> (days, valid). 1-30 = days; 88 = 0 days; 77/99/blank invalid."""
    s = pd.to_numeric(s, errors="coerce")
    days = np.full(len(s), np.nan)
    days = np.where((s >= 1) & (s <= 30), s, days)
    days = np.where(s == 88, 0.0, days)
    valid = np.isfinite(days)
    return days, valid


def fmd_from_days(days):
    """Frequent mental distress: >= 14 of the past 30 days."""
    return (days >= 14).astype(float)


# Binary outcomes. Each returns (y, valid) aligned to df rows; invalid rows
# (Don't know / Refused / not asked on the outcome) are dropped downstream.
def _outcome_fmd(df):
    days, valid = recode_menthlth(df["MENTHLTH"])
    return fmd_from_days(days), valid


def _outcome_vision(df):
    """BLIND: 1 = blind / serious difficulty seeing even with glasses, 2 = no;
    7/9/blank invalid."""
    v = pd.to_numeric(df["BLIND"], errors="coerce")
    y = np.where(v == 1, 1.0, np.where(v == 2, 0.0, np.nan))
    return y, np.isfinite(y)


OUTCOMES = {
    "fmd": (_outcome_fmd, "frequent mental distress (MENTHLTH >= 14 days)"),
    "vision": (_outcome_vision, "serious difficulty seeing, even with glasses (BLIND)"),
}


# ---------------------------------------------------------------------------
# covariates -> design matrix
# ---------------------------------------------------------------------------
_RACE_LABELS = {1: "White_NH", 2: "Black_NH", 3: "Asian_NH",
                4: "AIAN_NH", 5: "Hispanic", 6: "Other_NH"}
_EDUC_LABELS = {1: "LtHS", 2: "HSgrad", 3: "SomeColl", 4: "CollGrad"}


def _employ_group(v):
    # EMPLOY1: 1 wages, 2 self, 3 oow>=1yr, 4 oow<1yr, 5 homemaker, 6 student,
    # 7 retired, 8 unable, 9 refused/blank
    mapping = {1: "Employed", 2: "Employed", 3: "Unemployed", 4: "Unemployed",
               5: "Homemaker_Student", 6: "Homemaker_Student",
               7: "Retired", 8: "Unable"}
    return mapping.get(v, None)


def _marital_group(v):
    mapping = {1: "Married", 2: "Div_Sep", 4: "Div_Sep", 3: "Widowed",
               5: "NeverMarried", 6: "UnmarriedCouple"}
    return mapping.get(v, None)


def build_design(df, ref_race="White_NH", ref_educ="CollGrad",
                 ref_employ="Employed", ref_marital="Married"):
    """Build the design matrix (intercept + covariates) used by both the income
    and outcome models. Returns (X, colnames, valid_mask, aux) where ``valid_mask``
    flags rows with all covariates present and ``aux`` is a tidy DataFrame of the
    recoded categorical covariates for diagnostics.
    """
    n = len(df)
    age = pd.to_numeric(df["_AGE80"], errors="coerce").to_numpy(float)
    age_ok = np.isfinite(age) & (age >= 18) & (age <= 80)

    sex = pd.to_numeric(df["SEXVAR"], errors="coerce")
    female = (sex == 2).to_numpy(float)
    sex_ok = sex.isin([1, 2]).to_numpy()

    race = pd.to_numeric(df["_IMPRACE"], errors="coerce").map(_RACE_LABELS)
    educ = pd.to_numeric(df["_EDUCAG"], errors="coerce").map(_EDUC_LABELS)
    employ = pd.to_numeric(df["EMPLOY1"], errors="coerce").map(_employ_group)
    marital = pd.to_numeric(df["MARITAL"], errors="coerce").map(_marital_group)

    cov_ok = (age_ok & sex_ok & race.notna().to_numpy() & educ.notna().to_numpy()
              & employ.notna().to_numpy() & marital.notna().to_numpy())

    aux = pd.DataFrame({
        "age": age, "female": female, "race": race.to_numpy(),
        "educ": educ.to_numpy(), "employ": employ.to_numpy(),
        "marital": marital.to_numpy(),
    })

    # standardise age over valid rows for numerical conditioning
    age_mean = np.nanmean(age[cov_ok])
    age_sd = np.nanstd(age[cov_ok])
    age_z = (age - age_mean) / age_sd

    cols = {"intercept": np.ones(n), "age_z": age_z, "age_z2": age_z ** 2,
            "female": female}

    def add_dummies(series, labels, ref):
        for lab in labels:
            if lab == ref:
                continue
            cols[f"{series.name}_{lab}"] = (series == lab).to_numpy(float)

    race.name = "race"; add_dummies(race, list(dict.fromkeys(_RACE_LABELS.values())), ref_race)
    educ.name = "educ"; add_dummies(educ, list(_EDUC_LABELS.values()), ref_educ)
    employ.name = "employ"; add_dummies(employ, ["Employed", "Unemployed",
                                        "Homemaker_Student", "Retired", "Unable"], ref_employ)
    marital.name = "marital"; add_dummies(marital, ["Married", "Div_Sep", "Widowed",
                                          "NeverMarried", "UnmarriedCouple"], ref_marital)

    colnames = list(cols.keys())
    X = np.column_stack([cols[c] for c in colnames])
    # zero out NaNs in invalid rows so column_stack is finite (rows excluded anyway)
    X = np.where(np.isfinite(X), X, 0.0)
    return X, colnames, cov_ok, aux


# ---------------------------------------------------------------------------
# full recode
# ---------------------------------------------------------------------------
def make_recoded(df, outcome="fmd", top_mid_usd=TOP_BRACKET_MIDPOINT_USD,
                 rescale_weights=True):
    """Turn the raw BRFSS DataFrame into a :class:`Recoded` plus a diagnostics frame.

    ``outcome`` selects the binary outcome from :data:`OUTCOMES` ('fmd' or 'vision').
    Rows are kept only if the outcome and all covariates are present. Income may be
    a real bracket (``kind='bracket'``), Don't know (``'dk'``), or Refused
    (``'refused'``); income 'not asked' (blank) is dropped. Returns
    ``(Recoded, info)`` where ``info`` is a dict of counts/diagnostics and the
    diagnostics DataFrame is attached as ``info['aux']``.
    """
    outcome_fn, outcome_label = OUTCOMES[outcome]
    inc = pd.to_numeric(df["INCOME3"], errors="coerce")
    log_bounds = income_log_bounds()
    log_mid_map = income_log_midpoint(top_mid_usd)

    kind = np.full(len(df), "drop", dtype=object)
    log_lo = np.full(len(df), np.nan)
    log_hi = np.full(len(df), np.nan)
    log_mid = np.full(len(df), np.nan)

    for code, (lo, hi) in log_bounds.items():
        sel = (inc == code).to_numpy()
        kind[sel] = "bracket"
        log_lo[sel] = lo
        log_hi[sel] = hi
        log_mid[sel] = log_mid_map[code]
    kind[(inc == INCOME_DK).to_numpy()] = "dk"
    kind[(inc == INCOME_REFUSED).to_numpy()] = "refused"
    # everything else (blank / not asked) stays 'drop'

    y_all, y_valid = outcome_fn(df)

    X, colnames, cov_ok, aux = build_design(df)
    w = pd.to_numeric(df["_LLCPWT"], errors="coerce").to_numpy(float)
    strata = pd.to_numeric(df["_STSTR"], errors="coerce").to_numpy()
    w_ok = np.isfinite(w) & (w > 0)

    keep = (np.isin(kind, ["bracket", "dk", "refused"]) & y_valid & cov_ok & w_ok)

    wk = w[keep].copy()
    if rescale_weights:
        wk = wk * (wk.size / wk.sum())  # rescale to mean 1

    rec = Recoded(
        X=X[keep], y=y_all[keep], w=wk, kind=kind[keep].astype(str),
        log_lo=log_lo[keep], log_hi=log_hi[keep], log_mid=log_mid[keep],
    )
    aux_keep = aux.loc[keep].reset_index(drop=True)
    aux_keep["kind"] = rec.kind
    aux_keep["outcome"] = rec.y
    aux_keep["w"] = rec.w
    aux_keep["strata"] = strata[keep]

    info = {
        "n_raw": len(df),
        "n_kept": int(keep.sum()),
        "n_bracket": int((rec.kind == "bracket").sum()),
        "n_dk": int((rec.kind == "dk").sum()),
        "n_refused": int((rec.kind == "refused").sum()),
        "colnames": colnames,
        "aux": aux_keep,
        "strata": strata[keep],
        "outcome": outcome,
        "outcome_label": outcome_label,
        "dropped_income_blank": int((np.isin(kind, ["bracket", "dk", "refused"]) == False).sum()),
    }
    return rec, info


# ---------------------------------------------------------------------------
# serialize the analysis sample (so downstream notebooks load in seconds)
# ---------------------------------------------------------------------------
def recoded_to_frame(rec, colnames, strata):
    """Flatten a :class:`Recoded` (+ design column names and strata) to a DataFrame.
    Design columns get an ``X__`` prefix; ``+-inf`` bracket bounds round-trip fine."""
    out = pd.DataFrame(rec.X, columns=[f"X__{c}" for c in colnames])
    out["y"] = rec.y
    out["w"] = rec.w
    out["kind"] = rec.kind
    out["log_lo"] = rec.log_lo
    out["log_hi"] = rec.log_hi
    out["log_mid"] = rec.log_mid
    out["strata"] = strata
    return out


def frame_to_recoded(frame):
    """Inverse of :func:`recoded_to_frame`. Returns ``(Recoded, colnames, strata)``."""
    xcols = [c for c in frame.columns if c.startswith("X__")]
    rec = Recoded(
        X=frame[xcols].to_numpy(float),
        y=frame["y"].to_numpy(float),
        w=frame["w"].to_numpy(float),
        kind=frame["kind"].to_numpy().astype(str),
        log_lo=frame["log_lo"].to_numpy(float),
        log_hi=frame["log_hi"].to_numpy(float),
        log_mid=frame["log_mid"].to_numpy(float),
    )
    colnames = [c[3:] for c in xcols]
    strata = frame["strata"].to_numpy()
    return rec, colnames, strata
