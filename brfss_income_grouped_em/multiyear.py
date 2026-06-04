"""Multi-year BRFSS income-item nonresponse extraction.

For each annual LLCP file we pull only the income item, education, age group, and the
final weight, flag **Refused (99)** vs **Don't know (77)** on income (kept distinct),
and attach survey year and an approximate birth cohort. The income variable name
changed (INCOME2 with 8 brackets through 2020, INCOME3 with 11 from 2021), but the
77/99 missingness codes are constant — verify per year before trusting.
"""
import os
import zipfile

import numpy as np
import pandas as pd
import pyreadstat

YEARS = [2011, 2013, 2015, 2017, 2019, 2021, 2022, 2023]

INCOME_DK = 77
INCOME_REFUSED = 99

# _AGEG5YR code -> approximate age midpoint (14 = DK/Refused/Missing -> drop)
AGEG5YR_MID = {1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52, 8: 57,
               9: 62, 10: 67, 11: 72, 12: 77, 13: 82}
EDUC_LABELS = {1: "LtHS", 2: "HSgrad", 3: "SomeColl", 4: "CollGrad"}


def _xpt_path(raw_dir, year):
    d = os.path.join(raw_dir, str(year))
    xpt = os.path.join(d, f"LLCP{year}.XPT")
    if not os.path.exists(xpt):
        z = zipfile.ZipFile(os.path.join(d, f"LLCP{year}XPT.zip"))
        mem = [m for m in z.namelist() if m.upper().strip().endswith(".XPT")][0]
        with z.open(mem) as src, open(xpt, "wb") as out:
            out.write(src.read())
    return xpt


def _read_xport(path, **kw):
    """read_xport with a latin1 fallback (some annual XPTs aren't valid UTF-8)."""
    try:
        return pyreadstat.read_xport(path, **kw)
    except UnicodeDecodeError:
        return pyreadstat.read_xport(path, encoding="latin1", **kw)


def load_year(raw_dir, year):
    """Return a slim per-year frame with income missingness flags, educ, cohort, weight."""
    xpt = _xpt_path(raw_dir, year)
    _, meta = _read_xport(xpt, metadataonly=True)
    cols = set(meta.column_names)
    inc = "INCOME3" if "INCOME3" in cols else ("INCOME2" if "INCOME2" in cols else None)
    wt = "_LLCPWT" if "_LLCPWT" in cols else ("_FINALWT" if "_FINALWT" in cols else None)
    if inc is None or wt is None:
        raise RuntimeError(f"{year}: income var={inc}, weight={wt}, cols missing")
    use = [c for c in [inc, "_EDUCAG", "_AGEG5YR", wt] if c in cols]
    df, _ = _read_xport(xpt, usecols=use)
    df = df.rename(columns={inc: "INCOME", wt: "WT"})

    income = pd.to_numeric(df["INCOME"], errors="coerce")
    educ = pd.to_numeric(df["_EDUCAG"], errors="coerce")
    ageg = pd.to_numeric(df["_AGEG5YR"], errors="coerce")
    out = pd.DataFrame({
        "year": year,
        "income_var": inc,
        "w": pd.to_numeric(df["WT"], errors="coerce"),
        "refused": (income == INCOME_REFUSED).astype(float),
        "dk": (income == INCOME_DK).astype(float),
        "income_asked": income.notna().to_numpy(),   # blank = not asked
        "educ": educ.map(EDUC_LABELS),
        "age_mid": ageg.map(AGEG5YR_MID),
    })
    out["birth_year"] = year - out["age_mid"]
    out["cohort"] = (np.floor(out["birth_year"] / 10) * 10)  # decade of birth
    return out


def load_all(raw_dir, years=YEARS):
    frames = [load_year(raw_dir, y) for y in years]
    return pd.concat(frames, ignore_index=True)


def wprop(df, flag, by=None):
    """Weighted proportion of ``flag`` (with design-naive SE via effective n)."""
    def agg(g):
        w = g["w"].to_numpy()
        y = g[flag].to_numpy()
        ok = np.isfinite(w) & (w > 0) & np.isfinite(y)
        w, y = w[ok], y[ok]
        if w.sum() == 0:
            return pd.Series({"p": np.nan, "se": np.nan, "n": 0})
        p = np.average(y, weights=w)
        n_eff = w.sum() ** 2 / (w ** 2).sum()
        return pd.Series({"p": p, "se": np.sqrt(max(p * (1 - p), 0) / n_eff), "n": len(w)})
    if by is None:
        return agg(df)
    return df.groupby(by, observed=True).apply(agg, include_groups=False).reset_index()
