#!/usr/bin/env python
"""Simulate an ambulatory EHR delivery.

The real extract is limited-use data that cannot be browsed while writing
analysis code, so this script builds a synthetic delivery with the exact tables,
column names, column order and dtypes the private schema configuration describes
(see ``schema_config``). Analysis code developed against the simulation should
run unchanged against the real extract.

The simulator reproduces the delivery's *analytic hazards* rather than its
clinical detail: the documented fill rates including structured missingness, the
de-identification artifacts (age capped at 89, BMI and weight capped, height
redacted, three-digit ZIPs blanked where the population is small, redacted
diagnosis codes, death dates shifted forward by a few weeks), organization-scoped
practitioner ids, and the denominator traps (people with no clinical records at
all; a mortality table holding only decedents because the deceased indicator on
the person table is never populated).

Ground truth that de-identification destroys is written to a ``_truth/``
subdirectory so the output doubles as a testbed for recovery methods.

    python simulate_vnehr.py --n-patients 5000 --seed 12345
    python simulate_vnehr.py --format psv --out ../data/derived/vnehr_sim
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

import roles
import schema_config

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, "..", "data", "derived", "vnehr_sim"))

# Delivery covers roughly 2010 through mid-2025.
START_DAY = int(np.datetime64("2010-01-01", "D").astype("int64"))
EXTRACT_DAY = int(np.datetime64("2025-06-30", "D").astype("int64"))
EXTRACT_YEAR = 2025
YEAR = 365.25

# Fields flagged "Populated" whose fill rate rounds to 0.0 in the dictionary are
# rare rather than absent; give them a token presence so code that touches them
# is exercised without moving the reported rate off 0.0.
TRACE_RATE = 0.0004

# --------------------------------------------------------------------------
# schema, dtypes, and the missingness machinery
# --------------------------------------------------------------------------

def value_set(cfg, value_set_role):
    """Category list for a documented value set.

    The dictionary writes a marker inside its value sheets where a field may be
    left unset, so that marker is dropped rather than emitted as a category.
    """
    return [v["value"] for v in cfg.value_set(value_set_role) if v["value"] != cfg.null_marker]


def rate_of(cfg, table_role, field_role):
    """Documented fill rate as a proportion."""
    return cfg.fill_rate(table_role, field_role) / 100.0


def null_column(n, field):
    """An all-null column of the dtype implied by the dictionary."""
    base = field["type"]["base"]
    if base in ("date", "datetime"):
        return pd.Series(np.full(n, "NaT", dtype="datetime64[us]"))
    if base == "float":
        return pd.Series(np.full(n, np.nan))
    dtype = {"varchar": "string", "numeric": "Int64", "bit": "boolean"}.get(base, "string")
    return pd.Series(pd.array([pd.NA] * n, dtype=dtype))


def cast_column(values, field, n):
    """Cast generated content to the dtype the dictionary implies."""
    base = field["type"]["base"]
    if values is None:
        return null_column(n, field)
    if base in ("date", "datetime"):
        return pd.Series(np.asarray(values, dtype="datetime64[us]"))
    if base == "float":
        return pd.Series(np.asarray(values, dtype="float64"))
    if base == "numeric":
        return pd.Series(pd.array(np.asarray(values, dtype="int64"), dtype="Int64"))
    if base == "bit":
        return pd.Series(pd.array(np.asarray(values, dtype="bool"), dtype="boolean"))
    text = pd.Series(pd.array(np.asarray(values, dtype=object), dtype="string"))
    length = field["type"].get("length")
    if length:
        text = text.str.slice(0, length)
    return text


def exact_mask(n, rate, rng, weights=None):
    """Boolean keep-mask with exactly ``round(n * rate)`` True entries.

    ``weights`` gives each row a relative propensity to be observed; rows are
    then drawn without replacement in proportion to those weights
    (Efraimidis-Spirakis), which buys structured missingness while still
    landing on the documented marginal exactly.
    """
    k = int(round(n * rate))
    k = max(0, min(n, k))
    keep = np.zeros(n, dtype=bool)
    if k == 0:
        return keep
    if k == n:
        keep[:] = True
        return keep
    if weights is None:
        keep[rng.choice(n, size=k, replace=False)] = True
        return keep
    w = np.clip(np.asarray(weights, dtype="float64"), 1e-9, None)
    key = np.log(rng.random(n)) / w
    keep[np.argpartition(key, n - k)[n - k:]] = True
    return keep


def apply_fill(frame, fields, rng, weights=None, preset=()):
    """Null out values so every field lands on its documented fill rate.

    Fields named in ``preset`` already carry structural missingness (links that
    could not be made, vitals that were never taken) and are left alone.
    """
    weights = weights or {}
    preset = set(preset)
    n = len(frame)
    for field in fields:
        name = field["name"]
        if name not in frame.columns:
            frame[name] = null_column(n, field)
        if field["status"] != "Populated":
            frame[name] = null_column(n, field)
            continue
        if name in preset:
            continue
        rate = (field["fill_rate"] or 0.0) / 100.0
        if rate <= 0:
            rate = TRACE_RATE
        if rate >= 1.0 or n == 0:
            continue
        keep = exact_mask(n, rate, rng, weights.get(name))
        frame[name] = frame[name].mask(pd.Series(~keep, index=frame.index))
    return frame[[f["name"] for f in fields]]


def trim_to_rate(series, rate, rng):
    """Null out surplus non-null entries so a derived field meets its target.

    Activity-date fields are computed from the generated records, so their fill
    rate falls out of the simulation rather than being chosen; this only ever
    removes values, never invents them.
    """
    present = series.notna().to_numpy()
    n = len(series)
    target = int(round(n * rate))
    if present.sum() <= target:
        return series
    idx = np.flatnonzero(present)
    drop = rng.choice(idx, size=present.sum() - target, replace=False)
    cond = np.zeros(n, dtype=bool)
    cond[drop] = True
    return series.mask(pd.Series(cond, index=series.index))


def finalize(cfg, table_role, data, n):
    """Build the delivered frame: configured column order, configured dtypes."""
    frame = pd.DataFrame(index=pd.RangeIndex(n))
    for field in cfg.field_specs(table_role):
        frame[field["name"]] = cast_column(data.get(field["name"]), field, n)
    return frame


# --------------------------------------------------------------------------
# date helpers -- everything is generated as int64 days since the epoch
# --------------------------------------------------------------------------

def as_date(days):
    """Day numbers to a midnight datetime64 column (dictionary type ``date``)."""
    return np.asarray(days, dtype="int64").astype("datetime64[D]").astype("datetime64[us]")


def as_datetime(days, rng, lo_hour=7, hi_hour=19):
    """Day numbers to a datetime column carrying a plausible clinic time.

    Times are real in the delivery, so anything that groups by day has to
    truncate; keeping them here makes that failure visible locally.
    """
    days = np.asarray(days, dtype="int64")
    seconds = rng.integers(lo_hour * 3600, hi_hour * 3600, size=days.size)
    return (days.astype("datetime64[D]").astype("datetime64[s]")
            + seconds.astype("timedelta64[s]")).astype("datetime64[us]")


# --------------------------------------------------------------------------
# clinical catalogs
#
# LOINC / NDC / RxNorm / CVX / ICD-10 / CPT values below are drawn from the
# public vocabularies and are realistic in both shape and meaning, but they are
# not validated against any particular vocabulary release. Treat the simulator
# as a source of plausible codes, never as a terminology reference.
# --------------------------------------------------------------------------

# (panel, test, loinc, units, mean, sd, lognormal, low, high, condition, shift,
#  messy free-text variants used on the rows that carry no LOINC)
NUMERIC_TESTS = [
    (None, "Hemoglobin A1c", "4548-4", "%", 5.6, 0.6, False, 4.0, 15.0, "diabetes", 2.3,
     ["HGB A1C", "Hemoglobin A1c ", "hba1c", "A1C", "HbA1c (Glycohemoglobin)"]),
    ("Comprehensive metabolic panel", "Glucose", "2345-7", "mg/dL", 95, 18, False, 40, 500,
     "diabetes", 55, ["GLUCOSE", "glucose, serum ", "Glu", "BLOOD GLUCOSE"]),
    ("Comprehensive metabolic panel", "Creatinine", "2160-0", "mg/dL", 0.95, 0.25, True,
     0.3, 12.0, "ckd", 0.9, ["CREATININE", "creat", "Creatinine, Ser ", "CREAT SER"]),
    (None, "eGFR", "33914-3", "mL/min/1.73m2", 92, 18, False, 4, 140, "ckd", -38,
     ["EGFR", "GFR estimated", "eGFR (MDRD)", "gfr "]),
    ("Comprehensive metabolic panel", "Sodium", "2951-2", "mmol/L", 139, 3.0, False, 118, 158,
     None, 0, ["SODIUM", "Na", "sodium, serum", "NA+ "]),
    ("Comprehensive metabolic panel", "Potassium", "2823-3", "mmol/L", 4.2, 0.42, False,
     2.2, 7.2, "ckd", 0.35, ["POTASSIUM", "K", "potassium ", "K+"]),
    ("Comprehensive metabolic panel", "Chloride", "2075-0", "mmol/L", 102, 3.2, False, 80, 122,
     None, 0, ["CHLORIDE", "Cl", "chloride, serum "]),
    ("Comprehensive metabolic panel", "Carbon dioxide", "2028-9", "mmol/L", 25, 2.6, False,
     10, 40, None, 0, ["CO2", "co2, serum", "BICARB ", "Bicarbonate"]),
    ("Comprehensive metabolic panel", "Urea nitrogen", "3094-0", "mg/dL", 15, 5.0, True,
     3, 120, "ckd", 12, ["BUN", "bun ", "Urea Nitrogen (BUN)", "UREA N"]),
    ("Comprehensive metabolic panel", "Calcium", "17861-6", "mg/dL", 9.4, 0.45, False,
     6.0, 13.0, None, 0, ["CALCIUM", "ca", "calcium, total "]),
    ("Hepatic function panel", "Alanine aminotransferase", "1742-6", "U/L", 24, 0.5, True,
     4, 400, "masld", 16, ["ALT", "alt (sgpt)", "SGPT ", "ALT/SGPT"]),
    ("Hepatic function panel", "Aspartate aminotransferase", "1920-8", "U/L", 23, 0.45, True,
     4, 400, "masld", 11, ["AST", "ast (sgot)", "SGOT ", "AST/SGOT"]),
    ("Hepatic function panel", "Alkaline phosphatase", "6768-6", "U/L", 76, 0.32, True,
     15, 500, None, 0, ["ALK PHOS", "alkaline phosphatase ", "ALKP"]),
    ("Hepatic function panel", "Bilirubin total", "1975-2", "mg/dL", 0.6, 0.4, True,
     0.1, 12.0, None, 0, ["TBILI", "bilirubin, total", "T Bili "]),
    ("Comprehensive metabolic panel", "Albumin", "1751-7", "g/dL", 4.2, 0.36, False,
     1.5, 5.6, "ckd", -0.35, ["ALBUMIN", "alb", "albumin, serum "]),
    ("Comprehensive metabolic panel", "Protein total", "2885-2", "g/dL", 7.0, 0.45, False,
     4.0, 10.0, None, 0, ["TP", "total protein ", "PROTEIN TOTAL"]),
    ("CBC with differential", "Platelets", "777-3", "K/uL", 250, 0.27, True, 20, 800,
     "masld", -55, ["PLT", "platelet count ", "PLATELETS", "plt count"]),
    ("CBC with differential", "Hemoglobin", "718-7", "g/dL", 13.9, 1.5, False, 5.0, 19.0,
     "ckd", -1.6, ["HGB", "hemoglobin ", "HB", "Hgb (Hemoglobin)"]),
    ("CBC with differential", "Hematocrit", "4544-3", "%", 41.5, 4.3, False, 15, 60,
     "ckd", -4.5, ["HCT", "hematocrit ", "Hct%"]),
    ("CBC with differential", "Leukocytes", "6690-2", "K/uL", 7.0, 0.3, True, 1.0, 40.0,
     None, 0, ["WBC", "wbc count ", "White Blood Cell Count", "W.B.C."]),
    ("CBC with differential", "Erythrocytes", "789-8", "M/uL", 4.7, 0.5, False, 2.0, 7.0,
     None, 0, ["RBC", "rbc ", "Red Blood Cell Count"]),
    ("CBC with differential", "MCV", "787-2", "fL", 89, 6.0, False, 55, 125, None, 0,
     ["MCV", "mcv ", "Mean Corpuscular Volume"]),
    ("Lipid panel", "Cholesterol total", "2093-3", "mg/dL", 190, 0.2, True, 80, 420,
     "hyperlipidemia", 38, ["CHOL", "cholesterol, total ", "TOTAL CHOLESTEROL", "chol tot"]),
    ("Lipid panel", "Cholesterol in HDL", "2085-9", "mg/dL", 54, 0.28, True, 15, 130,
     "obesity", -11, ["HDL", "hdl chol ", "HDL-C", "Cholesterol HDL"]),
    ("Lipid panel", "Cholesterol in LDL", "13457-7", "mg/dL", 110, 0.28, True, 20, 300,
     "hyperlipidemia", 34, ["LDL", "ldl calc ", "LDL-C (calc)", "LDL CHOLESTEROL"]),
    ("Lipid panel", "Triglycerides", "2571-8", "mg/dL", 118, 0.48, True, 30, 900,
     "hyperlipidemia", 72, ["TRIG", "triglycerides ", "TRIGLYCERIDE", "trig."]),
    (None, "Thyrotropin", "3016-3", "uIU/mL", 2.0, 0.55, True, 0.01, 60.0, None, 0,
     ["TSH", "tsh ", "Thyroid Stimulating Hormone", "T.S.H."]),
    (None, "Thyroxine free", "3024-7", "ng/dL", 1.2, 0.22, False, 0.2, 4.0, None, 0,
     ["FREE T4", "ft4 ", "T4, Free"]),
    (None, "25-hydroxyvitamin D3", "1989-3", "ng/mL", 31, 0.4, True, 4, 120, None, 0,
     ["VIT D", "vitamin d 25-oh ", "25-OH VITAMIN D", "vit d, 25 hydroxy"]),
    (None, "Albumin/Creatinine ratio, urine", "9318-7", "mg/g", 12, 1.1, True, 1, 3000,
     "ckd", 220, ["UACR", "urine microalbumin/creat ratio ", "MICROALB/CREAT"]),
    (None, "Prostate specific antigen", "2857-1", "ng/mL", 1.1, 0.7, True, 0.05, 60.0,
     None, 0, ["PSA", "psa, total ", "P.S.A."]),
    (None, "Cobalamin (Vitamin B12)", "2132-9", "pg/mL", 480, 0.4, True, 90, 2000, None, 0,
     ["B12", "vitamin b12 ", "VIT B-12"]),
    (None, "Ferritin", "2276-4", "ng/mL", 90, 0.85, True, 3, 1500, None, 0,
     ["FERRITIN", "ferritin ", "FERR"]),
    (None, "INR", "6301-6", "{INR}", 1.1, 0.22, True, 0.8, 8.0, None, 0,
     ["INR", "inr ", "PT-INR", "Protime INR"]),
    (None, "Vitamin B1", "1959-6", "nmol/L", 130, 0.3, True, 20, 400, None, 0,
     ["THIAMINE", "vit b1 "]),
]

# Qualitative tests: (panel, test, loinc, result values with weights, messy variants)
QUALITATIVE_TESTS = [
    (None, "Protein, urine", "5804-0", ["Negative", "Trace", "1+", "2+", "3+"],
     [0.72, 0.14, 0.08, 0.04, 0.02], ["URINE PROTEIN", "protein, ua ", "UA PROTEIN"]),
    (None, "Streptococcus pyogenes rapid Ag", "11268-0", ["Negative", "Positive"],
     [0.82, 0.18], ["RAPID STREP", "strep a screen ", "STREP A AG"]),
    (None, "SARS-CoV-2 RNA", "94500-6", ["Not Detected", "Detected", "Invalid"],
     [0.86, 0.13, 0.01], ["COVID PCR", "sars-cov-2 pcr ", "COVID-19 NAA"]),
    (None, "Influenza virus A Ag", "80382-5", ["Negative", "Positive"], [0.88, 0.12],
     ["FLU A", "influenza a antigen ", "INFLUENZA A"]),
]

# (name, ndc, RxNorm CUI, strength, units, form, route, class)
DRUG_CATALOG = [
    ("Metformin HCl 500 MG oral tablet", "00093104801", "861007", "500", "MG", "Tabs", "Oral", "diabetes"),
    ("Metformin HCl ER 1000 MG oral tablet", "00093105001", "861004", "1000", "MG", "Tabs", "Oral", "diabetes"),
    ("Glipizide 5 MG oral tablet", "00049411066", "310490", "5", "MG", "Tabs", "Oral", "diabetes"),
    ("Empagliflozin 10 MG oral tablet", "00597015230", "1545658", "10", "MG", "Tabs", "Oral", "diabetes"),
    ("Semaglutide 1 MG/0.75 ML pen injector", "00169413013", "1991306", "1", "MG/0.75ML", "Soln", "Subcutaneous", "glp1"),
    ("Dulaglutide 0.75 MG/0.5 ML pen injector", "00002143380", "1551300", "0.75", "MG/0.5ML", "Soln", "Subcutaneous", "glp1"),
    ("Liraglutide 18 MG/3 ML pen injector", "00169406013", "897122", "18", "MG/3ML", "Soln", "Subcutaneous", "glp1"),
    ("Tirzepatide 5 MG/0.5 ML pen injector", "00002150680", "2601755", "5", "MG/0.5ML", "Soln", "Subcutaneous", "glp1"),
    ("Insulin glargine 100 UNT/ML injection", "00088222033", "285018", "100", "UNT/ML", "Soln", "Subcutaneous", "diabetes"),
    ("Lisinopril 10 MG oral tablet", "00093113201", "314076", "10", "MG", "Tabs", "Oral", "hypertension"),
    ("Lisinopril 20 MG oral tablet", "00093113301", "314077", "20", "MG", "Tabs", "Oral", "hypertension"),
    ("Amlodipine besylate 5 MG oral tablet", "00591036101", "197361", "5", "MG", "Tabs", "Oral", "hypertension"),
    ("Amlodipine besylate 10 MG oral tablet", "00591036201", "197362", "10", "MG", "Tabs", "Oral", "hypertension"),
    ("Losartan potassium 50 MG oral tablet", "00093730598", "979492", "50", "MG", "Tabs", "Oral", "hypertension"),
    ("Hydrochlorothiazide 25 MG oral tablet", "00143127201", "310798", "25", "MG", "Tabs", "Oral", "hypertension"),
    ("Metoprolol tartrate 50 MG oral tablet", "00378002310", "866924", "50", "MG", "Tabs", "Oral", "hypertension"),
    ("Metoprolol succinate ER 50 MG oral tablet", "00186100405", "866412", "50", "MG", "Tabs", "Oral", "hypertension"),
    ("Furosemide 20 MG oral tablet", "00054429725", "310429", "20", "MG", "Tabs", "Oral", "hypertension"),
    ("Atorvastatin calcium 10 MG oral tablet", "00378395405", "617312", "10", "MG", "Tabs", "Oral", "hyperlipidemia"),
    ("Atorvastatin calcium 40 MG oral tablet", "00378395605", "617314", "40", "MG", "Tabs", "Oral", "hyperlipidemia"),
    ("Simvastatin 20 MG oral tablet", "00006074054", "312961", "20", "MG", "Tabs", "Oral", "hyperlipidemia"),
    ("Rosuvastatin calcium 10 MG oral tablet", "00310075190", "859419", "10", "MG", "Tabs", "Oral", "hyperlipidemia"),
    ("Sertraline HCl 50 MG oral tablet", "00049049066", "312940", "50", "MG", "Tabs", "Oral", "depression"),
    ("Escitalopram oxalate 10 MG oral tablet", "00456202001", "351250", "10", "MG", "Tabs", "Oral", "depression"),
    ("Fluoxetine HCl 20 MG oral capsule", "00777310502", "310385", "20", "MG", "Caps", "Oral", "depression"),
    ("Albuterol sulfate 90 MCG/ACT inhaler", "00173068220", "745679", "90", "MCG/ACT", "Aero", "Inhalation", "asthma"),
    ("Fluticasone-salmeterol 250-50 MCG/DOSE inhaler", "00173069700", "896188", "250-50", "MCG/DOSE", "Aepb", "Inhalation", "asthma"),
    ("Montelukast sodium 10 MG oral tablet", "00006027754", "153165", "10", "MG", "Tabs", "Oral", "asthma"),
    ("Tiotropium bromide 18 MCG inhalation capsule", "00597007541", "485210", "18", "MCG", "Caps", "Inhalation", "copd"),
    ("Levothyroxine sodium 50 MCG oral tablet", "00074611519", "966224", "50", "MCG", "Tabs", "Oral", "general"),
    ("Omeprazole 20 MG delayed release capsule", "00093510901", "402014", "20", "MG", "Caps", "Oral", "general"),
    ("Pantoprazole sodium 40 MG oral tablet", "00008084181", "314200", "40", "MG", "Tabs", "Oral", "general"),
    ("Amoxicillin 500 MG oral capsule", "00093310901", "308192", "500", "MG", "Caps", "Oral", "acute"),
    ("Azithromycin 250 MG oral tablet", "00093715634", "308460", "250", "MG", "Tabs", "Oral", "acute"),
    ("Cephalexin 500 MG oral capsule", "00143992501", "309098", "500", "MG", "Caps", "Oral", "acute"),
    ("Ibuprofen 600 MG oral tablet", "00378018101", "197806", "600", "MG", "Tabs", "Oral", "acute"),
    ("Acetaminophen 500 MG oral tablet", "50580044810", "313782", "500", "MG", "Tabs", "Oral", "otc"),
    ("Aspirin 81 MG delayed release tablet", "00904404280", "243670", "81", "MG", "Tabs", "Oral", "otc"),
    ("Cetirizine HCl 10 MG oral tablet", "00078043415", "1011483", "10", "MG", "Tabs", "Oral", "otc"),
    ("Cholecalciferol 1000 UNT oral tablet", "00904582260", "349382", "1000", "UNT", "Tabs", "Oral", "otc"),
    ("Prednisone 10 MG oral tablet", "00054001825", "312615", "10", "MG", "Tabs", "Oral", "acute"),
    ("Gabapentin 300 MG oral capsule", "00093122901", "310431", "300", "MG", "Caps", "Oral", "general"),
    ("Hydrocodone-acetaminophen 5-325 MG oral tablet", "00406012301", "857002", "5-325", "MG", "Tabs", "Oral", "general"),
    ("Warfarin sodium 5 MG oral tablet", "00056017075", "855338", "5", "MG", "Tabs", "Oral", "general"),
    ("Apixaban 5 MG oral tablet", "00003089421", "1364445", "5", "MG", "Tabs", "Oral", "general"),
]

DRUG_SIGS = [
    "Take 1 tablet by mouth once daily", "Take 1 tablet by mouth twice daily",
    "TAKE 1 TABLET BY MOUTH EVERY DAY", "1 po qd", "Take one tablet daily with food",
    "Take 2 tablets by mouth at bedtime", "Inject 1 pen subcutaneously once weekly",
    "2 puffs inhaled every 4-6 hours as needed for shortness of breath",
    "take 1 capsule by mouth three times a day for 10 days", "1 tab po bid prn pain",
]

DRUG_FREQUENCY = ["daily", "twice daily", "3 times daily", "every 12 hours", "as needed",
                 "weekly", "at bedtime", "every 4-6 hours prn", "BID", "QD"]

DISCONTINUE_REASONS = ["Adverse reaction", "Ineffective", "Patient request", "Cost",
                       "Completed course", "Duplicate therapy", "No longer needed"]

# condition -> (icd10 codes, condition names)
CONDITION_CODES = {
    "diabetes": (["E11.9", "E11.65", "E11.22", "E11.40", "E11.21"],
                 ["Type 2 diabetes mellitus without complications",
                  "Type 2 diabetes mellitus with hyperglycemia",
                  "Type 2 diabetes mellitus with diabetic chronic kidney disease",
                  "Type 2 diabetes with diabetic neuropathy"]),
    "hypertension": (["I10", "I11.9", "I16.0"],
                     ["Essential (primary) hypertension", "Hypertension, unspecified",
                      "Hypertensive heart disease without heart failure"]),
    "hyperlipidemia": (["E78.5", "E78.2", "E78.00", "E78.1"],
                       ["Hyperlipidemia, unspecified", "Mixed hyperlipidemia",
                        "Pure hypercholesterolemia, unspecified"]),
    "obesity": (["E66.01", "E66.9", "E66.3", "Z68.35"],
                ["Morbid (severe) obesity due to excess calories", "Obesity, unspecified",
                 "Overweight"]),
    "ckd": (["N18.3", "N18.4", "N18.9", "N18.30", "N18.2"],
            ["Chronic kidney disease, stage 3 unspecified",
             "Chronic kidney disease, stage 4 (severe)",
             "Chronic kidney disease, unspecified"]),
    "masld": (["K76.0", "K75.81", "K74.01"],
              ["Nonalcoholic fatty liver disease (NAFLD)",
               "Nonalcoholic steatohepatitis (NASH)",
               "Metabolic dysfunction-associated steatotic liver disease"]),
    "depression": (["F32.9", "F33.1", "F32.A", "F41.1"],
                   ["Major depressive disorder, single episode, unspecified",
                    "Major depressive disorder, recurrent, moderate",
                    "Depression, unspecified", "Generalized anxiety disorder"]),
    "asthma": (["J45.909", "J45.40", "J45.20"],
               ["Unspecified asthma, uncomplicated", "Moderate persistent asthma, uncomplicated",
                "Mild intermittent asthma, uncomplicated"]),
    "copd": (["J44.9", "J44.1", "J43.9"],
             ["Chronic obstructive pulmonary disease, unspecified",
              "COPD with (acute) exacerbation", "Emphysema, unspecified"]),
}

ACUTE_CONDITIONS = [
    ("J06.9", "Acute upper respiratory infection, unspecified"),
    ("M54.50", "Low back pain, unspecified"),
    ("R51.9", "Headache, unspecified"),
    ("N39.0", "Urinary tract infection, site not specified"),
    ("J02.9", "Acute pharyngitis, unspecified"),
    ("L03.115", "Cellulitis of right lower limb"),
    ("R10.9", "Unspecified abdominal pain"),
    ("Z00.00", "Encounter for general adult medical examination without abnormal findings"),
    ("Z23", "Encounter for immunization"),
    ("M25.561", "Pain in right knee"),
    ("R53.83", "Other fatigue"),
    ("H66.90", "Otitis media, unspecified, unspecified ear"),
    ("K21.9", "Gastro-esophageal reflux disease without esophagitis"),
    ("E03.9", "Hypothyroidism, unspecified"),
    ("D50.9", "Iron deficiency anemia, unspecified"),
]

# Codes carried by the non-ICD-10 rows of the coded-diagnosis table.
ICD9_CODES = ["250.00", "401.9", "272.4", "278.00", "585.3", "571.8", "311", "493.90", "496"]
SNOMED_CONDITION_CODES = ["44054006", "38341003", "55822004", "414916001", "709044004",
                        "197315008", "35489007", "195967001", "13645005"]
CPT_CONDITION_CODES = ["99213", "99214", "99396", "99385", "99203"]

# (name, cpt, category role)
INTERVENTIONS = [
    ("Chest X-ray, 2 views", "71046", roles.INTERVENTION_IMAGING),
    ("Electrocardiogram, 12 lead", "93000", roles.INTERVENTION_DIAGNOSTIC_STUDY),
    ("Screening mammography, bilateral", "77067", roles.INTERVENTION_IMAGING),
    ("Colonoscopy, screening", "45378", roles.INTERVENTION_PROCEDURAL),
    ("Ultrasound, abdomen complete", "76700", roles.INTERVENTION_DIAGNOSTICS),
    ("CT abdomen and pelvis with contrast", "74177", roles.INTERVENTION_IMAGING),
    ("Transthoracic echocardiogram", "93306", roles.INTERVENTION_DIAGNOSTIC_STUDY),
    ("Spirometry", "94010", roles.INTERVENTION_OTHER_TESTING),
    ("DEXA bone density study", "77080", roles.INTERVENTION_IMAGING),
    ("MRI lumbar spine without contrast", "72148", roles.INTERVENTION_IMAGING),
    ("Depression screening PHQ-9", "96127", roles.INTERVENTION_FINDING),
    ("Diabetic foot exam", "G0245", roles.INTERVENTION_EXAM),
    ("Skin lesion biopsy", "11102", roles.INTERVENTION_PROCEDURAL),
    ("Immunization administration", "90471", roles.INTERVENTION_VACCINATION),
    ("Nerve conduction study", "95907", roles.INTERVENTION_OTHER_TESTING),
    ("Patient education, diabetes self-management", "G0108", roles.INTERVENTION_EDUCATION),
]

SERVICE_REQUEST_NAMES = ["Cardiology", "Gastroenterology", "Orthopedic Surgery", "Physical Therapy",
                  "Ophthalmology", "Dermatology", "Endocrinology", "Neurology", "Nephrology",
                  "General Surgery", "Behavioral Health", "Nutrition / Dietitian", "Podiatry",
                  "Urology", "Pulmonology", "Sleep Study", "Mammography", "Colonoscopy",
                  "Rheumatology", "Pain Management"]

# (vaccine name, cvx, cpt, covid-era only)
VACCINES = [
    ("Influenza, seasonal, injectable, preservative free", "140", "90686", False),
    ("Influenza, quadrivalent, injectable", "158", "90686", False),
    ("Influenza, high dose seasonal, preservative free", "135", "90662", False),
    ("COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose", "208", "91300", True),
    ("COVID-19, mRNA, LNP-S, PF, 100 mcg/0.5 mL dose", "207", "91301", True),
    ("COVID-19 vaccine, mRNA, 2023-2024 formula", "300", "91318", True),
    ("Zoster vaccine recombinant", "187", "90750", False),
    ("Tdap", "115", "90715", False),
    ("Td (adult), preservative free", "138", "90714", False),
    ("Pneumococcal conjugate PCV13", "133", "90670", False),
    ("Pneumococcal polysaccharide PPSV23", "33", "90732", False),
    ("Pneumococcal conjugate PCV20", "216", "90677", False),
    ("Hepatitis B, adult", "43", "90746", False),
    ("HPV9", "165", "90651", False),
    ("MMR", "03", "90707", False),
    ("Varicella", "21", "90716", False),
    ("RSV, prefusion F, recombinant", "305", "90679", False),
]

VACCINE_MANUFACTURERS = ["Sanofi Pasteur", "GlaxoSmithKline", "Merck Sharp & Dohme",
                              "Pfizer Inc", "Moderna US Inc", "Seqirus", "Novartis",
                              "Abbott Laboratories", "Janssen Products"]
MVX_CODES = ["PMC", "SKB", "MSD", "PFR", "MOD", "SEQ", "NOV"]

ALLERGENS = ["Penicillin", "Sulfa (sulfonamide antibiotics)", "Codeine", "Latex", "Peanuts",
             "Shellfish", "Bee stings", "Pollen", "Iodinated contrast", "Aspirin", "NSAIDs",
             "Morphine", "Tree nuts", "Dust mites", "Eggs", "Cephalexin", "Erythromycin",
             "Atorvastatin", "Amoxicillin", "Adhesive tape", "Cat dander", "Ragweed"]
INTOLERANCE_REACTIONS = ["Rash", "Hives", "Nausea", "Vomiting", "Swelling", "Anaphylaxis",
                     "Itching", "Shortness of breath", "Diarrhea", "Dizziness", "Cough",
                     "Angioedema", "GI upset", "Unknown"]
REACTION_SEVERITY = ["Mild", "Moderate", "Severe", "mild", "moderate", "Life threatening"]
INTOLERANCE_TYPES = ["Allergen", "User Defined", "Drug", "Food", "Environmental"]

SMOKING_TRUTH = ["Never smoker", "Former smoker", "Current every day smoker",
                 "Current some day smoker", "Unknown if ever smoked"]
SMOKING_TEXT = {
    "Never smoker": ["Never smoker", "never smoked", "Non-smoker", "NEVER SMOKER",
                     "Tobacco use: never", "nonsmoker"],
    "Former smoker": ["Former smoker", "Ex-smoker", "quit smoking", "FORMER SMOKER",
                      "Former smoker - quit 2011", "Tobacco use: former", "ex smoker"],
    "Current every day smoker": ["Current every day smoker", "Smoker", "current smoker",
                                 "CIGARETTE SMOKER", "smokes 1 ppd", "Tobacco use disorder",
                                 "Current smoker, every day"],
    "Current some day smoker": ["Current some day smoker", "occasional smoker",
                                "social smoker", "smokes occasionally"],
    "Unknown if ever smoked": ["Smoking status unknown", "Unknown if ever smoked",
                               "tobacco use unknown", "Smoking status not documented"],
}
# (category role, free-text descriptions filed under it)
NONSMOKING_BACKGROUND = [
    (roles.BACKGROUND_FAMILY,
     ["Family history of diabetes mellitus", "Family history of breast cancer",
      "Family history of coronary artery disease", "Family history of stroke"]),
    (roles.BACKGROUND_FAMILY_VARIANT,
     ["hypertension", "colon cancer", "diabetes mellitus type 2"]),
    (roles.BACKGROUND_SOCIAL,
     ["Alcohol use: occasional", "Lives alone", "Employed full time",
      "Regular exercise 3x/week", "Alcohol use: none"]),
    (roles.BACKGROUND_SURGERY,
     ["Appendectomy", "Cholecystectomy", "Total knee arthroplasty",
      "Cesarean section", "Tonsillectomy", "Coronary artery bypass graft"]),
    (roles.BACKGROUND_SURGERY_VARIANT,
     ["Hernia repair", "Cataract extraction, right eye", "Hysterectomy"]),
    (roles.BACKGROUND_DEVICE,
     ["CPAP machine", "Insulin pump", "Cardiac pacemaker", "Hearing aid, bilateral"]),
    (roles.BACKGROUND_PREVENTIVE_CARE,
     ["Colonoscopy 2019", "Mammogram 2022", "Annual wellness visit"]),
    (roles.BACKGROUND_TRAVEL, ["Travel to Mexico 2018", "Travel to Southeast Asia"]),
    (roles.BACKGROUND_PREGNANCY,
     ["G2P2", "Full term vaginal delivery", "Gestational diabetes"]),
]

FINDING_VITALS = [
    ("Blood pressure systolic", "8480-6", "mm[Hg]", 128, 16),
    ("Pulse", "8867-4", "/min", 76, 12),
    ("Body weight", "29463-7", "lb", 185, 45),
    ("Body temperature", "8310-5", "Cel", 36.8, 0.4),
    ("Respiratory rate", "9279-1", "/min", 16, 3),
    ("Pain severity score", "72514-3", "{score}", 3, 2.5),
    ("Oxygen saturation", "2708-6", "%", 97, 2),
    ("Head circumference", "9843-4", "cm", 45, 6),
]
FINDING_QUAL = [
    "Smoking status", "Depression screening", "Fall risk assessment", "Functional status",
    "Advance directive discussed", "Medication reconciliation", "Tobacco use screening",
    "Alcohol use screening", "Diabetic foot exam", "Diabetic eye exam", "Hearing screening",
    "Developmental screening", "Patient education provided", "Colorectal cancer screening",
    "Care plan reviewed", "Nutrition counseling", "Immunization reviewed", "Vision screening",
]
FINDING_QUAL_VALUES = ["Normal", "Abnormal", "Negative", "Positive", "Yes", "No",
                           "Completed", "Not indicated", "Discussed", "Low risk",
                           "Moderate risk", "WNL", "Refused", "Unable to perform", "Reviewed"]

PRACTITIONER_SPECIALTIES = [
    "Family Medicine", "Internal Medicine", "Pediatrics", "Nurse Practitioner",
    "Physician Assistant", "Obstetrics & Gynecology", "Cardiology", "Endocrinology",
    "Gastroenterology", "Nephrology", "Psychiatry", "Orthopedic Surgery", "Dermatology",
    "Registered Nurse", "Medical Assistant", "Office Staff", "Physical Therapy",
    "General Surgery", "Urology", "Pulmonology",
]
PRACTITIONER_CREDENTIALS = ["MD", "DO", "NP", "PA-C", "CNP", "CFNP", "LPN", "RN", "DPM"]
PRACTITIONER_TAXONOMY = ["207Q00000X", "207R00000X", "208000000X", "363L00000X", "363A00000X",
                     "207RC0000X", "207RE0101X", "2080P0214X", "207RX0202X"]

# state -> plausible zip3 prefixes; the redaction token is applied separately
STATE_ZIP3 = {
    "CA": ["900", "926", "941", "958"], "TX": ["750", "770", "787", "790"],
    "FL": ["331", "328", "336", "324"], "NY": ["100", "112", "142", "125"],
    "PA": ["191", "152", "170", "185"], "IL": ["606", "601", "617", "625"],
    "OH": ["441", "432", "452", "446"], "GA": ["303", "310", "316", "300"],
    "NC": ["272", "282", "275", "289"], "MI": ["481", "492", "495", "490"],
    "NJ": ["070", "080", "085", "088"], "VA": ["232", "222", "245", "241"],
    "WA": ["980", "981", "992", "985"], "AZ": ["850", "857", "863", "852"],
    "MA": ["021", "017", "015", "010"], "TN": ["372", "381", "374", "370"],
    "IN": ["462", "467", "473", "465"], "MO": ["631", "641", "654", "656"],
    "WI": ["532", "537", "544", "549"], "MN": ["554", "551", "559", "566"],
    "CO": ["802", "805", "816", "811"], "AL": ["352", "358", "366", "361"],
    "SC": ["294", "292", "296", "290"], "KY": ["402", "404", "421", "412"],
    "OR": ["972", "974", "977", "978"], "OK": ["731", "741", "734", "739"],
    "IA": ["502", "522", "515", "512"], "MS": ["392", "395", "387", "397"],
    "MT": ["590", "597", "594", "598"], "WY": ["820", "829", "826", "824"],
    "VT": ["054", "057", "058", "056"], "ND": ["580", "585", "588", "582"],
}
STATE_WEIGHTS = np.array(
    [8.0, 6.5, 5.5, 5.0, 4.2, 4.0, 3.6, 3.4, 3.3, 3.1, 2.8, 2.7, 2.5, 2.3, 2.2,
     2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.5, 0.4,
     0.4, 0.4]
)
# States whose zip3s are mostly small-population and therefore redacted.
SMALL_POP_STATES = {"MT", "WY", "VT", "ND"}


# --------------------------------------------------------------------------
# small sampling helpers
# --------------------------------------------------------------------------

def pick(rng, options, size, p=None):
    """Vectorised choice returning an object array of the given options."""
    values = np.asarray(options, dtype=object)
    return values[rng.choice(len(values), size=size, p=p)]


def categorical(rng, prob_matrix):
    """Row-wise categorical draw from an (n, k) matrix of probabilities."""
    gumbel = rng.gumbel(size=prob_matrix.shape)
    return np.argmax(np.log(np.clip(prob_matrix, 1e-12, None)) + gumbel, axis=1)


def admin_dates(days, rng, created_lag=(0, 3), update_lag=(0, 400)):
    """Record-keeping timestamps derived from a record's clinical date."""
    created = days + rng.integers(created_lag[0], created_lag[1] + 1, size=days.size)
    updated = created + rng.integers(update_lag[0], update_lag[1] + 1, size=days.size)
    created = np.minimum(created, EXTRACT_DAY)
    updated = np.minimum(updated, EXTRACT_DAY)
    return as_date(created), as_datetime(updated, rng)


def visit_intensity(days):
    """Relative encounter rate: EHR coverage grows, with a 2020 COVID trough."""
    frac = (np.asarray(days, dtype="float64") - START_DAY) / (EXTRACT_DAY - START_DAY)
    rate = 0.55 + 0.45 * frac
    year_frac = 1970 + np.asarray(days, dtype="float64") / YEAR
    deep = (year_frac >= 2020.20) & (year_frac < 2020.46)
    shallow = (year_frac >= 2020.46) & (year_frac < 2021.0)
    rate = np.where(deep, rate * 0.42, rate)
    rate = np.where(shallow, rate * 0.78, rate)
    return rate


def covid_weight(days):
    """0 before 2020, 1 through the pandemic years, 0.55 afterwards."""
    year_frac = 1970 + np.asarray(days, dtype="float64") / YEAR
    out = np.zeros(year_frac.shape)
    out = np.where((year_frac >= 2020.15) & (year_frac < 2022.0), 1.0, out)
    out = np.where(year_frac >= 2022.0, 0.55, out)
    return out


# --------------------------------------------------------------------------
# organizations and practitioners
# --------------------------------------------------------------------------

def build_organizations(cfg, rng, n_patients):
    """Organization frame plus the onboarding day that gates clinical activity."""
    blanked = cfg.deid.redacted_geography
    n = max(6, n_patients // 130)
    org_id = np.sort(rng.choice(np.arange(100_001, 999_999), size=n, replace=False))
    states = pick(rng, list(STATE_ZIP3), n, p=STATE_WEIGHTS / STATE_WEIGHTS.sum())
    zip3 = np.array([rng.choice(STATE_ZIP3[s]) for s in states], dtype=object)
    zip3 = np.where([s in SMALL_POP_STATES for s in states], blanked, zip3)
    zip3 = np.where(rng.random(n) < 0.05, blanked, zip3)
    onboard = START_DAY + (rng.beta(1.4, 2.2, size=n) * (EXTRACT_DAY - START_DAY - 400)).astype("int64")

    # A few organizations contribute no clinical records at all, which is what
    # keeps their activity dates from being fully populated.
    active = exact_mask(n, 0.974, rng)
    return pd.DataFrame({
        "organization_id": org_id, "state": states, "zip3": zip3,
        "onboard_day": onboard, "active": active,
    })


def build_practitioners(cfg, rng, orgs):
    """Practitioner frame; ids restart inside each organization, as documented.

    Practitioner ids are scoped to the organization, so the same integer names
    different people at different organizations. Joining on the practitioner id
    alone fans out; every join has to carry the organization id too. That trap is
    reproduced deliberately.
    """
    per_org = rng.integers(2, 26, size=len(orgs))
    org_id = np.repeat(orgs["organization_id"].to_numpy(), per_org)
    onboard = np.repeat(orgs["onboard_day"].to_numpy(), per_org)
    org_active = np.repeat(orgs["active"].to_numpy(), per_org)
    n = org_id.size

    practitioner_id = np.concatenate([
        np.sort(rng.choice(np.arange(1, 400), size=k, replace=False)) for k in per_org
    ])

    # Many EHR users (front-desk staff, retired clinicians) never touch a clinical
    # record, which is why practitioner activity dates are so sparsely populated.
    used = exact_mask(n, 0.782, rng) & org_active
    # Keep at least one usable practitioner per active organization.
    first = np.concatenate([[0], np.cumsum(per_org)[:-1]])
    used[first[orgs["active"].to_numpy()]] = True

    specialty = pick(rng, PRACTITIONER_SPECIALTIES, n)
    state = np.repeat(orgs["state"].to_numpy(), per_org)
    zip3 = np.repeat(orgs["zip3"].to_numpy(), per_org)
    zip3 = np.where(rng.random(n) < 0.03, cfg.deid.redacted_geography, zip3)
    return pd.DataFrame({
        "practitioner_id": practitioner_id, "organization_id": org_id,
        "stated_specialty": specialty, "stated_state": state, "stated_zip3": zip3,
        "onboard_day": onboard, "usable": used,
    })


def practitioner_sampler(practitioners, orgs):
    """Return a function mapping organization row indices to usable practitioners."""
    order = np.lexsort((~practitioners["usable"].to_numpy(),
                        practitioners["organization_id"].to_numpy()))
    sorted_practitioners = practitioners.iloc[order].reset_index(drop=True)
    oids = sorted_practitioners["organization_id"].to_numpy()
    starts = np.searchsorted(oids, orgs["organization_id"].to_numpy(), side="left")
    counts = np.searchsorted(oids, orgs["organization_id"].to_numpy(), side="right") - starts
    usable = sorted_practitioners["usable"].to_numpy()
    n_usable = np.array([usable[s:s + c].sum() for s, c in zip(starts, counts)])

    def sample(org_rows, rng):
        k = np.maximum(n_usable[org_rows], 1)
        offset = (rng.random(org_rows.size) * k).astype("int64")
        rows = starts[org_rows] + offset
        return sorted_practitioners["practitioner_id"].to_numpy()[rows], rows

    return sample, sorted_practitioners


# --------------------------------------------------------------------------
# people
# --------------------------------------------------------------------------

# People with no record in any clinical table; their activity dates are null.
NO_RECORD_RATE = 0.074

CONDITIONS = ["diabetes", "hypertension", "hyperlipidemia", "obesity", "ckd", "masld",
              "depression", "asthma", "copd"]


def build_patients(cfg, rng, n, orgs):
    """Latent state per person: demography, home organization, risk, conditions."""
    oldest = cfg.deid.age_ceiling
    # Opaque hash-like keys, as the real synthetic person identifier is.
    ids = np.unique(rng.integers(1, 2 ** 52, size=n))
    while ids.size < n:
        ids = np.unique(np.concatenate([ids, rng.integers(1, 2 ** 52, size=n - ids.size)]))
    person_id = np.array([f"HI{v:030x}" for v in rng.permutation(ids)], dtype=object)

    # Ambulatory panels skew adult and female; a small tail sits above the age cap.
    child = rng.random(n) < 0.17
    age = np.where(child,
                   np.clip(rng.normal(8, 5.5, n), 0, 17),
                   np.clip(rng.normal(49, 20.5, n), 18, 104))
    true_birth_year = (EXTRACT_YEAR - age).astype("int64")
    capped_birth_year = np.maximum(true_birth_year, EXTRACT_YEAR - oldest)
    age_capped = true_birth_year < EXTRACT_YEAR - oldest

    # Demographic categories are collapsed by the de-identification methodology,
    # so their permitted values come from the configuration too.
    genders = cfg.value_members(roles.GENDER)
    races = cfg.value_members(roles.RACE, [roles.RACE_WHITE, roles.RACE_BLACK,
                                           roles.RACE_ASIAN, roles.RACE_OTHER,
                                           roles.RACE_UNKNOWN])
    ethnicities = cfg.value_members(roles.ETHNICITY, [roles.ETHNICITY_NOT_HISPANIC,
                                                      roles.ETHNICITY_HISPANIC,
                                                      roles.ETHNICITY_UNKNOWN])
    gender = pick(rng, list(genders.values()), n, p=[0.552, 0.438, 0.010])
    race = pick(rng, list(races.values()), n, p=[0.66, 0.13, 0.045, 0.065, 0.10])
    ethnicity = pick(rng, list(ethnicities.values()), n, p=[0.79, 0.14, 0.07])

    active_orgs = np.flatnonzero(orgs["active"].to_numpy())
    home = rng.choice(active_orgs, size=n)
    state = orgs["state"].to_numpy()[home]
    zip3 = orgs["zip3"].to_numpy()[home]
    moved = rng.random(n) < 0.08
    state = np.where(moved, pick(rng, list(STATE_ZIP3), n), state)
    zip3 = np.array([rng.choice(STATE_ZIP3[s]) if m else z
                     for s, z, m in zip(state, zip3, moved)], dtype=object)
    zip3 = np.where([s in SMALL_POP_STATES for s in state], cfg.deid.redacted_geography, zip3)
    zip3 = np.where(rng.random(n) < 0.025, cfg.deid.redacted_geography, zip3)

    height_in = np.where(gender == genders[roles.GENDER_MALE],
                         rng.normal(69.3, 3.0, n), rng.normal(63.9, 2.9, n))
    height_in = np.where(gender == genders[roles.GENDER_UNKNOWN],
                         rng.normal(66.5, 4.0, n), height_in)
    height_in = np.where(age < 18, 20 + 2.6 * np.clip(age, 0, 17) + rng.normal(0, 2.0, n), height_in)

    # Latent BMI is heavy enough that the documented ceiling censors a visible
    # share, and the severe-obesity tail runs past the weight cap as well.
    bmi0 = np.exp(rng.normal(np.log(28.0), 0.215, n))
    severe = rng.random(n) < 0.03
    bmi0 = np.where(severe, bmi0 * rng.uniform(1.35, 1.85, n), bmi0)
    bmi0 = np.where(age < 18, np.clip(bmi0 * 0.62, 12, 38), bmi0)
    bmi_slope = rng.normal(0.16, 0.35, n)

    smoking = pick(rng, SMOKING_TRUTH, n, p=[0.50, 0.24, 0.14, 0.05, 0.07])

    adult = age >= 18
    logit = -1.6 + 0.045 * (age - 50) + 0.10 * (bmi0 - 28)
    def has(base, extra=0.0):
        p = 1.0 / (1.0 + np.exp(-(logit + base + extra)))
        return (rng.random(n) < np.where(adult, p, p * 0.08))

    conditions = {}
    conditions["obesity"] = (bmi0 >= 30) & (rng.random(n) < 0.55)
    conditions["hypertension"] = has(0.55)
    conditions["hyperlipidemia"] = has(0.35)
    conditions["diabetes"] = has(-0.55, 0.9 * conditions["obesity"])
    conditions["ckd"] = has(-1.9, 1.1 * conditions["diabetes"] + 0.6 * conditions["hypertension"])
    conditions["masld"] = has(-2.1, 1.4 * conditions["obesity"] + 0.8 * conditions["diabetes"])
    conditions["depression"] = (rng.random(n) < np.where(adult, 0.17, 0.05))
    conditions["asthma"] = (rng.random(n) < 0.09)
    conditions["copd"] = has(-3.0, 1.7 * np.isin(smoking, ["Former smoker", "Current every day smoker"]))

    has_records = ~exact_mask(n, NO_RECORD_RATE, rng)
    onboard = orgs["onboard_day"].to_numpy()[home]
    window_start = onboard + (rng.random(n) * np.maximum(EXTRACT_DAY - 30 - onboard, 1)).astype("int64")
    duration = np.clip(rng.exponential(4.0 * YEAR, n), 0.05 * YEAR, 14 * YEAR).astype("int64")
    window_end = np.minimum(window_start + duration, EXTRACT_DAY)

    patients = pd.DataFrame({
        "person_id": person_id, "capped_birth_year": capped_birth_year,
        "true_birth_year": true_birth_year, "true_age": age, "age_capped": age_capped,
        "gender": gender, "race": race, "ethnicity": ethnicity, "state": state, "zip3": zip3,
        "home_org_row": home, "height_in": height_in, "bmi0": bmi0,
        "bmi_slope": bmi_slope, "smoking": smoking, "has_records": has_records,
        "window_start": window_start, "window_end": window_end,
    })
    for name in CONDITIONS:
        patients["cond_" + name] = conditions[name]
    return patients


# --------------------------------------------------------------------------
# encounters -- everything else hangs off this table
# --------------------------------------------------------------------------

# Encounters carry a type role internally; the delivered string is resolved once,
# on the way out, so the weights below never have to name it.
ENCOUNTER_TYPE_ROLES = [roles.ENCOUNTER_IN_PERSON, roles.ENCOUNTER_REMOTE,
                        roles.ENCOUNTER_IMPORTED, roles.ENCOUNTER_EMPTY,
                        roles.ENCOUNTER_INCIDENTAL]
ENCOUNTER_TYPE_BASE = np.array([0.575, 0.090, 0.130, 0.120, 0.085])

# Per encounter type: expected findings, tests, drugs, conditions, interventions.
ENCOUNTER_YIELD = {
    roles.ENCOUNTER_IN_PERSON: (3.0, 1.15, 0.95, 1.00, 0.35),
    roles.ENCOUNTER_REMOTE: (0.80, 0.55, 0.75, 0.50, 0.15),
    roles.ENCOUNTER_IMPORTED: (1.20, 0.90, 1.30, 1.20, 0.20),
    roles.ENCOUNTER_INCIDENTAL: (0.30, 0.10, 0.12, 0.10, 0.05),
    roles.ENCOUNTER_EMPTY: (0.0, 0.0, 0.0, 0.0, 0.0),
}
# Vitals are recorded when the person is physically present.
VITAL_WEIGHT = {roles.ENCOUNTER_IN_PERSON: 1.0, roles.ENCOUNTER_REMOTE: 0.13,
                roles.ENCOUNTER_IMPORTED: 0.30, roles.ENCOUNTER_INCIDENTAL: 0.08,
                roles.ENCOUNTER_EMPTY: 0.015}


def build_visits(rng, patients, orgs, sample_practitioner):
    """Encounter rows plus the latent vitals that de-identification censors."""
    have = np.flatnonzero(patients["has_records"].to_numpy())
    start = patients["window_start"].to_numpy()[have]
    end = patients["window_end"].to_numpy()[have]
    years = np.maximum(end - start, 1) / YEAR

    n_cand = rng.poisson(3.2 * years) + 1
    pos = np.repeat(np.arange(have.size), n_cand)
    day = start[pos] + (rng.random(pos.size) * np.maximum(end - start, 1)[pos]).astype("int64")

    accept = rng.random(pos.size) < visit_intensity(day)
    counts = np.bincount(pos[accept], minlength=have.size)
    first = np.searchsorted(pos, np.arange(have.size))
    accept[first[counts == 0]] = True

    pos, day = pos[accept], day[accept]
    order = np.lexsort((day, pos))
    pos, day = pos[order], day[order]
    patient_row = have[pos]
    n = pos.size

    probs = np.tile(ENCOUNTER_TYPE_BASE, (n, 1))
    cw = covid_weight(day)
    probs[:, 1] = probs[:, 1] * (1.0 + 2.6 * cw)
    probs[:, 0] = probs[:, 0] * (1.0 - 0.18 * cw)
    probs = probs / probs.sum(axis=1, keepdims=True)
    encounter_type = np.asarray(ENCOUNTER_TYPE_ROLES, dtype=object)[categorical(rng, probs)]

    home = patients["home_org_row"].to_numpy()[patient_row]
    active = np.flatnonzero(orgs["active"].to_numpy())
    elsewhere = rng.random(n) < 0.07
    org_row = np.where(elsewhere, rng.choice(active, size=n), home)
    org_id = orgs["organization_id"].to_numpy()[org_row]
    practitioner_id, practitioner_row = sample_practitioner(org_row, rng)

    # Latent vitals; the delivered columns are the censored versions of these.
    years_since = (day - patients["window_start"].to_numpy()[patient_row]) / YEAR
    true_bmi = (patients["bmi0"].to_numpy()[patient_row]
                + patients["bmi_slope"].to_numpy()[patient_row] * years_since
                + rng.normal(0, 0.7, n))
    true_bmi = np.clip(true_bmi, 12.0, 92.0)
    height = patients["height_in"].to_numpy()[patient_row]
    true_weight = true_bmi * height ** 2 / 703.0

    visits = pd.DataFrame({
        "encounter_id": np.arange(1, n + 1, dtype="int64") * 3 + 1_000_000,
        "organization_id": org_id, "practitioner_id": practitioner_id,
        "person_id": patients["person_id"].to_numpy()[patient_row],
        "encounter_day": day, "encounter_type": encounter_type,
        "patient_row": patient_row, "org_row": org_row, "practitioner_row": practitioner_row,
        "true_bmi": true_bmi, "true_weight": true_weight, "true_height": height,
    })
    return visits


# --------------------------------------------------------------------------
# mortality: assign deaths, then truncate the record stream at the true date
# --------------------------------------------------------------------------

def assign_deaths(rng, patients, visits):
    """Mark decedents and pick the true (unshifted) death day.

    Deaths are placed inside or just after the follow-up window so that
    truncating records at the death date produces realistic right-censoring.
    """
    n = len(patients)
    age = patients["true_age"].to_numpy()
    # Tuned so decedents land in the 2-4% band the delivery shows overall.
    logit = -4.15 + 0.055 * (age - 50) + 0.6 * patients["cond_ckd"].to_numpy()
    p = 1.0 / (1.0 + np.exp(-logit))
    deceased = rng.random(n) < p

    first_visit = np.full(n, np.iinfo("int64").max, dtype="int64")
    if len(visits):
        rows = visits["patient_row"].to_numpy()
        np.minimum.at(first_visit, rows, visits["encounter_day"].to_numpy())
    first_visit = np.where(first_visit == np.iinfo("int64").max,
                           patients["window_start"].to_numpy(), first_visit)

    lo = first_visit + 1
    hi = np.maximum(patients["window_end"].to_numpy() + rng.integers(0, 900, n), lo + 1)
    hi = np.minimum(hi, EXTRACT_DAY)
    hi = np.maximum(hi, lo + 1)
    true_death_day = lo + (rng.random(n) * (hi - lo)).astype("int64")
    return deceased, true_death_day


def truncate_at_death(visits, deceased, true_death_day):
    """Drop encounters after the true death; keep at least the first."""
    if not len(visits):
        return visits
    rows = visits["patient_row"].to_numpy()
    limit = np.where(deceased[rows], true_death_day[rows], EXTRACT_DAY)
    return visits.loc[visits["encounter_day"].to_numpy() <= limit].reset_index(drop=True)


def build_mortality(cfg, rng, patients, deceased, true_death_day):
    """The mortality table holds decedents only, at a forward-shifted date."""
    f = cfg.fields(roles.DEATH)
    idx = np.flatnonzero(deceased)
    n = idx.size
    shift = rng.integers(cfg.deid.date_shift_min, cfg.deid.date_shift_max + 1, size=n)
    true_day = true_death_day[idx]
    shifted = true_day + shift
    data = {
        f.death_record_id: np.arange(1, n + 1, dtype="int64") + 5_000_000,
        f.person_id: patients["person_id"].to_numpy()[idx],
        f.deceased_indicator: np.ones(n, dtype=bool),
        f.date_of_death: as_date(shifted),
    }
    created, updated = admin_dates(shifted, rng, (0, 60), (0, 300))
    data[f.record_created], data[f.record_updated] = created, updated
    frame = finalize(cfg, roles.DEATH, data, n)
    frame = apply_fill(frame, cfg.field_specs(roles.DEATH), rng)
    truth = pd.DataFrame({
        f.person_id: patients["person_id"].to_numpy()[idx],
        "true_death_date": as_date(true_day),
        "shift_days": shift,
        "delivered_death_date": as_date(shifted),
    })
    return frame, truth


# --------------------------------------------------------------------------
# delivered tables
# --------------------------------------------------------------------------

def emit_visit(cfg, rng, visits):
    """Encounter table: de-identified vitals, with type-driven vitals gaps."""
    f = cfg.fields(roles.ENCOUNTER)
    n = len(visits)
    day = visits["encounter_day"].to_numpy()
    vt = visits["encounter_type"].to_numpy()
    type_names = cfg.value_members(roles.ENCOUNTER_TYPE)
    base_w = np.array([VITAL_WEIGHT[t] for t in vt])

    # Blood pressure is a single measurement, so systolic and diastolic share a
    # mask; weight gates BMI and BP gates pulse, matching the nesting implied by
    # the documented marginals.
    def rate(field_role):
        return rate_of(cfg, roles.ENCOUNTER, field_role)

    bp_keep = exact_mask(n, rate("systolic"), rng, base_w)
    wt_keep = exact_mask(n, rate("weight"), rng, base_w)
    bmi_keep = exact_mask(n, rate("bmi"), rng, base_w * (1.0 + 12.0 * wt_keep))
    pulse_keep = exact_mask(n, rate("pulse"), rng, base_w * (1.0 + 12.0 * bp_keep))
    resp_keep = exact_mask(n, rate("respiratory_rate"), rng, base_w * (1.0 + 4.0 * pulse_keep))
    temp_keep = exact_mask(n, rate("temperature"), rng, base_w * (1.0 + 4.0 * pulse_keep))

    low, high = cfg.deid.weight_floor, cfg.deid.weight_ceiling
    ceiling = cfg.deid.body_mass_ceiling
    weight = np.clip(visits["true_weight"].to_numpy(), low, high)
    bmi = np.minimum(visits["true_bmi"].to_numpy(), ceiling)
    systolic = np.clip(rng.normal(128, 16, n), 70, 240).round()
    diastolic = np.clip(systolic * 0.55 + rng.normal(6, 8, n), 35, 140).round()

    data = {
        f.encounter_id: visits["encounter_id"].to_numpy(),
        f.organization_id: visits["organization_id"].to_numpy(),
        f.practitioner_id: visits["practitioner_id"].to_numpy(),
        f.person_id: visits["person_id"].to_numpy(),
        f.encounter_date: as_datetime(day, rng),
        f.encounter_type: np.array([type_names[t] for t in vt], dtype=object),
        f.weight: np.where(wt_keep, weight.round(1), np.nan),
        f.bmi: np.where(bmi_keep, bmi.round(2), np.nan),
        f.systolic: systolic, f.diastolic: diastolic,
        f.pulse: np.where(pulse_keep, np.clip(rng.normal(76, 12, n), 35, 190).round(), np.nan),
        f.respiratory_rate: np.where(resp_keep, np.clip(rng.normal(16, 3, n), 6, 40).round(), np.nan),
        f.temperature: np.where(temp_keep, np.clip(rng.normal(98.2, 0.9, n), 93, 106).round(1), np.nan),
    }
    created, updated = admin_dates(day, rng)
    data[f.record_created], data[f.record_updated] = created, updated

    frame = finalize(cfg, roles.ENCOUNTER, data, n)
    frame[f.systolic] = frame[f.systolic].mask(pd.Series(~bp_keep, index=frame.index))
    frame[f.diastolic] = frame[f.diastolic].mask(pd.Series(~bp_keep, index=frame.index))
    preset = [f.weight, f.bmi, f.systolic, f.diastolic, f.pulse, f.respiratory_rate,
              f.temperature]
    frame = apply_fill(frame, cfg.field_specs(roles.ENCOUNTER), rng, preset=preset)

    truth = pd.DataFrame({
        f.encounter_id: visits["encounter_id"].to_numpy(),
        f.person_id: visits["person_id"].to_numpy(),
        f.encounter_date: as_date(day),
        "true_height_in": visits["true_height"].to_numpy().round(2),
        "true_weight_lb": visits["true_weight"].to_numpy().round(2),
        "true_bmi": visits["true_bmi"].to_numpy().round(3),
        "weight_capped": ((visits["true_weight"].to_numpy() < low)
                         | (visits["true_weight"].to_numpy() > high)),
        "bmi_capped": visits["true_bmi"].to_numpy() > ceiling,
        "vitals_recorded": wt_keep | bmi_keep | bp_keep,
    })
    return frame, truth


def expand_visits(rng, visits, which, scale=1.0):
    """Encounter row indices repeated by a Poisson draw on that type's yield."""
    lam = np.array([ENCOUNTER_YIELD[t][which]
                    for t in visits["encounter_type"].to_numpy()]) * scale
    return np.repeat(np.arange(len(visits)), rng.poisson(lam))


def emit_observation(cfg, rng, visits, rows, finding_cat):
    f = cfg.fields(roles.FINDING)
    n = rows.size
    day = visits["encounter_day"].to_numpy()[rows]
    kind = categorical(rng, np.tile([0.22, 0.55, 0.23], (n, 1)))  # vital / qualitative / other

    vital_idx = rng.integers(0, len(FINDING_VITALS), size=n)
    vital_name = np.array([FINDING_VITALS[i][0] for i in range(len(FINDING_VITALS))], dtype=object)[vital_idx]
    vital_loinc = np.array([FINDING_VITALS[i][1] for i in range(len(FINDING_VITALS))], dtype=object)[vital_idx]
    vital_units = np.array([FINDING_VITALS[i][2] for i in range(len(FINDING_VITALS))], dtype=object)[vital_idx]
    vital_mu = np.array([FINDING_VITALS[i][3] for i in range(len(FINDING_VITALS))])[vital_idx]
    vital_sd = np.array([FINDING_VITALS[i][4] for i in range(len(FINDING_VITALS))])[vital_idx]

    name = np.where(kind == 0, vital_name, pick(rng, FINDING_QUAL, n))
    name = np.where(kind == 2, pick(rng, FINDING_QUAL + list(vital_name[:1]), n), name)
    vital_labels = list(cfg.value_members(
        roles.FINDING_CATEGORY,
        [roles.FINDING_VITAL, roles.FINDING_VITAL_VARIANT, roles.FINDING_EXAM]).values())
    category = np.where(kind == 0, pick(rng, vital_labels, n, p=[0.6, 0.25, 0.15]),
                        pick(rng, finding_cat, n))

    data = {
        f.finding_id: np.arange(1, n + 1, dtype="int64") + 20_000_000,
        f.organization_id: visits["organization_id"].to_numpy()[rows],
        f.practitioner_id: visits["practitioner_id"].to_numpy()[rows],
        f.person_id: visits["person_id"].to_numpy()[rows],
        f.encounter_id: visits["encounter_id"].to_numpy()[rows],
        f.event_date: as_datetime(day, rng),
        f.finding_name: name,
        f.finding_category: category,
        f.loinc: np.where(kind == 0, vital_loinc, pick(rng, ["8302-2", "72166-2", "44261-6", "59408-5"], n)),
        f.snomed: pick(rng, ["229819007", "266919005", "8517006", "160603005", "365981007"], n),
        f.cpt: pick(rng, ["99213", "96127", "97802", "99406"], n),
        f.hcpcs: pick(rng, ["G8427", "G0444", "G8510"], n),
        f.result_text: pick(rng, FINDING_QUAL_VALUES, n),
        f.result_snomed: pick(rng, ["17621005", "263654008", "260385009"], n),
        f.result_numeric: np.round(rng.normal(vital_mu, vital_sd), 1),
        f.units: vital_units,
        f.reason: pick(rng, ["Routine screening", "Follow-up", "Patient request"], n),
        f.coded_reason: pick(rng, ["185349003", "390906007"], n),
    }
    created, updated = admin_dates(day, rng)
    data[f.record_created], data[f.record_updated] = created, updated

    frame = finalize(cfg, roles.FINDING, data, n)
    weights = {
        f.result_numeric: np.where(kind == 0, 1.0, 0.02),
        f.units: np.where(kind == 0, 1.0, 0.01),
        f.result_text: np.where(kind == 1, 1.0, 0.15),
        f.loinc: np.where(kind == 0, 1.0, 0.25),
    }
    return apply_fill(frame, cfg.field_specs(roles.FINDING), rng, weights=weights)


def emit_lab(cfg, rng, visits, rows, patients):
    """Test rows; the half without a LOINC carry messier free-text test names."""
    f = cfg.fields(roles.TEST_RESULT)
    n = rows.size
    day = visits["encounter_day"].to_numpy()[rows] + rng.integers(-2, 4, size=n)
    day = np.clip(day, START_DAY, EXTRACT_DAY)
    patient_row = visits["patient_row"].to_numpy()[rows]

    catalog_idx = rng.integers(0, len(NUMERIC_TESTS), size=n)
    is_qual = rng.random(n) < 0.09
    qual_idx = rng.integers(0, len(QUALITATIVE_TESTS), size=n)

    def col(pos, source=NUMERIC_TESTS, dtype=object):
        return np.array([row[pos] for row in source], dtype=dtype)

    panel = col(0)[catalog_idx]
    test = col(1)[catalog_idx]
    loinc = col(2)[catalog_idx]
    units = col(3)[catalog_idx]
    mean = col(4, dtype="float64")[catalog_idx]
    sd = col(5, dtype="float64")[catalog_idx]
    lognormal = col(6, dtype=bool)[catalog_idx]
    low = col(7, dtype="float64")[catalog_idx]
    high = col(8, dtype="float64")[catalog_idx]
    condition = col(9)[catalog_idx]
    shift = col(10, dtype="float64")[catalog_idx]

    # Test choice tracks the person's conditions, and results shift with them.
    affected = np.zeros(n, dtype=bool)
    for name in CONDITIONS:
        flag = patients["cond_" + name].to_numpy()[patient_row]
        affected |= flag & (condition == name)
    value = np.where(lognormal,
                     np.exp(rng.normal(np.log(np.maximum(mean, 1e-6)), sd)),
                     rng.normal(mean, sd))
    value = np.clip(value + affected * shift, low, high)
    value = np.round(value, 2)

    qual_panel = np.array([r[0] for r in QUALITATIVE_TESTS], dtype=object)[qual_idx]
    qual_test = np.array([r[1] for r in QUALITATIVE_TESTS], dtype=object)[qual_idx]
    qual_loinc = np.array([r[2] for r in QUALITATIVE_TESTS], dtype=object)[qual_idx]
    qual_result = np.array([rng.choice(QUALITATIVE_TESTS[i][3], p=QUALITATIVE_TESTS[i][4])
                            for i in qual_idx], dtype=object)

    panel = np.where(is_qual, qual_panel, panel)
    canonical = np.where(is_qual, qual_test, test)
    loinc = np.where(is_qual, qual_loinc, loinc)

    # Rows without a LOINC carry the messy free-text spellings, which is what
    # forces text-normalisation code to earn its keep.
    has_loinc = exact_mask(n, rate_of(cfg, roles.TEST_RESULT, "loinc"), rng)
    messy_pool = [QUALITATIVE_TESTS[i][5] if q else NUMERIC_TESTS[c][11]
                  for i, c, q in zip(qual_idx, catalog_idx, is_qual)]
    draw = rng.integers(0, 100, size=n)
    messy = np.array([pool[j % len(pool)] for pool, j in zip(messy_pool, draw)], dtype=object)
    test_text = np.where(has_loinc, canonical, messy)
    full_name = np.where(pd.isna(panel), test_text,
                         np.array([f"{p} - {t}" for p, t in zip(panel, test_text)], dtype=object))

    ref_lo, ref_hi = np.round(mean * 0.78, 1), np.round(mean * 1.22, 1)
    ref = np.array([f"{lo:g} - {hi:g}" for lo, hi in zip(ref_lo, ref_hi)], dtype=object)
    ref = np.where(is_qual, "Negative", ref)
    abnormal = np.where(is_qual, qual_result != "Negative",
                        (value < ref_lo) | (value > ref_hi))

    data = {
        f.test_result_id: np.arange(1, n + 1, dtype="int64") + 30_000_000,
        f.organization_id: visits["organization_id"].to_numpy()[rows],
        f.ordering_practitioner_id: visits["practitioner_id"].to_numpy()[rows],
        f.person_id: visits["person_id"].to_numpy()[rows],
        f.order_date: as_datetime(day - 1, rng),
        f.specimen_date: as_datetime(day, rng),
        f.report_date: as_datetime(day + 1, rng),
        f.test_description: full_name, f.panel: panel, f.test: test_text, f.loinc: loinc,
        f.cpt: pick(rng, ["80053", "80061", "85025", "83036", "84443", "81003"], n),
        f.hcpcs: pick(rng, ["G0480", "G0483"], n),
        f.snomed: pick(rng, ["104177005", "271062006"], n),
        f.result_text: qual_result,
        f.result_snomed: pick(rng, ["260385009", "10828004", "82334004"], n),
        f.result_modifier: pick(rng, ["<", ">", "<=", ">="], n),
        f.result_numeric: value, f.units: units, f.reference_range: ref,
        f.abnormal_flag: abnormal,
        f.final_flag: rng.random(n) < 0.93,
        f.order_flag: np.ones(n, dtype=bool),
        f.result_flag: rng.random(n) < 0.997,
    }
    created, updated = admin_dates(day, rng)
    data[f.record_created], data[f.record_updated] = created, updated

    frame = finalize(cfg, roles.TEST_RESULT, data, n)
    frame[f.loinc] = frame[f.loinc].mask(pd.Series(~has_loinc, index=frame.index))
    in_panel = ~pd.isna(panel)
    weights = {
        f.panel: np.where(in_panel, 1.0, 0.01),
        f.result_numeric: np.where(is_qual, 0.03, 1.0),
        f.result_text: np.where(is_qual, 1.0, 0.02),
        f.units: np.where(is_qual, 0.05, 1.0),
        f.reference_range: np.where(is_qual, 0.3, 1.0),
        f.abnormal_flag: np.where(is_qual, 0.4, 1.0),
        f.result_modifier: np.where(is_qual, 0.05, 1.0),
    }
    frame = apply_fill(frame, cfg.field_specs(roles.TEST_RESULT), rng, weights=weights,
                       preset=[f.loinc])
    return frame, patient_row


def emit_medication(cfg, rng, visits, rows, patients, prescription_actions, activity_types):
    f = cfg.fields(roles.DRUG_EXPOSURE)
    n = rows.size
    day = visits["encounter_day"].to_numpy()[rows]
    patient_row = visits["patient_row"].to_numpy()[rows]

    # Prefer drugs that match the person's conditions; fall back to anything.
    drug_class = np.array([d[7] for d in DRUG_CATALOG], dtype=object)
    idx = rng.integers(0, len(DRUG_CATALOG), size=n)
    for _ in range(3):
        cls = drug_class[idx]
        ok = np.isin(cls, ["general", "otc", "acute"])
        for name in CONDITIONS:
            ok |= (cls == name) & patients["cond_" + name].to_numpy()[patient_row]
        ok |= (cls == "glp1") & (patients["cond_diabetes"].to_numpy()[patient_row]
                                 | patients["cond_obesity"].to_numpy()[patient_row])
        idx = np.where(ok, idx, rng.integers(0, len(DRUG_CATALOG), size=n))

    def col(pos):
        return np.array([d[pos] for d in DRUG_CATALOG], dtype=object)[idx]

    action = pick(rng, activity_types, n, p=[0.13, 0.05, 0.52, 0.30])
    quantity = pick(rng, ["30", "60", "90", "1", "12", "20", "100"], n)
    data = {
        f.drug_exposure_id: np.arange(1, n + 1, dtype="int64") + 40_000_000,
        f.organization_id: visits["organization_id"].to_numpy()[rows],
        f.administering_practitioner_id: visits["practitioner_id"].to_numpy()[rows],
        f.prescribing_practitioner_id: visits["practitioner_id"].to_numpy()[rows],
        f.documenting_practitioner_id: visits["practitioner_id"].to_numpy()[rows],
        f.person_id: visits["person_id"].to_numpy()[rows],
        f.encounter_id: visits["encounter_id"].to_numpy()[rows],
        f.prescription_date: as_datetime(day, rng),
        f.begin_date: as_datetime(day, rng),
        f.end_date: as_datetime(np.minimum(day + rng.integers(14, 900, n), EXTRACT_DAY), rng),
        f.administration_date: as_datetime(day, rng),
        f.documentation_date: as_datetime(day, rng),
        f.drug_name: col(0), f.ndc: col(1), f.rxnorm_code: col(2),
        f.hcpcs: pick(rng, ["J1815", "J3490", "S0630"], n),
        f.prescription_action: pick(rng, prescription_actions, n),
        f.dose: pick(rng, ["1", "2", "0.5", "1.5", "3"], n),
        f.units: col(4), f.strength: col(3), f.form: col(5), f.route: col(6),
        f.quantity: quantity,
        f.frequency: pick(rng, DRUG_FREQUENCY, n),
        f.daily_frequency: pick(rng, ["1", "2", "3", "4"], n, p=[0.55, 0.30, 0.10, 0.05]),
        f.daily_amount: pick(rng, ["1", "2", "4"], n),
        f.duration_days: pick(rng, [10, 14, 30, 60, 90], n).astype("int64"),
        f.refills: pick(rng, [0, 1, 2, 3, 5, 11], n, p=[0.30, 0.15, 0.15, 0.20, 0.15, 0.05]).astype("int64"),
        f.dispense_as_written_flag: rng.random(n) < 0.11,
        f.over_the_counter_flag: np.array([c == "otc" for c in drug_class[idx]]),
        f.electronic_order_flag: rng.random(n) < 0.72,
        f.activity_type: action,
        f.sig: pick(rng, DRUG_SIGS, n),
        f.stop_reason: pick(rng, DISCONTINUE_REASONS, n),
    }
    created, updated = admin_dates(day, rng)
    data[f.record_created], data[f.record_updated] = created, updated

    frame = finalize(cfg, roles.DRUG_EXPOSURE, data, n)
    given = action == cfg.value_member(roles.ACTIVITY_TYPE, roles.ACTIVITY_ADMINISTRATION)
    ordered = action == cfg.value_member(roles.ACTIVITY_TYPE, roles.ACTIVITY_PRESCRIPTION)
    stopped = action == cfg.value_member(roles.ACTIVITY_TYPE, roles.ACTIVITY_DISCONTINUATION)
    weights = {
        f.administration_date: np.where(given, 1.0, 0.01),
        f.administering_practitioner_id: np.where(given, 1.0, 0.01),
        f.prescription_date: np.where(ordered, 1.0, 0.05),
        f.prescribing_practitioner_id: np.where(ordered, 1.0, 0.05),
        f.end_date: np.where(stopped, 1.0, 0.25),
        f.stop_reason: np.where(stopped, 1.0, 0.01),
        f.over_the_counter_flag: np.where(data[f.over_the_counter_flag], 1.0, 0.02),
        f.sig: np.where(ordered, 1.0, 0.2),
    }
    return (apply_fill(frame, cfg.field_specs(roles.DRUG_EXPOSURE), rng, weights=weights),
            patient_row)


def emit_problem(cfg, rng, visits, rows, patients):
    f = cfg.fields(roles.CONDITION)
    n = rows.size
    day = visits["encounter_day"].to_numpy()[rows]
    patient_row = visits["patient_row"].to_numpy()[rows]

    # Chronic conditions for people who have them, acute ones otherwise.
    cond_matrix = np.column_stack([patients["cond_" + c].to_numpy()[patient_row] for c in CONDITIONS])
    weight = cond_matrix.astype("float64") + 0.02
    chosen = categorical(rng, weight / weight.sum(axis=1, keepdims=True))
    use_chronic = cond_matrix[np.arange(n), chosen] & (rng.random(n) < 0.72)

    acute_code = np.array([c for c, _ in ACUTE_CONDITIONS], dtype=object)
    acute_name = np.array([m for _, m in ACUTE_CONDITIONS], dtype=object)
    pick_acute = rng.integers(0, len(ACUTE_CONDITIONS), size=n)

    code = np.empty(n, dtype=object)
    name = np.empty(n, dtype=object)
    for j, cond in enumerate(CONDITIONS):
        sel = use_chronic & (chosen == j)
        if not sel.any():
            continue
        codes, names = CONDITION_CODES[cond]
        code[sel] = pick(rng, codes, sel.sum())
        name[sel] = pick(rng, names, sel.sum())
    code = np.where(use_chronic, code, acute_code[pick_acute])
    name = np.where(use_chronic, name, acute_name[pick_acute])

    onset = day - np.where(use_chronic, rng.integers(0, 3000, n), rng.integers(0, 30, n))
    data = {
        f.condition_id: np.arange(1, n + 1, dtype="int64") + 50_000_000,
        f.organization_id: visits["organization_id"].to_numpy()[rows],
        f.practitioner_id: visits["practitioner_id"].to_numpy()[rows],
        f.person_id: visits["person_id"].to_numpy()[rows],
        f.encounter_id: visits["encounter_id"].to_numpy()[rows],
        f.begin_date: as_datetime(np.maximum(onset, START_DAY - 4000), rng),
        f.end_date: as_datetime(np.minimum(day + rng.integers(1, 700, n), EXTRACT_DAY), rng),
        f.condition_name: name,
    }
    created, updated = admin_dates(day, rng, (0, 1), (0, 500))
    data[f.record_created], data[f.record_updated] = created, updated
    frame = finalize(cfg, roles.CONDITION, data, n)
    weights = {f.end_date: np.where(use_chronic, 0.05, 1.0)}
    frame = apply_fill(frame, cfg.field_specs(roles.CONDITION), rng, weights=weights)
    return frame, patient_row, code


def emit_problem_code(cfg, rng, problems, problem_codes, histories):
    """One code row per condition (sometimes two) plus a background-linked minority.

    Exactly one of the two parent links is populated per row, and the split
    reproduces the two documented fill rates.
    """
    f = cfg.fields(roles.CONDITION_CODE)
    n_prob = len(problems)
    extra = exact_mask(n_prob, 0.14, rng)
    prob_pos = np.concatenate([np.arange(n_prob), np.flatnonzero(extra)])
    n_linked = prob_pos.size
    n_hist = int(round(n_linked * cfg.fill_rate(roles.CONDITION_CODE, "background_id")
                       / cfg.fill_rate(roles.CONDITION_CODE, "condition_id")))
    n_hist = min(n_hist, len(histories))
    hist_pos = rng.choice(len(histories), size=n_hist, replace=False) if n_hist else np.array([], dtype="int64")

    n = n_linked + n_hist
    condition_id = np.concatenate([problems[cfg.field(roles.CONDITION, "condition_id")]
                                   .to_numpy()[prob_pos], np.zeros(n_hist, dtype="int64")])
    background_id = np.concatenate([
        np.zeros(n_linked, dtype="int64"),
        histories[cfg.field(roles.CLINICAL_BACKGROUND, "background_id")].to_numpy()[hist_pos]])
    link_is_condition = np.concatenate([np.ones(n_linked, bool), np.zeros(n_hist, bool)])

    icd10 = np.concatenate([problem_codes[prob_pos], pick(rng, ["Z87.891", "Z72.0", "F17.210"], n_hist)])
    kind = categorical(rng, np.tile([0.83, 0.05, 0.10, 0.02], (n, 1)))
    code = np.where(kind == 0, icd10, pick(rng, SNOMED_CONDITION_CODES, n))
    code = np.where(kind == 1, pick(rng, ICD9_CODES, n), code)
    code = np.where(kind == 3, pick(rng, CPT_CONDITION_CODES, n), code)
    code_type = np.asarray(list(cfg.value_members(roles.CODE_TYPE, [
        roles.CODE_TYPE_ICD10, roles.CODE_TYPE_ICD9, roles.CODE_TYPE_SNOMED,
        roles.CODE_TYPE_CPT]).values()), dtype=object)[kind]

    # De-identification blanks the codes that make re-identification easy.
    redacted = exact_mask(n, 0.006, rng)
    code = np.where(redacted, cfg.deid.redacted_code, code)

    data = {
        f.condition_code_id: np.arange(1, n + 1, dtype="int64") + 60_000_000,
        f.condition_id: condition_id, f.background_id: background_id,
        f.code: code, f.code_type: code_type,
    }
    frame = finalize(cfg, roles.CONDITION_CODE, data, n)
    frame[f.condition_id] = frame[f.condition_id].mask(
        pd.Series(~link_is_condition, index=frame.index))
    frame[f.background_id] = frame[f.background_id].mask(
        pd.Series(link_is_condition, index=frame.index))
    return apply_fill(frame, cfg.field_specs(roles.CONDITION_CODE), rng,
                      preset=[f.condition_id, f.background_id])


def emit_history(cfg, rng, visits, patients, smoking_categories, other_categories):
    """Background rows, including the smoking-status mess the dictionary describes.

    Many background categories are annotated as being used for smoking-status
    documentation, and the status itself lives in the free-text name. Recovering
    a person's smoking status therefore means parsing that text.
    """
    f = cfg.fields(roles.CLINICAL_BACKGROUND)
    n = max(1, int(len(visits) * 0.32))
    rows = rng.choice(len(visits), size=n)
    day = visits["encounter_day"].to_numpy()[rows]
    patient_row = visits["patient_row"].to_numpy()[rows]
    smoking_truth = patients["smoking"].to_numpy()[patient_row]

    is_smoking = rng.random(n) < 0.45
    smoke_text = np.array([SMOKING_TEXT[s][rng.integers(len(SMOKING_TEXT[s]))]
                           for s in smoking_truth], dtype=object)

    category_names = cfg.value_members(roles.BACKGROUND_CATEGORY)
    other_idx = rng.integers(0, len(NONSMOKING_BACKGROUND), size=n)
    other_cat = np.array([category_names[NONSMOKING_BACKGROUND[i][0]] for i in other_idx],
                         dtype=object)
    other_name = np.array([NONSMOKING_BACKGROUND[i][1][rng.integers(len(NONSMOKING_BACKGROUND[i][1]))]
                           for i in other_idx], dtype=object)

    # A minority of non-smoking rows use the other dictionary categories verbatim.
    other_cat = np.where(rng.random(n) < 0.20, pick(rng, other_categories, n), other_cat)
    category = np.where(is_smoking, pick(rng, smoking_categories, n), other_cat)
    name = np.where(is_smoking, smoke_text, other_name)

    data = {
        f.background_id: np.arange(1, n + 1, dtype="int64") + 70_000_000,
        f.organization_id: visits["organization_id"].to_numpy()[rows],
        f.practitioner_id: visits["practitioner_id"].to_numpy()[rows],
        f.person_id: visits["person_id"].to_numpy()[rows],
        f.begin_date: as_datetime(day - rng.integers(0, 4000, n), rng),
        f.end_date: as_datetime(day, rng),
        f.background_name: name, f.background_category: category,
        f.negation_flag: rng.random(n) < 0.5,
    }
    created, updated = admin_dates(day, rng, (0, 1), (0, 400))
    data[f.record_created], data[f.record_updated] = created, updated
    frame = finalize(cfg, roles.CLINICAL_BACKGROUND, data, n)
    frame = apply_fill(frame, cfg.field_specs(roles.CLINICAL_BACKGROUND), rng)
    return frame, patient_row


def emit_procedure(cfg, rng, visits, rows, intervention_categories):
    f = cfg.fields(roles.INTERVENTION)
    n = rows.size
    day = visits["encounter_day"].to_numpy()[rows]
    idx = rng.integers(0, len(INTERVENTIONS), size=n)
    name = np.array([p[0] for p in INTERVENTIONS], dtype=object)[idx]
    cpt = np.array([p[1] for p in INTERVENTIONS], dtype=object)[idx]
    category_names = cfg.value_members(roles.INTERVENTION_CATEGORY)
    category = np.array([category_names[p[2]] for p in INTERVENTIONS], dtype=object)[idx]
    category = np.where(rng.random(n) < 0.15, pick(rng, intervention_categories, n), category)

    data = {
        f.intervention_id: np.arange(1, n + 1, dtype="int64") + 80_000_000,
        f.organization_id: visits["organization_id"].to_numpy()[rows],
        f.performing_practitioner_id: visits["practitioner_id"].to_numpy()[rows],
        f.ordering_practitioner_id: visits["practitioner_id"].to_numpy()[rows],
        f.person_id: visits["person_id"].to_numpy()[rows],
        f.order_date: as_datetime(day, rng),
        f.event_date: as_datetime(day + rng.integers(0, 30, n), rng),
        f.result_date: as_datetime(day + rng.integers(0, 35, n), rng),
        f.intervention_name: name, f.intervention_category: category, f.cpt: cpt,
        f.hcpcs: pick(rng, ["G0121", "G0439", "G0442"], n),
        f.loinc: pick(rng, ["24627-2", "34534-8", "18744-3"], n),
        f.snomed: pick(rng, ["73761001", "168731009", "40701008"], n),
        f.comment: pick(rng, ["No acute findings.", "Within normal limits.",
                              "Result: 4.2", "See scanned report.",
                              "Mild degenerative changes noted.", "Negative for malignancy"], n),
        f.units: pick(rng, ["mm", "cm", "%", "score"], n),
        f.reason: pick(rng, ["Screening", "Diagnostic", "Follow-up"], n),
        f.coded_reason: pick(rng, ["171121004", "410429000"], n),
    }
    created, updated = admin_dates(day, rng, (0, 2), (0, 400))
    data[f.record_created], data[f.record_updated] = created, updated
    frame = finalize(cfg, roles.INTERVENTION, data, n)
    frame = apply_fill(frame, cfg.field_specs(roles.INTERVENTION), rng)
    return frame, visits["patient_row"].to_numpy()[rows]


def emit_referral(cfg, rng, visits):
    f = cfg.fields(roles.SERVICE_REQUEST)
    n = max(1, int(len(visits) * 0.18))
    rows = rng.choice(len(visits), size=n)
    day = visits["encounter_day"].to_numpy()[rows]
    data = {
        f.service_request_id: np.arange(1, n + 1, dtype="int64") + 90_000_000,
        f.organization_id: visits["organization_id"].to_numpy()[rows],
        f.practitioner_id: visits["practitioner_id"].to_numpy()[rows],
        f.person_id: visits["person_id"].to_numpy()[rows],
        f.completion_date: as_datetime(day + rng.integers(1, 120, n), rng),
        f.service_request_name: pick(rng, SERVICE_REQUEST_NAMES, n),
        f.snomed: pick(rng, ["306253008", "183856001", "306110002"], n),
        f.cpt: pick(rng, ["99242", "99243", "99244"], n),
        f.hcpcs: pick(rng, ["S0620", "G0463"], n),
        f.loinc: pick(rng, ["11488-4"], n),
        f.reason: pick(rng, ["Evaluation and management", "Second opinion",
                             "Specialist evaluation", "Diagnostic workup"], n),
        f.coded_reason: pick(rng, ["308292007", "183519002"], n),
    }
    created, updated = admin_dates(day, rng, (0, 1), (0, 300))
    data[f.record_created], data[f.record_updated] = created, updated
    frame = finalize(cfg, roles.SERVICE_REQUEST, data, n)
    frame = apply_fill(frame, cfg.field_specs(roles.SERVICE_REQUEST), rng)
    return frame, visits["patient_row"].to_numpy()[rows]


def emit_immunization(cfg, rng, visits, vaccination_statuses):
    f = cfg.fields(roles.VACCINATION)
    n = max(1, int(len(visits) * 0.20))
    rows = rng.choice(len(visits), size=n)
    day = visits["encounter_day"].to_numpy()[rows]

    # COVID-19 products cannot predate December 2020.
    covid_ok = day >= int(np.datetime64("2020-12-01", "D").astype("int64"))
    idx = rng.integers(0, len(VACCINES), size=n)
    is_covid = np.array([VACCINES[i][3] for i in idx])
    while (is_covid & ~covid_ok).any():
        bad = is_covid & ~covid_ok
        idx = np.where(bad, rng.integers(0, len(VACCINES), size=n), idx)
        is_covid = np.array([VACCINES[i][3] for i in idx])

    data = {
        f.vaccination_id: np.arange(1, n + 1, dtype="int64") + 110_000_000,
        f.organization_id: visits["organization_id"].to_numpy()[rows],
        f.practitioner_id: visits["practitioner_id"].to_numpy()[rows],
        f.person_id: visits["person_id"].to_numpy()[rows],
        f.administration_date: as_datetime(day, rng),
        f.vaccine_name: np.array([VACCINES[i][0] for i in idx], dtype=object),
        f.cvx: np.array([VACCINES[i][1] for i in idx], dtype=object),
        f.rxnorm_code: pick(rng, ["1594660", "798304", "2468231", "1657001"], n),
        f.ndc: pick(rng, ["49281041450", "58160081052", "00006402302"], n),
        f.snomed: pick(rng, ["836398006", "871908002"], n),
        f.cpt: np.array([VACCINES[i][2] for i in idx], dtype=object),
        f.hcpcs: pick(rng, ["G0008", "G0009", "M0201"], n),
        f.vaccination_status: pick(rng, vaccination_statuses, n),
        f.dose: pick(rng, ["0.5", "0.3", "1", "0.25"], n),
        f.route: pick(rng, ["Intramuscular", "Subcutaneous", "Intranasal", "Intradermal"], n),
        f.series: pick(rng, ["1 of 2", "2 of 2", "Dose 1", "Dose 2", "Booster", "3 of 3"], n),
        f.manufacturer: pick(rng, VACCINE_MANUFACTURERS, n),
        f.mvx: pick(rng, MVX_CODES, n),
        f.refusal_reason: pick(rng, ["Patient objection", "Allergy", "Medical contraindication"], n),
        f.historical_flag: rng.random(n) < 0.5,
    }
    created, updated = admin_dates(day, rng)
    data[f.record_created], data[f.record_updated] = created, updated
    frame = finalize(cfg, roles.VACCINATION, data, n)
    frame = apply_fill(frame, cfg.field_specs(roles.VACCINATION), rng)
    return frame, visits["patient_row"].to_numpy()[rows]


def emit_allergy(cfg, rng, visits):
    f = cfg.fields(roles.INTOLERANCE)
    n = max(1, int(len(visits) * 0.10))
    rows = rng.choice(len(visits), size=n)
    day = visits["encounter_day"].to_numpy()[rows]
    data = {
        f.intolerance_id: np.arange(1, n + 1, dtype="int64") + 120_000_000,
        f.organization_id: visits["organization_id"].to_numpy()[rows],
        f.practitioner_id: visits["practitioner_id"].to_numpy()[rows],
        f.person_id: visits["person_id"].to_numpy()[rows],
        f.intolerance_name: pick(rng, ALLERGENS, n),
        f.snomed: pick(rng, ["91936005", "294505008", "300916003"], n),
        f.ndc: pick(rng, ["00093310901", "00054001825"], n),
        f.rxnorm_code: pick(rng, ["7980", "10180", "2670"], n),
        f.intolerance_type: pick(rng, INTOLERANCE_TYPES, n),
        f.resolution_flag: rng.random(n) < 0.5,
    }
    created, updated = admin_dates(day, rng, (0, 1), (0, 500))
    data[f.record_created], data[f.record_updated] = created, updated
    frame = finalize(cfg, roles.INTOLERANCE, data, n)
    frame = apply_fill(frame, cfg.field_specs(roles.INTOLERANCE), rng)
    return frame, visits["patient_row"].to_numpy()[rows], day


def emit_allergy_reaction(cfg, rng, allergies, allergy_days):
    f = cfg.fields(roles.INTOLERANCE_REACTION)
    counts = rng.integers(1, 3, size=len(allergies))
    pos = np.repeat(np.arange(len(allergies)), counts)
    n = pos.size
    day = allergy_days[pos]
    data = {
        f.reaction_id: np.arange(1, n + 1, dtype="int64") + 130_000_000,
        f.intolerance_id: allergies[cfg.field(roles.INTOLERANCE, "intolerance_id")].to_numpy()[pos],
        f.reaction_name: pick(rng, INTOLERANCE_REACTIONS, n),
        f.reaction_snomed: pick(rng, ["247472004", "271807003", "422587007"], n),
        f.severity: pick(rng, REACTION_SEVERITY, n),
    }
    created, updated = admin_dates(day, rng, (0, 1), (0, 500))
    data[f.record_created], data[f.record_updated] = created, updated
    frame = finalize(cfg, roles.INTOLERANCE_REACTION, data, n)
    return apply_fill(frame, cfg.field_specs(roles.INTOLERANCE_REACTION), rng)


# --------------------------------------------------------------------------
# optional back-links to the condition table
# --------------------------------------------------------------------------

class ConditionIndex:
    """Lets a child row draw a condition key belonging to its own person."""

    def __init__(self, problem_patient_rows, problem_ids, n_patients):
        order = np.argsort(problem_patient_rows, kind="stable")
        self.rows = np.asarray(problem_patient_rows)[order]
        self.ids = np.asarray(problem_ids, dtype="int64")[order]
        keys = np.arange(n_patients)
        self.start = np.searchsorted(self.rows, keys, side="left")
        self.count = np.searchsorted(self.rows, keys, side="right") - self.start

    def attach(self, frame, field, rate, patient_rows, rng):
        """Populate ``field`` at the documented rate, patients permitting."""
        n = len(frame)
        eligible = self.count[patient_rows] > 0
        keep = exact_mask(n, rate, rng, weights=np.where(eligible, 1.0, 0.0)) & eligible
        offset = (rng.random(n) * np.maximum(self.count[patient_rows], 1)).astype("int64")
        # People with no conditions index past the end; ``keep`` discards them.
        slot = np.minimum(self.start[patient_rows] + offset, max(self.ids.size - 1, 0))
        values = self.ids[slot] if self.ids.size else np.zeros(n, dtype="int64")
        col = pd.Series(pd.array(values, dtype="Int64"), index=frame.index)
        frame[field] = col.mask(pd.Series(~keep, index=frame.index))
        return frame


# --------------------------------------------------------------------------
# activity dates, computed from the delivered (already masked) records
# --------------------------------------------------------------------------

ACTIVITY_DATE_ROLES = {
    roles.ENCOUNTER: ["encounter_date"],
    roles.DRUG_EXPOSURE: ["administration_date", "prescription_date", "documentation_date"],
    roles.FINDING: ["event_date"],
    roles.TEST_RESULT: ["report_date", "order_date"],
    roles.VACCINATION: ["administration_date"],
    roles.INTOLERANCE: ["record_created"],
    roles.CONDITION: ["record_created"],
    roles.CLINICAL_BACKGROUND: ["record_created"],
    roles.INTERVENTION: ["order_date", "event_date", "result_date"],
    roles.SERVICE_REQUEST: ["record_created"],
}
# A practitioner is keyed by (organization, practitioner); each table names the
# practitioner column differently.
PRACTITIONER_KEY_ROLES = {
    roles.ENCOUNTER: ["practitioner_id"],
    roles.DRUG_EXPOSURE: ["administering_practitioner_id", "prescribing_practitioner_id",
                          "documenting_practitioner_id"],
    roles.FINDING: ["practitioner_id"], roles.TEST_RESULT: ["ordering_practitioner_id"],
    roles.VACCINATION: ["practitioner_id"], roles.INTOLERANCE: ["practitioner_id"],
    roles.CONDITION: ["practitioner_id"], roles.CLINICAL_BACKGROUND: ["practitioner_id"],
    roles.INTERVENTION: ["performing_practitioner_id", "ordering_practitioner_id"],
    roles.SERVICE_REQUEST: ["practitioner_id"],
}


def _row_span(frame, cols):
    cols = [c for c in cols if c in frame.columns]
    sub = frame[cols]
    return sub.min(axis=1), sub.max(axis=1)


def activity_spans(cfg, tables, key_of):
    """min/max clinical date per key, over the fields each table contributes.

    ``key_of(table_role, frame)`` returns a single key Series (or None to skip
    the table). Spans are computed from the delivered, already-masked columns so
    the activity dates agree with what an analyst would recompute.
    """
    blocks = []
    for role, date_roles in ACTIVITY_DATE_ROLES.items():
        frame = tables[role]
        if not len(frame):
            continue
        key = key_of(role, frame)
        if key is None:
            continue
        lo, hi = _row_span(frame, [cfg.field(role, r) for r in date_roles])
        block = pd.DataFrame({"key": key.to_numpy(), "lo": lo.to_numpy(), "hi": hi.to_numpy()})
        blocks.append(block.dropna())
    if not blocks:
        return pd.Series(dtype="datetime64[us]"), pd.Series(dtype="datetime64[us]")
    allrows = pd.concat(blocks, ignore_index=True)
    agg = allrows.groupby("key").agg(lo=("lo", "min"), hi=("hi", "max"))
    return agg["lo"], agg["hi"]


def emit_patient(cfg, rng, patients, spans):
    f = cfg.fields(roles.PERSON)
    n = len(patients)
    low, high = spans
    key = patients["person_id"].to_numpy()
    earliest = low.reindex(key).to_numpy()
    latest = high.reindex(key).to_numpy()
    created = np.minimum(patients["window_start"].to_numpy() - rng.integers(0, 400, n), EXTRACT_DAY)

    data = {
        f.person_id: key,
        # The birth column is a date holding 1 January of the (capped) year.
        f.birth_date: (patients["capped_birth_year"].to_numpy() - 1970
                       ).astype("datetime64[Y]").astype("datetime64[us]"),
        f.gender: patients["gender"].to_numpy(),
        f.race: patients["race"].to_numpy(),
        f.ethnicity: patients["ethnicity"].to_numpy(),
        f.state: patients["state"].to_numpy(),
        f.zip3: patients["zip3"].to_numpy(),
        f.first_activity: earliest,
        f.last_activity: latest,
        f.record_created: as_date(created),
        f.record_updated: as_datetime(np.maximum(created, patients["window_end"].to_numpy()), rng),
    }
    frame = finalize(cfg, roles.PERSON, data, n)
    frame[f.first_activity] = trim_to_rate(
        frame[f.first_activity], rate_of(cfg, roles.PERSON, "first_activity"), rng)
    frame[f.last_activity] = trim_to_rate(
        frame[f.last_activity], rate_of(cfg, roles.PERSON, "last_activity"), rng)
    preset = [f.first_activity, f.last_activity]
    return apply_fill(frame, cfg.field_specs(roles.PERSON), rng, preset=preset)


def emit_practice(cfg, rng, orgs, spans):
    f = cfg.fields(roles.ORGANIZATION)
    n = len(orgs)
    low, high = spans
    key = orgs["organization_id"].to_numpy()
    data = {
        f.organization_id: key,
        f.state: orgs["state"].to_numpy(),
        f.zip3: orgs["zip3"].to_numpy(),
        f.first_activity: low.reindex(key).to_numpy(),
        f.last_activity: high.reindex(key).to_numpy(),
        f.record_created: as_date(orgs["onboard_day"].to_numpy() - rng.integers(0, 200, n)),
    }
    frame = finalize(cfg, roles.ORGANIZATION, data, n)
    frame[f.first_activity] = trim_to_rate(
        frame[f.first_activity], rate_of(cfg, roles.ORGANIZATION, "first_activity"), rng)
    frame[f.last_activity] = trim_to_rate(
        frame[f.last_activity], rate_of(cfg, roles.ORGANIZATION, "last_activity"), rng)
    preset = [f.first_activity, f.last_activity]
    return apply_fill(frame, cfg.field_specs(roles.ORGANIZATION), rng, preset=preset)


def emit_provider(cfg, rng, practitioners, spans):
    f = cfg.fields(roles.PRACTITIONER)
    n = len(practitioners)
    low, high = spans
    key = np.array([f"{p}:{q}" for p, q in zip(practitioners["organization_id"].to_numpy(),
                                               practitioners["practitioner_id"].to_numpy())],
                   dtype=object)
    data = {
        f.practitioner_id: practitioners["practitioner_id"].to_numpy(),
        f.organization_id: practitioners["organization_id"].to_numpy(),
        f.npi_credential: pick(rng, PRACTITIONER_CREDENTIALS, n),
        f.npi_taxonomy_code: pick(rng, PRACTITIONER_TAXONOMY, n),
        f.npi_taxonomy_specialty: pick(rng, PRACTITIONER_SPECIALTIES, n),
        f.npi_state: practitioners["stated_state"].to_numpy(),
        f.stated_specialty: practitioners["stated_specialty"].to_numpy(),
        f.stated_state: practitioners["stated_state"].to_numpy(),
        f.stated_zip3: practitioners["stated_zip3"].to_numpy(),
        f.first_activity: low.reindex(key).to_numpy(),
        f.last_activity: high.reindex(key).to_numpy(),
        f.record_created: as_date(practitioners["onboard_day"].to_numpy()),
        f.record_updated: as_datetime(practitioners["onboard_day"].to_numpy() + 30, rng),
    }
    frame = finalize(cfg, roles.PRACTITIONER, data, n)
    frame[f.first_activity] = trim_to_rate(
        frame[f.first_activity], rate_of(cfg, roles.PRACTITIONER, "first_activity"), rng)
    frame[f.last_activity] = trim_to_rate(
        frame[f.last_activity], rate_of(cfg, roles.PRACTITIONER, "last_activity"), rng)
    preset = [f.first_activity, f.last_activity]
    return apply_fill(frame, cfg.field_specs(roles.PRACTITIONER), rng, preset=preset)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def write_tables(tables, cfg, out_dir, fmt):
    """Write one file per table; PSV mirrors a plain pipe-delimited delivery."""
    os.makedirs(out_dir, exist_ok=True)
    for role, frame in tables.items():
        name = cfg.table(role)
        if fmt == "parquet":
            frame.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
            continue
        text = frame.copy()
        for field in cfg.field_specs(role):
            base, col = field["type"]["base"], field["name"]
            if base == "date":
                text[col] = text[col].dt.strftime("%Y-%m-%d")
            elif base == "datetime":
                text[col] = text[col].dt.strftime("%Y-%m-%d %H:%M:%S")
            elif base == "bit":
                text[col] = text[col].map({True: "1", False: "0"})
        text.to_csv(os.path.join(out_dir, f"{name}.psv"), sep="|", index=False, na_rep="")


def write_truth(truth, out_dir):
    """Ground truth destroyed by de-identification, for methods validation."""
    path = os.path.join(out_dir, "_truth")
    os.makedirs(path, exist_ok=True)
    for name, frame in truth.items():
        frame.to_parquet(os.path.join(path, f"truth_{name}.parquet"), index=False)
    return path


def fill_report(tables, cfg, tolerance=3.0):
    """Achieved vs documented fill rate for every field.

    A field is flagged only when it misses the tolerance *and* the miss exceeds
    three binomial standard errors. The smallest tables hold only tens of rows
    and can land only on coarse multiples of 100/n, so a fixed tolerance alone
    reports sampling noise as a defect.
    """
    rows = []
    for role in cfg.table_roles():
        frame = tables[role]
        n = len(frame)
        for field in cfg.field_specs(role):
            documented = field["fill_rate"] or 0.0
            achieved = 100.0 * frame[field["name"]].notna().sum() / n if n else 0.0
            proportion = documented / 100.0
            std_error = (
                100.0 * math.sqrt(max(proportion * (1 - proportion), 0.0) / n) if n else 0.0
            )
            deviation = abs(achieved - documented)
            rows.append({
                "table": cfg.table(role), "field": field["name"],
                "documented": round(documented, 2), "achieved": round(achieved, 2),
                "diff": round(achieved - documented, 2),
                "n": n, "se": round(std_error, 2),
                "flag": "OFF" if deviation > tolerance and deviation > 3 * std_error else "",
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def simulate(n_patients, seed, cfg):
    rng = np.random.default_rng(seed)
    tables, truth = {}, {}

    orgs = build_organizations(cfg, rng, n_patients)
    practitioners = build_practitioners(cfg, rng, orgs)
    sample_practitioner, practitioners = practitioner_sampler(practitioners, orgs)
    patients = build_patients(cfg, rng, n_patients, orgs)

    visits = build_visits(rng, patients, orgs, sample_practitioner)
    deceased, true_death_day = assign_deaths(rng, patients, visits)
    visits = truncate_at_death(visits, deceased, true_death_day)

    finding_cat = value_set(cfg, roles.FINDING_CATEGORY)
    intervention_cat = value_set(cfg, roles.INTERVENTION_CATEGORY)
    background_cat = cfg.value_set(roles.BACKGROUND_CATEGORY)
    smoking_cat = [v["value"] for v in background_cat if v["value"] != cfg.null_marker
                   and "smoking status" in (v["definition"] or "").lower()]
    other_cat = [v["value"] for v in background_cat
                 if v["value"] != cfg.null_marker and v["value"] not in smoking_cat]
    prescription_actions = value_set(cfg, roles.PRESCRIPTION_ACTION)
    activity_types = value_set(cfg, roles.ACTIVITY_TYPE)
    vaccination_status = value_set(cfg, roles.VACCINATION_STATUS)

    tables[roles.ENCOUNTER], truth["encounter"] = emit_visit(cfg, rng, visits)

    obs_rows = expand_visits(rng, visits, 0)
    tables[roles.FINDING] = emit_observation(cfg, rng, visits, obs_rows, finding_cat)
    obs_patient = visits["patient_row"].to_numpy()[obs_rows]

    lab_rows = expand_visits(rng, visits, 1)
    tables[roles.TEST_RESULT], lab_patient = emit_lab(cfg, rng, visits, lab_rows, patients)

    med_rows = expand_visits(rng, visits, 2)
    tables[roles.DRUG_EXPOSURE], med_patient = emit_medication(
        cfg, rng, visits, med_rows, patients, prescription_actions, activity_types)

    prob_rows = expand_visits(rng, visits, 3)
    tables[roles.CONDITION], prob_patient, prob_codes = emit_problem(
        cfg, rng, visits, prob_rows, patients)

    proc_rows = expand_visits(rng, visits, 4)
    tables[roles.INTERVENTION], proc_patient = emit_procedure(
        cfg, rng, visits, proc_rows, intervention_cat)

    tables[roles.CLINICAL_BACKGROUND], hist_patient = emit_history(
        cfg, rng, visits, patients, smoking_cat, other_cat)
    tables[roles.SERVICE_REQUEST], ref_patient = emit_referral(cfg, rng, visits)
    tables[roles.VACCINATION], imm_patient = emit_immunization(
        cfg, rng, visits, vaccination_status)
    tables[roles.INTOLERANCE], allergy_patient, allergy_days = emit_allergy(cfg, rng, visits)
    tables[roles.INTOLERANCE_REACTION] = emit_allergy_reaction(
        cfg, rng, tables[roles.INTOLERANCE], allergy_days)
    tables[roles.CONDITION_CODE] = emit_problem_code(
        cfg, rng, tables[roles.CONDITION], prob_codes, tables[roles.CLINICAL_BACKGROUND])

    condition_key = cfg.field(roles.CONDITION, "condition_id")
    index = ConditionIndex(prob_patient, tables[roles.CONDITION][condition_key].to_numpy(),
                           n_patients)
    for role, patient_rows in [
        (roles.TEST_RESULT, lab_patient), (roles.FINDING, obs_patient),
        (roles.DRUG_EXPOSURE, med_patient), (roles.INTERVENTION, proc_patient),
        (roles.SERVICE_REQUEST, ref_patient),
    ]:
        index.attach(tables[role], cfg.field(role, "condition_id"),
                     rate_of(cfg, role, "condition_id"), patient_rows, rng)

    tables[roles.DEATH], truth["death"] = build_mortality(
        cfg, rng, patients, deceased, true_death_day)

    person_key = cfg.field(roles.PERSON, "person_id")
    org_key = cfg.field(roles.ORGANIZATION, "organization_id")

    def patient_key(_role, frame):
        return frame[person_key]

    def practice_key(_role, frame):
        return frame[org_key]

    def provider_key(role, frame):
        names = [cfg.field(role, r) for r in PRACTITIONER_KEY_ROLES[role]]
        cols = [c for c in names if c in frame.columns]
        first = frame[cols].bfill(axis=1).iloc[:, 0]
        return frame[org_key].astype("string") + ":" + first.astype("string")

    tables[roles.PERSON] = emit_patient(
        cfg, rng, patients, activity_spans(cfg, tables, patient_key))
    tables[roles.ORGANIZATION] = emit_practice(
        cfg, rng, orgs, activity_spans(cfg, tables, practice_key))
    tables[roles.PRACTITIONER] = emit_provider(
        cfg, rng, practitioners, activity_spans(cfg, tables, provider_key))

    truth["person"] = pd.DataFrame({
        person_key: patients["person_id"].to_numpy(),
        "true_birth_year": patients["true_birth_year"].to_numpy(),
        "true_age_at_extract": patients["true_age"].to_numpy().round(2),
        "delivered_birth_year": patients["capped_birth_year"].to_numpy(),
        "age_capped": patients["age_capped"].to_numpy(),
        "true_smoking_status": patients["smoking"].to_numpy(),
        "true_height_in": patients["height_in"].to_numpy().round(2),
        "baseline_bmi": patients["bmi0"].to_numpy().round(3),
        "deceased": deceased,
        "has_clinical_records": patients["has_records"].to_numpy(),
        **{f"cond_{c}": patients["cond_" + c].to_numpy() for c in CONDITIONS},
    })

    ordered = {role: tables[role] for role in cfg.table_roles()}
    return ordered, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-patients", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--format", choices=["parquet", "psv"], default="parquet",
                        help="the real delivery format is not yet known")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--config", default=schema_config.DEFAULT_PATH)
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="percentage points of fill-rate drift to tolerate")
    args = parser.parse_args()

    try:
        cfg = schema_config.SchemaConfig.load(args.config)
    except schema_config.ConfigError as exc:
        raise SystemExit(str(exc)) from None
    started = time.time()
    tables, truth = simulate(args.n_patients, args.seed, cfg)
    write_tables(tables, cfg, args.out, args.format)
    truth_dir = write_truth(truth, args.out)

    print(f"\nVNEHR simulation: {args.n_patients} patients, seed {args.seed}, "
          f"{time.time() - started:.1f}s -> {args.out}")
    print(f"truth tables -> {truth_dir}\n")
    print(f"{'table':20s} {'rows':>10s}  columns")
    for role, frame in tables.items():
        print(f"{cfg.table(role):20s} {len(frame):>10,d}  {len(frame.columns):3d}")

    report = fill_report(tables, cfg, args.tolerance)
    off = report[report["flag"] == "OFF"]
    print(f"\nfill-rate check: {len(report)} fields, {len(off)} off by "
          f"more than {args.tolerance:g} percentage points")
    if len(off):
        print(off.to_string(index=False))
    else:
        worst = report.reindex(report["diff"].abs().sort_values(ascending=False).index).head(10)
        print("largest deviations:")
        print(worst.to_string(index=False))
    return 1 if len(off) else 0


if __name__ == "__main__":
    sys.exit(main())
