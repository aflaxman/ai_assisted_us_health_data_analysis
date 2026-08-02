# Veradigm EHR — cancer-related lymphedema

Analysis code and study planning for a licensed ambulatory EHR dataset. The
research question is cancer-related lymphedema; the licensed cohort is cancer
patients across five site groups whose treatment causes it.

**Read `CLAUDE.md` first.** It carries the binding license constraints and
overrides the repository-root conventions. The short version: the Data is never
read by any assistant, never leaves its environment except as suppressed
aggregates, and the vendor's schema never enters this repository.

## Documents

| File | Purpose |
|---|---|
| `CLAUDE.md` | Binding license rules. Read before doing anything. |
| `DELIVERY_CONFORMANCE.md` | Acceptance test — confirm the delivery matches what was licensed. **Blocks everything else.** |
| `ANALYSIS_PLAN.md` | Phased analysis plan, from feasibility through closeout. |
| `lymphedema.md` | Domain feasibility reasoning, and what additional data to request. |

## Working model

The Data is licensed; the analysis code is not. So the two are kept apart:

1. **Describe the layout once, privately.** `extract_schema.py` reads the vendor's
   data dictionary and writes `config.local.json`, which holds every
   vendor-specific identifier, type, documented completeness figure, and
   permitted-value list. That file is gitignored and never committed — the
   license bars derivative schemas.
2. **Committed code speaks in roles.** `roles.py` defines generic clinical
   concepts, and `schema_config.py` resolves them against the private config at
   runtime. Nothing in this repository names a vendor table or column.
3. **Develop against synthetic data.** `simulate_vnehr.py` generates a fake
   dataset with the same shape, completeness structure, referential integrity,
   and deidentification artifacts, plus a truth directory holding the quantities
   deidentification destroys — so methods that try to recover them can be scored.
   Its cancer cohort adds the study's own subject matter: five site groups, the
   operations and drugs that treat them, an unobserved lymphedema state, and the
   five imperfect traces that state leaves in an ambulatory record.
4. **Only the analyst runs anything on the Data**, and results come back as
   aggregates with small cells suppressed.

## Setup

```bash
uv venv && uv pip install -r requirements.txt

# One time: build the private config from the vendor data dictionary.
# The dictionary is confidential and lives outside this repo.
.venv/bin/python extract_schema.py --dictionary <path-to-dictionary>

# Synthetic dataset for development: cancer cohort, 2020-2025, by default.
.venv/bin/python simulate_vnehr.py --n-patients 5000 --seed 0

# Dry run of the profiler against the synthetic data.
.venv/bin/python inventory_real_data.py \
    --root <synthetic-data-dir> --out outputs/sim_profile.json
```

`config.example.json` shows the private config's structure using placeholder
names, so the wiring is documented without disclosing anything.

## The profiler

`inventory_real_data.py` is the only script intended to run against the Data, and
the analyst runs it, not an assistant. It emits aggregate metadata only, under
three rules:

- Identifier columns never have their values emitted.
- Clinical vocabulary codes are reported as suppressed value counts, never as
  numeric quantiles — a quantile would emit a rare code with no cell-size guard.
- Every listed value must occur at least `--min-cell` times, default 20.

Verified against synthetic data: zero patient identifiers present, zero code
columns carrying quantiles, and the smallest listed count across a whole profile
is exactly the threshold.

Its first job is not analysis but acceptance testing — see
`DELIVERY_CONFORMANCE.md`.

## The simulated cancer cohort

`--cohort cancer`, the default, is the testbed for the research question. Every
person carries a malignancy in one of the five site groups over a five-year
window, with the operation recorded as free text, adjuvant endocrine therapy
refilled for years, and taxane chemotherapy. The constraints that make the study
hard are reproduced rather than wished away: the date of an operation is
recorded for only a small minority of entries, radiation is almost never visible
because it happens elsewhere, and laterality appears nowhere.

Lymphedema is simulated as an **unobserved** state driven by nodal-surgery
extent, radiation, chemotherapy, site, and body mass, arriving 12 to 36 months
after treatment. The record shows only five imperfect traces of it — diagnosis
codes, therapy referrals, compression equipment, manual therapy and bioimpedance
procedures, and recurrent limb cellulitis — each with poor sensitivity on its
own. Crucially the five are **not conditionally independent given the state**:
one per-person care-engagement propensity drives both how often someone is seen
and how readily each trace is written down, so they agree far more often than
independence predicts. That is the dependence structure the planned measurement
model has to handle, and a testbed without it could not tell a naive latent
class model from one that models the dependence.

`_truth/` records what the delivery cannot: true status, onset date, surgical
extent, treatment index date, and the care-engagement value.

`--cohort general` keeps the earlier general-population content and the longer
window, for work on the completeness and de-identification machinery alone.

## Status

Delivery conformance is **unconfirmed**, and nothing else should start until it
is. The analysis plan's cohort description comes from the agreement, not from
the delivery, and the two have not been reconciled.

Nothing that consumes the latent state has been built. The phenotype and the
measurement model in the plan's Phase 2 are gated on the first open license
question, so the simulator supplies the testbed and stops there.
