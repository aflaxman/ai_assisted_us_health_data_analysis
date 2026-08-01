# CLAUDE.md — Veradigm EHR analysis project

These rules override the repository-root `CLAUDE.md` wherever the two conflict.
In particular the root file's convention of writing data under a shared
`data/raw/` and `data/derived/` tree **does not apply to licensed Data** — only
to synthetic fixtures.

## ⛔ Absolute constraint — read first

This project uses patient-level EHR data licensed from **Veradigm LLC** under a
restrictive Data License Agreement (effective 9 July 2026, one-year term).
**Never read, open, load, print, sample, summarize, embed, or transmit any
Veradigm Data, or any file derived from it that contains row-level records.**
Sending Data to an AI or model backend counts as prohibited third-party access
and breaches the license. When in doubt, do not touch the file — ask.

## Where the Data lives (off-limits)

- The Data root is configured locally and **must never live inside this git
  repo**, be written into any committed file, or appear in a commit message.
- You operate only inside this repository, on code. You have no reason to read
  anything under the Data root. Do not `cat`, `head`, `ls`, `read_csv`, glob, or
  index it.
- If you notice Data files reachable from the working directory, **stop and
  warn me.**

## What you may help with

- Writing and refactoring analysis code (Python, environment managed with `uv`).
- Study design, statistical approach, SAP drafting, method selection.
- Docstrings, tests, and examples that run on **synthetic fixtures only**.
- Interpreting output I paste **only if it is already aggregated**, with no
  small cells.

## Repo hygiene — never commit

- Data, extracts, or any row-level content.
- Veradigm's data dictionary, data layout, schema, or DDL, or any structure
  "substantially similar or derivative" of theirs. The license bars derivative
  schemas. **This includes table names, field names, field descriptions,
  documented fill rates, permitted-value lists, and entity-relationship
  diagrams**, in documentation as much as in code.
- Credentials, tokens, or connection strings.
- Absolute cluster paths that reveal where the Data resides.

**Prescribed pattern:** all Veradigm-specific names, types, fill rates, and
value sets live in a single gitignored private config. Committed code reads
identifiers from that config at runtime and contains no Veradigm literals.
Committed prose describes methods and study design, never the schema.

## Analysis output rules

- Anything intended to leave the environment must be **aggregated**, with
  small-cell suppression per the Statistician Certification safeguards.
- Never write code whose purpose is re-identification, record linkage against
  external identifiers, or matching patients or providers.
- Do not write code that reconstructs fields the vendor deliberately redacted
  without written approval recorded below.

## Prohibited by license — do not build (§1c)

- **No ML or algorithm development as a deliverable.** Do not train models,
  build classifiers, or improve an algorithm as a product of this Data. Using a
  model purely as an in-house analysis step for a publication is a **gray
  area** — flag it, do not assume it is allowed.
- **Do not enhance SmartVA-Analyze, Vivarium, or any tool or product** using
  this Data.
- **No comingling.** Do not write pipelines that merge this Data with NHANES,
  GBD, BRFSS, area-deprivation indices, life tables, or any other source, or
  feed it into a multi-source microsimulation, without explicit written
  approval recorded below. Narrative comparison to published figures in a
  discussion section is not comingling; statistical calibration, benchmarking,
  or record linkage is.

## Purpose and term

Non-commercial research and academic publication only. One-year term expiring
~9 July 2027. All Data and backups destroyed within **5 days** of study
completion or termination, with a certificate of destruction to Veradigm;
audit-readiness retained for 6 months. Suspected breach → notify
privacy@veradigm.com immediately.

Because destruction is irreversible and quick, **every aggregate result needed
for publication must be extracted before the analysis freeze.** You cannot go
back for a forgotten table.

## If asked to do something on these lists

Refuse the specific action, name the rule it hits, and propose a compliant
alternative — run it in-house, aggregate first, use a synthetic fixture. Do not
silently comply.

## Open questions — resolve in writing before relying on them

1. **§1c(x):** Does using a model *as an analysis method* for a standalone
   publication, with no model deliverable and no tool improvement, fall outside
   the ML/algorithm-development bar? This gates the latent-class measurement
   model that is central to the analysis plan.
2. **§1c(ix), §1b:** Comingling — confirm before any external benchmarking,
   area-level linkage, or multi-source simulation.
3. **§1a vs §1c(viii):** Reconcile "cite Veradigm as the source" against "do not
   identify Veradigm as a provider."

Not legal advice. Confirm ambiguous items with the UW SoM Contracting &
Regulatory Unit with Veradigm sign-off in writing before proceeding.

<!-- Approvals log: record any written Veradigm/UW-Contracting sign-off that
     relaxes a rule above, with date. Nothing approved as of 2026-08-01. -->
