# Delivery conformance check

**Status: blocking.** Nothing in `ANALYSIS_PLAN.md` proceeds until this passes.

If the delivery exceeds the licensed scope, then every record outside that scope
is unlicensed possession, and any analysis run on it is analysis of data we have
no right to — results that would have to be discarded and recomputed anyway.
Conformance is therefore both a legal gate and a cheap way to avoid wasted work.

This is an acceptance test, not an accusation. Run it the same way whether or not
you expect it to pass.

## What we agreed to receive

From the executed agreement, as summarized in the compliance plan. **Confirm each
line against Exhibit A of the signed document rather than against this file** —
this is a working restatement, not the contract.

| Criterion | Agreed |
|---|---|
| Population | Deidentified cancer patients |
| Size | ~2.9 million patients |
| Cancer sites | Breast, melanoma, female genital, male genital, urinary |
| Period | 2020–2025 |
| Content | Structured ambulatory EHR records |
| Deidentification | Applied, per the vendor's stated methodology |

## Tier 0 — resolve on paper first, with no data contact

Do these before touching anything. They cost nothing, carry no risk, and may
settle the question outright.

1. **Ask the vendor what they shipped.** Request the cohort definition they
   applied, the patient count, the record counts per file, and the extract date.
   A one-line answer may resolve this faster than any amount of profiling, and it
   creates the written record you will want either way.
2. **Reconcile the transmittal against Exhibit A.** Compare any manifest, README,
   or transmittal note included with the delivery to the agreed specification.
3. **Check the delivery's own metadata.** File names, folder naming, and dates —
   available from a directory listing without opening any file.
4. **Rule out the false alarm.** If the suspicion rests on the data dictionary
   looking generic, that is weak evidence. A cohort restriction is a row filter,
   not a schema change, so the vendor would ship their standard network-wide
   dictionary with a cancer extract, and its documented fill rates would describe
   the whole network rather than your subset. The dictionary predating the
   agreement is likewise normal. **Neither fact indicates a wrong delivery.**

## Tier 1 — volume triage, from file metadata alone

A quick reconciliation that needs only file sizes and formats, not contents.

The delivered volume implies a patient count only once the format is known, and
the implication differs by roughly an order of magnitude:

- **Plain delimited text.** ~200 GB is broadly *consistent* with 2.9M cancer
  patients over five years. Cancer patients are heavy utilizers, and a few
  hundred records each across all record types lands in this range.
- **Compressed text or columnar (Parquet).** The same 200 GB implies several
  times to more than ten times that many patients. That would be a strong signal
  of over-delivery — most likely a network-wide extract rather than the licensed
  cohort.

So **determine the format first.** It is the cheapest discriminating test
available and it needs no data access beyond reading file extensions and headers.

Treat this as triage, not proof. Compression ratios and per-patient record
volumes both vary enough that the conclusion needs Tier 2 confirmation.

## Tier 2 — cohort conformance, aggregate counts only

Run only if Tiers 0 and 1 leave the question open. Every output is a count or a
proportion with small cells suppressed. **Stop at the first disconfirming result**
rather than completing the full battery — once you know the delivery is
out of scope, further processing is unnecessary handling of unlicensed data.

1. **Cohort size.** Distinct patient identifiers across the delivery. Compare
   to ~2.9M. Judge on order of magnitude, not precision: a 10–20% difference is
   plausible cohort-definition drift, while a multiple is a different delivery.
2. **Cancer conformance.** The proportion of patients carrying any malignancy
   diagnosis. If a substantial share have none, the cohort was not built the way
   we expect, and a general-population extract is the leading explanation.
3. **Site-group conformance.** The distribution across the five agreed site
   groups, and — the more informative half — **the volume of patients whose only
   malignancy falls outside them.** Common cancers absent from the agreement, such
   as lung or colorectal, appearing as sole primaries in quantity is direct
   evidence of over-delivery.
4. **Period conformance.** Date ranges per record type. Distinguish two cases
   that look alike and are not: records *created* outside 2020–2025, which is a
   scope problem, versus records created inside the window that *describe* earlier
   events, such as a diagnosis with an onset date years prior, which is normal and
   expected.
5. **Record-type inventory.** Confirm the delivery contains only the structured
   record types covered by the agreement. Anything resembling a
   notes-derived or natural-language-processed product is a separate offering and
   would need its own license — the dictionary contains a stray reference
   suggesting such a product exists, so check specifically for it.

## Tier 3 — deidentification conformance

**Escalate immediately if any of these fail.** A delivery that is less
deidentified than agreed is a privacy incident, not a scope dispute, and it is
more serious than receiving too many patients.

Confirm that the vendor's stated transformations were actually applied — the age
ceiling, the body-mass and weight ceilings, the suppressed geography for
small-population areas, the date shifting on mortality records, and the
redaction of sensitive diagnosis codes. Each is checkable from an aggregate
distribution: a cap shows up as a pile-up at the boundary, a redaction as a
sentinel value or an absent category.

The specific transformations and how to test each are in the un-committed working
notes, which is also where the profiler that computes them lives.

## Decision matrix

| Finding | Meaning | Action |
|---|---|---|
| All tiers pass | Delivery conforms | Record the result, proceed to Phase 1 |
| Cohort size materially high; non-agreed cancers present in volume | Over-delivery, likely network-wide extract | Stop. Escalate. |
| Cohort size materially low, or agreed sites missing | Under-delivery | Stop. Request re-delivery. |
| Records outside 2020–2025 by creation date | Period over-scope | Stop. Escalate. |
| Unlicensed record types present | Scope breach | Stop. Escalate. |
| Deidentification not applied as stated | **Privacy incident** | Stop. Escalate immediately, notify per the breach clause. |
| Ambiguous | Insufficient information | Do not proceed on assumption. Resolve with the vendor in writing. |

## If the delivery does not conform

1. **Stop processing.** No analysis, no further profiling beyond what established
   the finding.
2. **Do not delete anything unilaterally.** Deletion without instruction destroys
   the evidence of what was received and may conflict with the destruction and
   certification procedure the agreement specifies. Preserve the delivery as-is
   and secure access to it.
3. **Notify UW Contracting and the vendor in writing**, promptly. For a
   deidentification failure, follow the breach-notification clause and its
   timeline rather than ordinary correspondence.
4. **Obtain written instruction** on remediation — secure destruction with a
   certificate, re-delivery of a conforming extract, or an amended agreement that
   covers what was actually sent. Record the outcome in the approvals log in
   `CLAUDE.md`.
5. **Document the discovery**: what was found, who found it, when, what was
   accessed in the course of finding it, and what was done next. Write this
   contemporaneously.

## Recording the result

Whatever the outcome, write a short signed conformance record — date, who ran it,
which tiers were run, the aggregate figures observed, and the conclusion. File it
with the project's regulatory documentation. If the delivery conforms, this is
the artifact that lets you demonstrate the check was done. If it does not, it is
the first entry in the remediation record.

Attach the same record, updated, to the closeout package alongside the
certificate of destruction.

## Tooling status

The profiler needed for Tiers 1 through 3 exists but is quarantined pending the
compliance refactor described in `ANALYSIS_PLAN.md`. It already emits aggregates
with small-cell suppression and never emits identifiers. Two additions are
required before it can serve as the conformance instrument: the cancer site-group
classification for Tier 2, and explicit pass/fail assertions for the Tier 3
transformations rather than descriptive distributions alone.
