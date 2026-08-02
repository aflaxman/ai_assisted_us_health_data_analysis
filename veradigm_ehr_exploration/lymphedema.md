# Cancer-related lymphedema: feasibility reasoning

Background for `ANALYSIS_PLAN.md`. Written without vendor schema detail, so it is
safe to commit. Code sets named below are from recall and **must be verified
against a current code list** before use.

## The central difficulty

Lymphedema is caused by treatment — axillary or pelvic node dissection, regional
irradiation, taxane chemotherapy — and appears 12 to 36 months later, often
longer.

Those two facts sit on opposite sides of this dataset's boundary. **The exposure
happens in hospitals and specialty centers; the outcome shows up in ambulatory
practice.** The data sees the second and largely misses the first. Any design has
to be built around that asymmetry rather than pretending it away.

There is a second, deeper problem, and it is the more interesting one. Lymphedema
is chronically under-coded. It is diagnosed clinically, often by a therapist
rather than a physician, and frequently documented as "arm swelling" in a note
that never becomes a structured code. Published validation of administrative
codes against measurement-based diagnosis puts sensitivity well below complete.
So a diagnosis code is not a measurement of lymphedema; it is a noisy indicator
of it.

That reframes the project. **The right use of this dataset is not to estimate
lymphedema incidence directly, but to estimate the measurement model** — and let
the incidence estimate fall out of it with honest uncertainty. That is a
methodological question, which is where this dataset is strongest and where the
two-mechanism modelling in `brfss_income_grouped_em/` already applies.

## Five traces, with different error structures

The saving grace is that lymphedema leaves several independent marks, and an
ambulatory record captures most of them:

| Signal | Error character |
|---|---|
| Diagnosis codes for lymphedema (`I97.2` postmastectomy, `I89.0` unspecified) | Specific, insensitive |
| Referrals to lymphedema therapy, physical therapy, or occupational therapy | Sensitive to care-seeking |
| Compression garments or pumps recorded as durable medical equipment | Very specific, sparsely captured |
| Manual therapy (`97140`) and bioimpedance for lymphedema assessment (`93702`) | Very specific, low sensitivity |
| Recurrent limb cellulitis, via antibiotic courses and diagnosis codes | Sensitive to severe disease, non-specific |

Their differing error structures are what make a latent-variable model
identifiable rather than wishful. Note the last one especially: recurrent
cellulitis in a swollen limb is a hallmark complication, and antibiotic
prescribing is among the best-captured exposures available.

Exclude hereditary lymphedema (`Q82.0`) throughout.

## Handles on the exposure side

**Adjuvant endocrine therapy is an excellent survivorship marker.** Tamoxifen,
anastrozole, and letrozole are prescribed and refilled in ambulatory care for
five to ten years after breast cancer, and drug records carry national drug and
normalized-drug identifiers on the large majority of rows. This identifies a
breast-cancer-survivor cohort more reliably than waiting for a cancer diagnosis
code to appear on a primary care problem list, and its start date is the most
promising surrogate index date. An analogous treatment-initiation anchor should
be sought for each of the other site groups.

**Structured surgical history** carries mastectomy and node dissection as free
text — which is exactly what a primary care practice records about an operation
performed elsewhere.

## What is missing, and what each absence costs

**Surgical extent is absent**, and it is the dominant risk factor: sentinel node
biopsy versus full axillary dissection is roughly a fourfold difference in risk,
and nodes examined is the continuous version of it. Free-text surgical history
may separate the two for some patients, unreliably.

**Dates for procedures in structured history are sparse.** You will know a patient had
a mastectomy and not know when. For a disease defined by its latency after a
procedure, this is the weakest link in the whole design and the constraint I
would resolve first.

**Radiation is almost certainly invisible**, being delivered in radiation
oncology centers rather than ambulatory practices.

**Laterality is not captured anywhere**, so the standard research design —
comparing the affected limb to the patient's own contralateral limb — is
unavailable.

**Left truncation will be severe.** Given the long latency and a five-year
window, a large share of the apparent cohort will be prevalent rather than
incident cases, and separating them requires the treatment date that is sparsely
recorded.

**Breast is far more tractable than the other sites.** The postmastectomy
diagnosis code is specific; lower-limb lymphedema after gynecologic, urinary, or
melanoma surgery falls to a nonspecific code that also catches venous
insufficiency, heart failure, and obesity-related edema. The scale here is
genuinely attractive for those understudied sites, since single-institution
cohorts are too small — but the ascertainment problem is much worse, and they
should be a separate, more uncertain stratum rather than pooled.

## The body-mass ceiling lands squarely on this question

Obesity is the strongest *modifiable* risk factor for breast cancer-related
lymphedema, and risk keeps climbing into class III. The deidentification ceiling
on body-mass index coincides with the class III boundary, so the most important
modifiable risk factor is censored precisely where the risk is highest.

Two observations make this less bleak than it sounds.

**Recorded weight is far less censored than the index.** The weight ceiling sits
well above the weight at which an average-height woman reaches the index ceiling,
so weight retains variation across exactly the range where the index has
collapsed to a point mass. Weight is also recorded repeatedly, and
post-diagnosis weight *gain* is itself a documented risk factor — a within-patient
trajectory needing no height at all. **Weight trajectory, not the index, is the
right exposure here**, and it is the one deidentification left mostly intact.

**Height is arithmetically determined** wherever weight and an uncensored index
appear together, and height is stable within a person — so a single such record
would uncensor every other record for that patient. Whether reconstructing a
deliberately redacted field is permitted is a license question, not a technical
one, and it is listed in `CLAUDE.md` as requiring written approval. For this
application it is worth actually asking, because it is the difference between
estimating a dose-response across class III obesity and truncating at the
clinically decisive threshold.

## What to request from the vendor

Ranked by how much each would change what is answerable. The first two are well
above the rest.

**1. Linked claims.** This fixes the largest gap. Claims would supply the
inpatient surgery that defines the exposure — dissection versus sentinel biopsy,
and plausibly nodes examined — plus radiation oncology, durable medical equipment
claims for compression garments, and **payer and insurance type**, which converts
the Medicare coverage-change analysis from an age proxy into a real policy
evaluation. They would also capture out-of-network care, which is what a
denominator requires. Ask specifically whether the linkage is open or closed
claims: closed claims give true enrollment spans and a denominator, open claims
do not.

**2. Note-derived structured data.** Lymphedema is the paradigm case where notes
beat codes — limb circumference measurements, laterality, symptom descriptions,
and severity staging are routinely written and almost never coded. This is the
difference between a noisy binary indicator and an actual measurement, and it
would let the measurement model be validated against something approaching a
reference standard. Note that this is a separate product and would need its own
license; confirm scope before assuming it is included.

**3. Oncology-enriched or registry-linked variables.** Ask whether stage, receptor
status, tumor size, and above all **nodes examined and nodes positive** are
available for an oncology subset. Nodes examined is the exposure variable for
this entire literature.

**4. Better clinician specialty.** The registry-derived specialty field is
essentially empty, so there is no reliable way to distinguish an oncology
practice from a primary care or rehabilitation practice. Ascertainment of every
signal above depends on practice type. This is probably cheap to fix and it
affects everything.

**5. External mortality linkage.** Cancer survivors face competing mortality, and
in-record death capture is incomplete. Ask whether a national index or commercial
linkage is offered. Note that using it would raise the comingling question in
`CLAUDE.md`.

**6. Site-level geography.** The facility identifier is unpopulated throughout, so
sub-practice geography and site-level clustering are unavailable.

Two questions worth asking that cost nothing: what share of the network is
oncology or rehabilitation practices, and how deep the historical look-back goes —
since long latency plus left truncation makes history worth more here than in
most studies.
