# NHANES Mortality by Liver Fibrosis Status

### Cohort selection rationale

| Cohort | Max FU (months) | % ≥36m | % ≥24m | UCOD detail | Elastography |
|--------|----------------|--------|--------|-------------|-------------|
| 2007-2008 | ~159 | ~96% | ~98% | Full (10 groups) | No |
| 2011-2012 | ~113 | ~97% | ~98% | Full (10 groups) | No |
| 2017-2018 | ~37  | ~2%  | ~51% | Coarsened (3 groups) | Yes (VCTE) |

### Key findings

**Unmatched analysis:**
- FIB-4-defined fibrosis consistently associated with 13–25× higher crude mortality
- LSM-based fibrosis (2017-2018) shows ~3.5× crude risk ratio

**Progressive matching reveals confounding structure:**
- Matching on age+sex produces the largest attenuation (crude RR ~13–25× drops to ~3–6×),
  consistent with age being embedded in the FIB-4 formula
- Adding BMI, SBP, and smoking produces modest further attenuation
- Adding fasting-subsample variables (LDL-C, FPG) restricts the sample to ~⅓ of participants;
  the RR estimate changes but wider CIs reflect the loss of power
- After full matching, fibrosis remains associated with elevated mortality (RR > 1),
  though small matched samples yield wide confidence intervals

### Limitations
1. **Fibrosis is proxy-defined** — FIB-4 and LSM are not histological diagnoses
2. **FIB-4 includes age** — crude RR overstates the association; matched/adjusted estimates more reliable
3. **Public-use COD coarsening** — Only 3 UCOD groups available for 2017-2018
4. **Short follow-up** for 2017-2018 (~51% have ≥24 months)
5. **Fasting subsample** — LDL-C and FPG restrict matching to ~⅓ of the cohort
6. **Small matched samples** — Many cells have <10 events
7. **No survey weights** — Unweighted estimates; not nationally representative
