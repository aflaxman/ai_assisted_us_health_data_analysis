"""Generic names for the clinical concepts this project's code works with.

Analysis code refers to a *role* -- ``person``, ``encounter``, ``begin_date`` --
and never to the identifier a particular delivery happens to use. ``SchemaConfig``
resolves roles against a private configuration at runtime, so everything here is
vocabulary-neutral and describes only what the concept means clinically.

Roles are grouped by the entity they belong to. Field roles are listed
alphabetically; the delivered column order comes from the configuration, not from
this file.
"""

# --------------------------------------------------------------------------
# entities
# --------------------------------------------------------------------------

PERSON = "person"                                # one row per individual
ORGANIZATION = "organization"                    # the group that runs the EHR
PRACTITIONER = "practitioner"                    # a user of the EHR
ENCOUNTER = "encounter"                          # a contact, in person or not
CONDITION = "condition"                          # an active or resolved diagnosis
CONDITION_CODE = "condition_code"                # coded form of a diagnosis
CLINICAL_BACKGROUND = "clinical_background"      # family, social, surgical background
TEST_RESULT = "test_result"                      # laboratory orders and results
FINDING = "finding"                              # vitals, exam findings, assessments
DRUG_EXPOSURE = "drug_exposure"                  # prescribed, administered or recorded
INTERVENTION = "intervention"                    # procedures and imaging
VACCINATION = "vaccination"
INTOLERANCE = "intolerance"                      # adverse-reaction and sensitivity records
INTOLERANCE_REACTION = "intolerance_reaction"    # the reactions an intolerance caused
SERVICE_REQUEST = "service_request"              # request for another clinical service
DEATH = "death"                                  # decedents only

TABLES = (
    PERSON, ORGANIZATION, PRACTITIONER, ENCOUNTER, CONDITION, CONDITION_CODE,
    CLINICAL_BACKGROUND, TEST_RESULT, FINDING, DRUG_EXPOSURE, INTERVENTION,
    VACCINATION, INTOLERANCE, INTOLERANCE_REACTION, SERVICE_REQUEST, DEATH,
)

# --------------------------------------------------------------------------
# value sets -- categorical fields whose permitted values the delivery documents
# --------------------------------------------------------------------------

ENCOUNTER_TYPE = "encounter_type"
FINDING_CATEGORY = "finding_category"
PRESCRIPTION_ACTION = "prescription_action"
ACTIVITY_TYPE = "activity_type"
INTERVENTION_CATEGORY = "intervention_category"
BACKGROUND_CATEGORY = "background_category"
VACCINATION_STATUS = "vaccination_status"

VALUE_SETS = (
    ENCOUNTER_TYPE, FINDING_CATEGORY, PRESCRIPTION_ACTION, ACTIVITY_TYPE,
    INTERVENTION_CATEGORY, BACKGROUND_CATEGORY, VACCINATION_STATUS,
)

# --------------------------------------------------------------------------
# individual value-set members the code names
#
# Only the members the simulator or an analysis actually singles out appear
# here; a role is a description of what the category means, never the string the
# delivery uses for it. Some sets carry two spellings of one concept, which the
# ``_VARIANT`` suffix marks.
#
# The last four categories below are documented in a field description rather
# than in a value sheet, so they have members but no entry in ``VALUE_SETS``.
# --------------------------------------------------------------------------

ENCOUNTER_IN_PERSON = "encounter_in_person"
ENCOUNTER_REMOTE = "encounter_remote"          # documentation-only contact
ENCOUNTER_IMPORTED = "encounter_imported"      # loaded from another system
ENCOUNTER_EMPTY = "encounter_empty"            # contact carrying no content
ENCOUNTER_INCIDENTAL = "encounter_incidental"  # content, none of it clinical

BACKGROUND_FAMILY = "background_family"
BACKGROUND_FAMILY_VARIANT = "background_family_variant"
BACKGROUND_SOCIAL = "background_social"
BACKGROUND_SURGERY = "background_surgery"
BACKGROUND_SURGERY_VARIANT = "background_surgery_variant"
BACKGROUND_DEVICE = "background_device"
BACKGROUND_PREVENTIVE_CARE = "background_preventive_care"
BACKGROUND_TRAVEL = "background_travel"
BACKGROUND_PREGNANCY = "background_pregnancy"

INTERVENTION_IMAGING = "intervention_imaging"
INTERVENTION_DIAGNOSTIC_STUDY = "intervention_diagnostic_study"
INTERVENTION_PROCEDURAL = "intervention_procedural"
INTERVENTION_DIAGNOSTICS = "intervention_diagnostics"
INTERVENTION_OTHER_TESTING = "intervention_other_testing"
INTERVENTION_FINDING = "intervention_finding"
INTERVENTION_EXAM = "intervention_exam"
INTERVENTION_VACCINATION = "intervention_vaccination"
INTERVENTION_EDUCATION = "intervention_education"

FINDING_VITAL = "finding_vital"
FINDING_VITAL_VARIANT = "finding_vital_variant"
FINDING_EXAM = "finding_exam"

ACTIVITY_ADMINISTRATION = "activity_administration"
ACTIVITY_PRESCRIPTION = "activity_prescription"
ACTIVITY_DISCONTINUATION = "activity_discontinuation"

GENDER = "gender"
GENDER_FEMALE = "gender_female"
GENDER_MALE = "gender_male"
GENDER_UNKNOWN = "gender_unknown"

RACE = "race"
RACE_WHITE = "race_white"
RACE_BLACK = "race_black"
RACE_ASIAN = "race_asian"
RACE_OTHER = "race_other"
RACE_UNKNOWN = "race_unknown"

ETHNICITY = "ethnicity"
ETHNICITY_HISPANIC = "ethnicity_hispanic"
ETHNICITY_NOT_HISPANIC = "ethnicity_not_hispanic"
ETHNICITY_UNKNOWN = "ethnicity_unknown"

CODE_TYPE = "code_type"                        # which vocabulary a code came from
CODE_TYPE_ICD10 = "code_type_icd10"
CODE_TYPE_ICD9 = "code_type_icd9"
CODE_TYPE_SNOMED = "code_type_snomed"
CODE_TYPE_CPT = "code_type_cpt"

VALUE_MEMBERS = {
    ENCOUNTER_TYPE: (
        ENCOUNTER_IN_PERSON, ENCOUNTER_REMOTE, ENCOUNTER_IMPORTED, ENCOUNTER_EMPTY,
        ENCOUNTER_INCIDENTAL,
    ),
    BACKGROUND_CATEGORY: (
        BACKGROUND_FAMILY, BACKGROUND_FAMILY_VARIANT, BACKGROUND_SOCIAL, BACKGROUND_SURGERY,
        BACKGROUND_SURGERY_VARIANT, BACKGROUND_DEVICE, BACKGROUND_PREVENTIVE_CARE,
        BACKGROUND_TRAVEL, BACKGROUND_PREGNANCY,
    ),
    INTERVENTION_CATEGORY: (
        INTERVENTION_IMAGING, INTERVENTION_DIAGNOSTIC_STUDY, INTERVENTION_PROCEDURAL,
        INTERVENTION_DIAGNOSTICS, INTERVENTION_OTHER_TESTING, INTERVENTION_FINDING,
        INTERVENTION_EXAM, INTERVENTION_VACCINATION, INTERVENTION_EDUCATION,
    ),
    FINDING_CATEGORY: (FINDING_VITAL, FINDING_VITAL_VARIANT, FINDING_EXAM),
    ACTIVITY_TYPE: (ACTIVITY_ADMINISTRATION, ACTIVITY_PRESCRIPTION, ACTIVITY_DISCONTINUATION),
    GENDER: (GENDER_FEMALE, GENDER_MALE, GENDER_UNKNOWN),
    RACE: (RACE_WHITE, RACE_BLACK, RACE_ASIAN, RACE_OTHER, RACE_UNKNOWN),
    ETHNICITY: (ETHNICITY_HISPANIC, ETHNICITY_NOT_HISPANIC, ETHNICITY_UNKNOWN),
    CODE_TYPE: (CODE_TYPE_ICD10, CODE_TYPE_ICD9, CODE_TYPE_SNOMED, CODE_TYPE_CPT),
}

# --------------------------------------------------------------------------
# de-identification
#
# The delivery is de-identified before it ships: ages and body measurements are
# capped, dates of death are shifted, and geography and some diagnosis codes are
# replaced by a sentinel. The thresholds and sentinels themselves are properties
# of the delivery, so the configuration supplies them.
# --------------------------------------------------------------------------

DEIDENTIFICATION_KEYS = (
    "age_ceiling",           # oldest age the delivery will represent
    "body_mass_ceiling",     # BMI above this is reported at this value
    "weight_floor",          # weight outside these bounds is reported at them
    "weight_ceiling",
    "date_shift_min",        # date of death is moved forward by this many days
    "date_shift_max",
    "redacted_geography",    # stands in for a sparsely populated area
    "redacted_code",         # stands in for a re-identifying diagnosis code
)


# --------------------------------------------------------------------------
# fields, per entity
# --------------------------------------------------------------------------

FIELDS = {
    PERSON: (
        "birth_date", "country", "deceased_indicator", "ethnicity", "first_activity",
        "gender", "last_activity", "person_id", "race", "record_created", "record_updated",
        "state", "zip3"
    ),
    ORGANIZATION: (
        "first_activity", "imputed_specialty", "last_activity", "organization_id",
        "record_created", "record_updated", "site_id", "state", "zip3"
    ),
    PRACTITIONER: (
        "first_activity", "last_activity", "npi_credential", "npi_state",
        "npi_taxonomy_code", "npi_taxonomy_specialty", "organization_id", "practitioner_id",
        "record_created", "record_updated", "referring_flag", "site_id", "stated_specialty",
        "stated_state", "stated_zip3"
    ),
    ENCOUNTER: (
        "bmi", "diastolic", "encounter_date", "encounter_id", "encounter_reason",
        "encounter_type", "height", "organization_id", "person_id", "practitioner_id",
        "pulse", "record_created", "record_updated", "respiratory_rate", "site_id",
        "systolic", "temperature", "weight"
    ),
    CONDITION: (
        "begin_date", "condition_id", "condition_name", "encounter_id", "end_date",
        "organization_id", "person_id", "practitioner_id", "record_created",
        "record_updated", "site_id"
    ),
    CONDITION_CODE: (
        "background_id", "code", "code_type", "condition_code_id", "condition_id"
    ),
    CLINICAL_BACKGROUND: (
        "background_category", "background_id", "background_name", "begin_date", "end_date",
        "negation_flag", "organization_id", "person_id", "practitioner_id",
        "record_created", "record_updated", "site_id"
    ),
    TEST_RESULT: (
        "abnormal_flag", "condition_id", "cpt", "final_flag", "hcpcs", "historical_flag",
        "loinc", "onsite_flag", "order_date", "order_flag", "ordering_practitioner_id",
        "organization_id", "panel", "person_id", "record_created", "record_updated",
        "reference_range", "report_date", "result_flag", "result_modifier",
        "result_numeric", "result_snomed", "result_text", "site_id", "snomed",
        "specimen_date", "test", "test_description", "test_result_id", "units"
    ),
    FINDING: (
        "coded_reason", "condition_id", "cpt", "encounter_id", "event_date",
        "finding_category", "finding_id", "finding_name", "hcpcs", "historical_flag",
        "loinc", "organization_id", "person_id", "practitioner_id", "reason",
        "record_created", "record_updated", "result_numeric", "result_snomed",
        "result_text", "site_id", "snomed", "units"
    ),
    DRUG_EXPOSURE: (
        "activity_type", "administering_practitioner_id", "administration_date",
        "begin_date", "condition_id", "daily_amount", "daily_frequency",
        "dispense_as_written_flag", "documentation_date", "documenting_practitioner_id",
        "dose", "drug_exposure_id", "drug_name", "duration_days", "electronic_order_flag",
        "encounter_id", "end_date", "form", "frequency", "hcpcs",
        "managing_practitioner_id", "ndc", "organization_id", "over_the_counter_flag",
        "person_id", "prescribing_practitioner_id", "prescription_action",
        "prescription_date", "quantity", "record_created", "record_updated", "refills",
        "route", "rxnorm_code", "sig", "site_id", "stop_reason", "strength", "units"
    ),
    INTERVENTION: (
        "coded_reason", "comment", "condition_id", "cpt", "event_date", "hcpcs",
        "historical_flag", "intervention_category", "intervention_id", "intervention_name",
        "loinc", "onsite_flag", "order_date", "ordering_practitioner_id", "organization_id",
        "performing_practitioner_id", "person_id", "reason", "record_created",
        "record_updated", "result_date", "site_id", "snomed", "units"
    ),
    VACCINATION: (
        "administration_date", "administration_setting", "cpt", "cvx", "dose", "hcpcs",
        "historical_flag", "manufacturer", "mvx", "ndc", "organization_id", "person_id",
        "practitioner_id", "record_created", "record_updated", "refusal_reason", "route",
        "rxnorm_code", "series", "site_id", "snomed", "stock_supply_flag", "timing_window",
        "vaccination_id", "vaccination_status", "vaccine_name"
    ),
    INTOLERANCE: (
        "intolerance_category", "intolerance_id", "intolerance_name", "intolerance_type",
        "ndc", "onset_window", "organization_id", "person_id", "practitioner_id",
        "record_created", "record_updated", "resolution_flag", "rxnorm_code", "site_id",
        "snomed", "unii"
    ),
    INTOLERANCE_REACTION: (
        "intolerance_id", "reaction_id", "reaction_name", "reaction_snomed",
        "record_created", "record_updated", "severity"
    ),
    SERVICE_REQUEST: (
        "coded_reason", "completion_date", "condition_id", "cpt", "hcpcs",
        "historical_flag", "internal_routing_flag", "loinc", "organization_id", "person_id",
        "practitioner_id", "reason", "record_created", "record_updated",
        "service_request_id", "service_request_name", "site_id", "snomed",
        "transmitted_flag"
    ),
    DEATH: (
        "date_of_death", "death_record_id", "deceased_indicator", "person_id",
        "record_created", "record_updated"
    ),
}

# --------------------------------------------------------------------------
# cross-cutting groups
# --------------------------------------------------------------------------

# Fields carrying a value from a public clinical vocabulary (LOINC, SNOMED CT,
# NDC, RxNorm, CVX, MVX, UNII, CPT, HCPCS, ICD) or a geographic code. They parse
# as numbers but are categories, so profiling code must never summarise them as
# quantiles. The group is matched in every entity that has such a field.
CODE_FIELDS = (
    "code", "coded_reason", "cpt", "cvx", "hcpcs", "loinc", "mvx", "ndc",
    "npi_taxonomy_code", "reaction_snomed", "result_snomed", "rxnorm_code", "snomed",
    "stated_zip3", "unii", "zip3",
)

# Child field -> entity holding the matching primary key.
PARENT_OF = {
    "person_id": PERSON,
    "encounter_id": ENCOUNTER,
    "condition_id": CONDITION,
    "background_id": CLINICAL_BACKGROUND,
    "intolerance_id": INTOLERANCE,
    "organization_id": ORGANIZATION,
}
