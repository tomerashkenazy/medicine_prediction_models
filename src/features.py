"""Feature definitions for stage 2 mortality prediction."""

TARGET_COLUMN = "hospital_death"
ID_COLUMN = "encounter_id"
PATIENT_ID_COLUMN = "patient_id"

STAGE1_NUMERIC_FEATURES = [
    "age",
    "bmi",
    "d1_heartrate_max",
    "d1_heartrate_min",
    "d1_mbp_max",
    "d1_mbp_min",
    "d1_spo2_max",
    "d1_spo2_min",
    "d1_lactate_max",
    "d1_lactate_min",
    "d1_creatinine_max",
    "d1_creatinine_min",
    "d1_glucose_max",
    "d1_glucose_min",
]

STAGE1_BINARY_FEATURES = [
    "immunosuppression",
    "hepatic_failure",
]

MODEL_FEATURES = STAGE1_NUMERIC_FEATURES + STAGE1_BINARY_FEATURES

APACHE_BENCHMARK_COLUMN = "apache_4a_hospital_death_prob"
EXCLUDED_APACHE_PREDICTION_COLUMNS = [
    "apache_4a_hospital_death_prob",
    "apache_4a_icu_death_prob",
]

EXCLUDED_MODEL_COLUMNS = [
    TARGET_COLUMN,
    ID_COLUMN,
    PATIENT_ID_COLUMN,
    *EXCLUDED_APACHE_PREDICTION_COLUMNS,
]

CATEGORICAL_CODE_FEATURES = [
    "hospital_id",
    "icu_id",
    "apache_2_diagnosis",
    "apache_3j_diagnosis",
]

REDUCED_LOGISTIC_EXCLUDED_FEATURES = CATEGORICAL_CODE_FEATURES.copy()

FEATURE_REDUCTION_SETS = {
    "reduced_logistic_no_codes": REDUCED_LOGISTIC_EXCLUDED_FEATURES,
}


def get_raw_feature_columns(
    columns,
    additional_excluded_columns: list[str] | None = None,
) -> list[str]:
    """Return all raw model inputs, excluding labels, row IDs, and model outputs."""
    excluded = set(EXCLUDED_MODEL_COLUMNS)
    if additional_excluded_columns:
        excluded.update(additional_excluded_columns)
    return [column for column in columns if column not in excluded]
