"""Validate required modeling and analysis outputs."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data_loading import load_unlabeled_data
from src.features import EXCLUDED_MODEL_COLUMNS, get_raw_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FINAL_MODELS = ["logistic_elastic_net", "random_forest", "gradient_boosted_trees"]
REQUIRED_ANALYSIS_FILES = [
    "stage1/tables/cohort_accounting.csv",
    "stage1/tables/feature_accounting.csv",
    "stage1/tables/missingness_summary.csv",
    "stage1/tables/survivor_vs_nonsurvivor_summary.csv",
    "stage1/tables/high_correlation_pairs.csv",
    "stage2/tables/final_model_comparison.csv",
    "stage2/tables/final_threshold_metrics.csv",
    "stage2/tables/calibration_by_decile.csv",
    "stage3/tables/logistic_elastic_net_coefficients.csv",
    "stage3/tables/permutation_importance.csv",
    "stage3/tables/local_case_selection.csv",
    "stage3/tables/apache_variable_overlap.csv",
]


def validate() -> list[str]:
    messages = []
    unlabeled = load_unlabeled_data()
    features = get_raw_feature_columns(unlabeled.columns)
    leakage = [column for column in EXCLUDED_MODEL_COLUMNS if column in features]
    if leakage:
        messages.append(f"FAIL: leakage columns found in feature list: {leakage}")
    else:
        messages.append("PASS: leakage columns excluded from feature list.")

    X_unlabeled = unlabeled[features].head(5)
    for model in FINAL_MODELS:
        prediction_path = OUTPUT_DIR / "predictions" / f"{model}_unlabeled_predictions.csv"
        predictions = pd.read_csv(prediction_path)
        checks = [
            predictions.shape == (39308, 2),
            predictions.columns.tolist() == ["encounter_id", "hospital_death"],
            predictions["hospital_death"].notna().all(),
            predictions["hospital_death"].between(0, 1).all(),
        ]
        if all(checks):
            messages.append(f"PASS: prediction file valid for {model}.")
        else:
            messages.append(f"FAIL: prediction file invalid for {model}.")

        model_path = OUTPUT_DIR / "models" / f"{model}.joblib"
        loaded_model = joblib.load(model_path)
        probabilities = loaded_model.predict_proba(X_unlabeled)[:, 1]
        if len(probabilities) == len(X_unlabeled):
            messages.append(f"PASS: saved model reloads and predicts for {model}.")
        else:
            messages.append(f"FAIL: saved model prediction length mismatch for {model}.")

        oof_path = OUTPUT_DIR / "reports" / f"{model}_oof_predictions.csv"
        oof = pd.read_csv(oof_path)
        required_columns = {"observed_hospital_death", "predicted_mortality_probability"}
        if required_columns <= set(oof.columns):
            messages.append(f"PASS: OOF file has required columns for {model}.")
        else:
            messages.append(f"FAIL: OOF file missing required columns for {model}.")

    apache = pd.read_csv(OUTPUT_DIR / "stage2" / "tables" / "final_model_comparison.csv")
    if "apache_4a_hospital_death_prob" in apache["model"].values:
        messages.append("PASS: APACHE benchmark included in final comparison.")
    else:
        messages.append("FAIL: APACHE benchmark missing from final comparison.")

    for relative_path in REQUIRED_ANALYSIS_FILES:
        path = OUTPUT_DIR / relative_path
        if path.exists():
            messages.append(f"PASS: required output exists: {relative_path}")
        else:
            messages.append(f"FAIL: required output missing: {relative_path}")

    return messages


def main() -> None:
    messages = validate()
    report_path = OUTPUT_DIR / "validation_report.txt"
    report_path.write_text("\n".join(messages) + "\n", encoding="utf-8")
    print("\n".join(messages))


if __name__ == "__main__":
    main()

