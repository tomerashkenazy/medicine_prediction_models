"""SHAP-only local explanations for the reduced no-code models."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse

from src.analysis_stage3 import raw_group_name
from src.cohort import apply_stage1_cohort
from src.data_loading import load_training_data
from src.features import TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "outputs" / "reduced_logistic_no_codes"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "shap"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODELS = [
    "reduced_logistic_elastic_net",
    "reduced_gradient_boosted_trees",
    "reduced_random_forest",
]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def transformed_frame(preprocessor, X_data: pd.DataFrame) -> pd.DataFrame:
    transformed = preprocessor.transform(X_data)
    values = transformed.toarray() if sparse.issparse(transformed) else transformed
    return pd.DataFrame(values, columns=preprocessor.get_feature_names_out(), index=X_data.index)


def class_one_shap_values(values: object) -> np.ndarray:
    values_array = np.asarray(values)
    if values_array.ndim == 3:
        return values_array[:, :, 1]
    if values_array.ndim == 2:
        return values_array
    raise ValueError(f"Unsupported SHAP value shape: {values_array.shape}")


def choose_cases(reference_model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    probabilities = reference_model.predict_proba(X)[:, 1]
    predictions = pd.DataFrame(
        {
            "row_index": np.arange(len(X)),
            "observed_hospital_death": y.to_numpy(),
            "predicted_mortality_probability": probabilities,
        }
    )
    cases = [
        (
            "true_positive",
            predictions[(predictions["observed_hospital_death"] == 1) & (predictions["predicted_mortality_probability"] >= 0.8)]
            .sort_values("predicted_mortality_probability", ascending=False)
            .head(1),
        ),
        (
            "true_negative",
            predictions[(predictions["observed_hospital_death"] == 0) & (predictions["predicted_mortality_probability"] <= 0.05)]
            .sort_values("predicted_mortality_probability")
            .head(1),
        ),
        (
            "false_positive",
            predictions[(predictions["observed_hospital_death"] == 0) & (predictions["predicted_mortality_probability"] >= 0.8)]
            .sort_values("predicted_mortality_probability", ascending=False)
            .head(1),
        ),
        (
            "false_negative",
            predictions[(predictions["observed_hospital_death"] == 1) & (predictions["predicted_mortality_probability"] <= 0.2)]
            .sort_values("predicted_mortality_probability")
            .head(1),
        ),
    ]
    rows = []
    for case_type, case_df in cases:
        if not case_df.empty:
            row = case_df.iloc[0].to_dict()
            row["case_type"] = case_type
            rows.append(row)
    selected = pd.DataFrame(rows)
    selected.to_csv(TABLE_DIR / "reduced_shap_case_selection.csv", index=False)
    return selected


def build_explainer(model_name: str, classifier, background: pd.DataFrame):
    if "logistic" in model_name:
        return shap.LinearExplainer(classifier, background)
    return shap.TreeExplainer(classifier)


def save_model_shap(model_name: str, X: pd.DataFrame, selected_cases: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    background = X.sample(min(500, len(X)), random_state=42)
    background_transformed = transformed_frame(preprocessor, background)
    explainer = build_explainer(model_name, classifier, background_transformed)
    rows = []

    for _, case in selected_cases.iterrows():
        row_index = int(case["row_index"])
        case_X = X.iloc[[row_index]]
        case_transformed = transformed_frame(preprocessor, case_X)
        explanation = explainer(case_transformed)
        shap_values = class_one_shap_values(explanation.values).ravel()
        predicted_probability = float(model.predict_proba(case_X)[0, 1])
        shap_df = pd.DataFrame(
            {
                "model": model_name,
                "case_type": case["case_type"],
                "row_index": row_index,
                "transformed_feature": case_transformed.columns,
                "shap_value": shap_values,
            }
        )
        shap_df["raw_feature"] = shap_df["transformed_feature"].apply(
            lambda name: raw_group_name(name, X.columns.to_list())
        )
        grouped = (
            shap_df.groupby(["model", "case_type", "row_index", "raw_feature"], as_index=False)[
                "shap_value"
            ]
            .sum()
            .sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
        )
        rows.append(grouped)

        top = grouped.head(12).sort_values("shap_value")
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = np.where(top["shap_value"] >= 0, "#D55E00", "#0072B2")
        ax.barh(top["raw_feature"], top["shap_value"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"{model_name}: {case['case_type']} "
            f"(predicted risk={predicted_probability:.1%})"
        )
        ax.set_xlabel("SHAP value for hospital death prediction")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{model_name}_shap_{case['case_type']}.png", dpi=150)
        plt.close(fig)

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ensure_dirs()
    reference_model = joblib.load(MODEL_DIR / "reduced_logistic_elastic_net.joblib")
    feature_columns = reference_model.named_steps["preprocessor"].feature_names_in_.tolist()
    df = apply_stage1_cohort(load_training_data())
    X = df[feature_columns]
    y = df[TARGET_COLUMN].astype(int)
    selected_cases = choose_cases(reference_model, X, y)
    all_rows = []
    for model_name in MODELS:
        all_rows.append(save_model_shap(model_name, X, selected_cases))
    pd.concat(all_rows, ignore_index=True).to_csv(
        TABLE_DIR / "reduced_local_shap_values.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
