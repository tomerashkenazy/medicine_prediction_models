"""Stage 3 explainability analyses."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from src.cohort import apply_stage1_cohort
from src.data_loading import load_training_data
from src.features import TARGET_COLUMN, get_raw_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
FINAL_MODELS = ["logistic_elastic_net", "random_forest", "gradient_boosted_trees"]
APACHE_CONCEPT_KEYWORDS = {
    "age": ["age"],
    "diagnosis/body system": ["diagnosis", "bodysystem"],
    "gcs": ["gcs"],
    "ventilation/intubation": ["ventilated", "intubated", "fio2", "pao2", "paco2"],
    "map/blood pressure": ["map", "mbp", "diasbp", "sysbp"],
    "renal/urine": ["creatinine", "bun", "urine"],
    "labs/metabolic": ["wbc", "bilirubin", "albumin", "glucose", "sodium", "potassium"],
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def modeling_data() -> tuple[pd.DataFrame, pd.Series]:
    df = apply_stage1_cohort(load_training_data())
    feature_columns = get_raw_feature_columns(df.columns)
    return df[feature_columns], df[TARGET_COLUMN].astype(int)


def raw_group_name(feature_name: str, raw_columns: list[str]) -> str:
    if feature_name.startswith("missingindicator_"):
        return feature_name.replace("missingindicator_", "")
    if feature_name in raw_columns:
        return feature_name
    for column in sorted(raw_columns, key=len, reverse=True):
        if feature_name.startswith(f"{column}_"):
            return column
    return feature_name


def save_logistic_coefficients(X: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODEL_DIR / "logistic_elastic_net.joblib")
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_.ravel()
    coef_df = pd.DataFrame(
        {
            "transformed_feature": feature_names,
            "coefficient": coefficients,
        }
    )
    coef_df["raw_feature"] = coef_df["transformed_feature"].apply(
        lambda name: raw_group_name(name, X.columns.to_list())
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(TABLE_DIR / "logistic_elastic_net_coefficients.csv", index=False)

    top_positive = coef_df.sort_values("coefficient", ascending=False).head(15)
    top_negative = coef_df.sort_values("coefficient").head(15)
    plot_df = pd.concat([top_negative, top_positive]).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = np.where(plot_df["coefficient"] >= 0, "#D55E00", "#0072B2")
    ax.barh(plot_df["transformed_feature"], plot_df["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Elastic-Net Logistic Regression: Top Coefficients")
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "logistic_top_coefficients.png", dpi=150)
    plt.close(fig)
    return coef_df


def save_permutation_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    sample_size = min(5000, len(X))
    X_sample = X.sample(sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    rows = []
    for model_name in FINAL_MODELS:
        model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
        result = permutation_importance(
            model,
            X_sample,
            y_sample,
            scoring="roc_auc",
            n_repeats=2,
            random_state=42,
            n_jobs=1,
        )
        model_df = pd.DataFrame(
            {
                "model": model_name,
                "raw_feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        rows.append(model_df)
    importance = pd.concat(rows, ignore_index=True)
    importance.to_csv(TABLE_DIR / "permutation_importance.csv", index=False)

    top = (
        importance.groupby("raw_feature", as_index=False)["importance_mean"]
        .mean()
        .sort_values("importance_mean", ascending=False)
        .head(20)
    )
    plot_df = importance[importance["raw_feature"].isin(top["raw_feature"])]
    fig, ax = plt.subplots(figsize=(10, 8))
    width = 0.25
    y_positions = np.arange(len(top))
    for offset, model_name in zip([-width, 0, width], FINAL_MODELS):
        values = (
            plot_df[plot_df["model"] == model_name]
            .set_index("raw_feature")
            .reindex(top["raw_feature"])["importance_mean"]
            .fillna(0)
        )
        ax.barh(y_positions + offset, values, height=width, label=model_name)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top["raw_feature"])
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC drop after permutation")
    ax.set_title("Permutation Importance Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "permutation_importance_comparison.png", dpi=150)
    plt.close(fig)
    return importance


def save_partial_dependence(X: pd.DataFrame) -> None:
    model = joblib.load(MODEL_DIR / "logistic_elastic_net.joblib")
    preferred = [
        "age",
        "gcs_motor_apache",
        "d1_creatinine_max",
        "d1_bun_max",
        "d1_mbp_min",
        "d1_lactate_max",
    ]
    features = [feature for feature in preferred if feature in X.columns]
    sample = X.sample(min(5000, len(X)), random_state=42)
    for feature in features:
        fig, ax = plt.subplots(figsize=(6, 5))
        PartialDependenceDisplay.from_estimator(model, sample, [feature], ax=ax)
        ax.set_title(f"Partial Dependence: {feature}")
        fig.tight_layout()
        safe_name = feature.replace("/", "_")
        fig.savefig(FIGURE_DIR / f"partial_dependence_{safe_name}.png", dpi=150)
        plt.close(fig)


def choose_local_cases() -> pd.DataFrame:
    oof = pd.read_csv(REPORT_DIR / "logistic_elastic_net_oof_predictions.csv")
    y = oof["observed_hospital_death"]
    p = oof["predicted_mortality_probability"]
    cases = [
        ("true_positive", oof[(y == 1) & (p >= 0.8)].sort_values(p.name, ascending=False).head(1)),
        ("true_negative", oof[(y == 0) & (p <= 0.05)].sort_values(p.name).head(1)),
        ("false_positive", oof[(y == 0) & (p >= 0.8)].sort_values(p.name, ascending=False).head(1)),
        ("false_negative", oof[(y == 1) & (p <= 0.2)].sort_values(p.name).head(1)),
    ]
    rows = []
    for label, case_df in cases:
        if not case_df.empty:
            row = case_df.iloc[0].to_dict()
            row["case_type"] = label
            rows.append(row)
    selected = pd.DataFrame(rows)
    selected.to_csv(TABLE_DIR / "local_case_selection.csv", index=False)
    return selected


def save_local_contributions(X: pd.DataFrame, selected_cases: pd.DataFrame) -> None:
    model = joblib.load(MODEL_DIR / "logistic_elastic_net.joblib")
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    coefficients = classifier.coef_.ravel()
    feature_names = preprocessor.get_feature_names_out()
    rows = []
    for _, case in selected_cases.iterrows():
        row_index = int(case["row_index"])
        transformed = preprocessor.transform(X.iloc[[row_index]])
        values = transformed.toarray().ravel() if sparse.issparse(transformed) else transformed.ravel()
        contributions = values * coefficients
        contrib_df = pd.DataFrame(
            {
                "case_type": case["case_type"],
                "row_index": row_index,
                "transformed_feature": feature_names,
                "contribution": contributions,
            }
        )
        contrib_df["raw_feature"] = contrib_df["transformed_feature"].apply(
            lambda name: raw_group_name(name, X.columns.to_list())
        )
        grouped = (
            contrib_df.groupby(["case_type", "row_index", "raw_feature"], as_index=False)[
                "contribution"
            ]
            .sum()
            .sort_values("contribution", key=lambda s: s.abs(), ascending=False)
        )
        rows.append(grouped)

        top = grouped.head(20).sort_values("contribution")
        fig, ax = plt.subplots(figsize=(8, 7))
        colors = np.where(top["contribution"] >= 0, "#D55E00", "#0072B2")
        ax.barh(top["raw_feature"], top["contribution"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Local Contributions: {case['case_type']}")
        ax.set_xlabel("Log-odds contribution")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"local_case_{case['case_type']}.png", dpi=150)
        plt.close(fig)
    pd.concat(rows, ignore_index=True).to_csv(
        TABLE_DIR / "local_case_contributions.csv",
        index=False,
    )


def save_apache_overlap(coef_df: pd.DataFrame, importance: pd.DataFrame) -> None:
    top_coefficients = set(coef_df.head(50)["raw_feature"])
    top_importance = set(
        importance.groupby("raw_feature")["importance_mean"]
        .mean()
        .sort_values(ascending=False)
        .head(50)
        .index
    )
    variables = sorted(top_coefficients | top_importance)
    rows = []
    for variable in variables:
        lower = variable.lower()
        concepts = [
            concept
            for concept, keywords in APACHE_CONCEPT_KEYWORDS.items()
            if any(keyword in lower for keyword in keywords)
        ]
        rows.append(
            {
                "raw_feature": variable,
                "in_top_logistic_coefficients": variable in top_coefficients,
                "in_top_permutation_importance": variable in top_importance,
                "apache_style_concept": "; ".join(concepts) if concepts else "additional signal",
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "apache_variable_overlap.csv", index=False)


def main() -> None:
    ensure_dirs()
    X, y = modeling_data()
    coef_df = save_logistic_coefficients(X)
    importance = save_permutation_importance(X, y)
    save_partial_dependence(X)
    selected_cases = choose_local_cases()
    save_local_contributions(X, selected_cases)
    save_apache_overlap(coef_df, importance)


if __name__ == "__main__":
    main()

