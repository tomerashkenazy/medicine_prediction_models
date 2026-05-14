"""Stage 1 data preparation and descriptive analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cohort import apply_stage1_cohort
from src.data_loading import load_training_data
from src.features import EXCLUDED_MODEL_COLUMNS, TARGET_COLUMN, get_raw_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage1"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"


DESCRIPTIVE_COLUMNS = [
    "age",
    "bmi",
    "apache_4a_hospital_death_prob",
    "d1_heartrate_max",
    "d1_mbp_min",
    "d1_spo2_min",
    "d1_lactate_max",
    "d1_creatinine_max",
    "d1_bun_max",
    "gcs_motor_apache",
    "ventilated_apache",
    "intubated_apache",
    "hepatic_failure",
    "immunosuppression",
]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_cohort_accounting(raw_df: pd.DataFrame, cohort_df: pd.DataFrame) -> None:
    pediatric_rows = int(((raw_df["age"] < 18) & raw_df["age"].notna()).sum())
    missing_age_rows = int(raw_df["age"].isna().sum())
    unique_patients = int(raw_df["patient_id"].nunique()) if "patient_id" in raw_df else np.nan
    accounting = pd.DataFrame(
        [
            {"measure": "raw_training_rows", "value": len(raw_df)},
            {"measure": "known_pediatric_rows_age_lt_18_excluded", "value": pediatric_rows},
            {"measure": "missing_age_rows_retained", "value": missing_age_rows},
            {"measure": "unique_patient_id_count", "value": unique_patients},
            {"measure": "final_modeling_rows", "value": len(cohort_df)},
            {"measure": "under_4_hour_icu_los_filter_applied", "value": "no"},
            {
                "measure": "under_4_hour_icu_los_filter_reason",
                "value": "No direct ICU length-of-stay column; pre_icu_los_days is pre-ICU time.",
            },
        ]
    )
    accounting.to_csv(TABLE_DIR / "cohort_accounting.csv", index=False)


def save_feature_accounting(cohort_df: pd.DataFrame) -> None:
    feature_columns = get_raw_feature_columns(cohort_df.columns)
    categorical_columns = cohort_df[feature_columns].select_dtypes(include=["object"]).columns
    numeric_columns = cohort_df[feature_columns].select_dtypes(include=[np.number]).columns
    rows = [
        {"measure": "total_raw_columns", "value": len(cohort_df.columns)},
        {"measure": "included_model_feature_count", "value": len(feature_columns)},
        {"measure": "numeric_feature_count", "value": len(numeric_columns)},
        {"measure": "object_categorical_feature_count", "value": len(categorical_columns)},
    ]
    rows.extend({"measure": f"excluded_{column}", "value": column} for column in EXCLUDED_MODEL_COLUMNS)
    pd.DataFrame(rows).to_csv(TABLE_DIR / "feature_accounting.csv", index=False)


def save_missingness(cohort_df: pd.DataFrame) -> None:
    missingness = (
        cohort_df.isna()
        .mean()
        .rename("missing_fraction")
        .reset_index()
        .rename(columns={"index": "variable"})
        .sort_values("missing_fraction", ascending=False)
    )
    missingness["missing_percent"] = 100 * missingness["missing_fraction"]
    missingness.to_csv(TABLE_DIR / "missingness_summary.csv", index=False)

    top = missingness.head(30).sort_values("missing_percent")
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(top["variable"], top["missing_percent"], color="#4C78A8")
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Top 30 Variables by Missingness")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "top_missingness.png", dpi=150)
    plt.close(fig)


def _summarize_continuous(df: pd.DataFrame, column: str) -> dict[str, float | str]:
    values = df[column].dropna()
    return {
        "variable": column,
        "type": "continuous",
        "n_nonmissing": int(values.shape[0]),
        "median": values.median(),
        "q1": values.quantile(0.25),
        "q3": values.quantile(0.75),
        "mean": values.mean(),
        "std": values.std(),
    }


def _summarize_binary(df: pd.DataFrame, column: str) -> dict[str, float | str]:
    values = df[column].dropna()
    return {
        "variable": column,
        "type": "binary",
        "n_nonmissing": int(values.shape[0]),
        "positive_count": int((values == 1).sum()),
        "positive_percent": 100 * float((values == 1).mean()) if len(values) else np.nan,
    }


def save_survivor_summary(cohort_df: pd.DataFrame) -> None:
    rows = []
    for outcome, group in cohort_df.groupby(TARGET_COLUMN):
        outcome_label = "non_survivor" if outcome == 1 else "survivor"
        for column in DESCRIPTIVE_COLUMNS:
            if column not in group.columns:
                continue
            unique_values = set(group[column].dropna().unique().tolist())
            if unique_values <= {0, 1}:
                summary = _summarize_binary(group, column)
            else:
                summary = _summarize_continuous(group, column)
            summary["outcome_group"] = outcome_label
            rows.append(summary)
    pd.DataFrame(rows).to_csv(TABLE_DIR / "survivor_vs_nonsurvivor_summary.csv", index=False)


def save_collinearity(cohort_df: pd.DataFrame) -> None:
    feature_columns = get_raw_feature_columns(cohort_df.columns)
    numeric_df = cohort_df[feature_columns].select_dtypes(include=[np.number])
    # Drop constants and extremely sparse columns for a stable correlation figure.
    numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]
    corr = numeric_df.corr(method="spearman", min_periods=1000)

    pairs = []
    columns = corr.columns.to_list()
    values = corr.to_numpy()
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            value = values[i, j]
            if pd.notna(value) and abs(value) >= 0.8:
                pairs.append(
                    {
                        "variable_1": left,
                        "variable_2": columns[j],
                        "spearman_rho": value,
                        "abs_spearman_rho": abs(value),
                    }
                )
    pd.DataFrame(pairs).sort_values("abs_spearman_rho", ascending=False).to_csv(
        TABLE_DIR / "high_correlation_pairs.csv",
        index=False,
    )

    # Plot the most complete/highly observed numeric variables to keep labels readable.
    selected = numeric_df.isna().mean().sort_values().head(50).index
    plot_corr = numeric_df[selected].corr(method="spearman", min_periods=1000)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(plot_corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(selected)))
    ax.set_yticks(range(len(selected)))
    ax.set_xticklabels(selected, rotation=90, fontsize=6)
    ax.set_yticklabels(selected, fontsize=6)
    ax.set_title("Spearman Correlation Heatmap: 50 Most Complete Numeric Features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "numeric_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    raw_df = load_training_data()
    cohort_df = apply_stage1_cohort(raw_df)
    save_cohort_accounting(raw_df, cohort_df)
    save_feature_accounting(cohort_df)
    save_missingness(cohort_df)
    save_survivor_summary(cohort_df)
    save_collinearity(cohort_df)


if __name__ == "__main__":
    main()

