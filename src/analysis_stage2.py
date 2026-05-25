"""Stage 2 final model comparison, thresholds, and calibration analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from src.cohort import apply_stage1_cohort
from src.data_loading import load_training_data
from src.evaluation import probability_metrics, summarize_thresholds
from src.features import APACHE_BENCHMARK_COLUMN, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
DEFAULT_FINAL_MODELS = ["logistic_elastic_net", "random_forest", "gradient_boosted_trees"]
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]


def ensure_dirs(table_dir: Path, figure_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)


def load_oof_predictions(report_dir: Path, final_models: list[str]) -> dict[str, pd.DataFrame]:
    predictions = {}
    for model in final_models:
        path = report_dir / f"{model}_oof_predictions.csv"
        predictions[model] = pd.read_csv(path)
    return predictions


def apache_predictions() -> pd.DataFrame:
    df = apply_stage1_cohort(load_training_data())
    apache = df[[TARGET_COLUMN, APACHE_BENCHMARK_COLUMN]].dropna()
    apache = apache[
        (apache[APACHE_BENCHMARK_COLUMN] >= 0) & (apache[APACHE_BENCHMARK_COLUMN] <= 1)
    ]
    return pd.DataFrame(
        {
            "observed_hospital_death": apache[TARGET_COLUMN].astype(int).to_numpy(),
            "predicted_mortality_probability": apache[APACHE_BENCHMARK_COLUMN].to_numpy(),
        }
    )


def save_final_comparison(
    oof_by_model: dict[str, pd.DataFrame],
    apache_df: pd.DataFrame,
    table_dir: Path,
) -> None:
    rows = []
    for model, df in oof_by_model.items():
        metrics = probability_metrics(
            df["observed_hospital_death"],
            df["predicted_mortality_probability"],
        )
        metrics["model"] = model
        metrics["n"] = len(df)
        rows.append(metrics)
    apache_metrics = probability_metrics(
        apache_df["observed_hospital_death"],
        apache_df["predicted_mortality_probability"],
    )
    apache_metrics["model"] = APACHE_BENCHMARK_COLUMN
    apache_metrics["n"] = len(apache_df)
    rows.append(apache_metrics)
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    comparison.to_csv(table_dir / "final_model_comparison.csv", index=False)


def save_combined_curves(
    oof_by_model: dict[str, pd.DataFrame],
    apache_df: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for model, df in oof_by_model.items():
        RocCurveDisplay.from_predictions(
            df["observed_hospital_death"],
            df["predicted_mortality_probability"],
            name=model,
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
        apache_df["observed_hospital_death"],
        apache_df["predicted_mortality_probability"],
        name="APACHE benchmark",
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Final ROC Curves")
    fig.tight_layout()
    fig.savefig(figure_dir / "final_roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for model, df in oof_by_model.items():
        PrecisionRecallDisplay.from_predictions(
            df["observed_hospital_death"],
            df["predicted_mortality_probability"],
            name=model,
            ax=ax,
        )
    PrecisionRecallDisplay.from_predictions(
        apache_df["observed_hospital_death"],
        apache_df["predicted_mortality_probability"],
        name="APACHE benchmark",
        ax=ax,
    )
    ax.set_title("Final Precision-Recall Curves")
    fig.tight_layout()
    fig.savefig(figure_dir / "final_precision_recall_curves.png", dpi=150)
    plt.close(fig)


def save_thresholds(oof_by_model: dict[str, pd.DataFrame], table_dir: Path) -> None:
    rows = []
    for model, df in oof_by_model.items():
        table = summarize_thresholds(
            df["observed_hospital_death"],
            df["predicted_mortality_probability"],
            THRESHOLDS,
        )
        table.insert(0, "model", model)
        table["flagged_high_risk"] = table["tp"] + table["fp"]
        table["flagged_high_risk_percent"] = 100 * table["flagged_high_risk"] / len(df)
        rows.append(table)
    thresholds = pd.concat(rows, ignore_index=True)
    thresholds.to_csv(table_dir / "final_threshold_metrics.csv", index=False)

    recommended = thresholds[thresholds["threshold"].isin([0.10, 0.20, 0.30])].copy()
    recommended.to_csv(table_dir / "recommended_cutoff_summary.csv", index=False)


def calibration_by_decile(df: pd.DataFrame, model: str) -> pd.DataFrame:
    out = df.copy()
    out["risk_decile"] = pd.qcut(
        out["predicted_mortality_probability"],
        q=10,
        labels=False,
        duplicates="drop",
    )
    grouped = out.groupby("risk_decile", observed=True).agg(
        n=("observed_hospital_death", "size"),
        mean_predicted_risk=("predicted_mortality_probability", "mean"),
        observed_mortality=("observed_hospital_death", "mean"),
        min_predicted_risk=("predicted_mortality_probability", "min"),
        max_predicted_risk=("predicted_mortality_probability", "max"),
    )
    grouped.insert(0, "model", model)
    return grouped.reset_index()


def save_calibration(
    oof_by_model: dict[str, pd.DataFrame],
    final_models: list[str],
    table_dir: Path,
    figure_dir: Path,
) -> None:
    deciles = pd.concat(
        [calibration_by_decile(df, model) for model, df in oof_by_model.items()],
        ignore_index=True,
    )
    deciles.to_csv(table_dir / "calibration_by_decile.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    for model in final_models:
        data = deciles[deciles["model"] == model]
        ax.plot(data["mean_predicted_risk"], data["observed_mortality"], marker="o", label=model)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Mean predicted mortality")
    ax.set_ylabel("Observed mortality")
    ax.set_title("Calibration by Predicted-Risk Decile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "calibration_curves.png", dpi=150)
    plt.close(fig)


def save_risk_distribution(
    oof_by_model: dict[str, pd.DataFrame],
    final_models: list[str],
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(len(final_models), 1, figsize=(8, 10), sharex=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 1, 41)
    for ax, model in zip(axes, final_models):
        df = oof_by_model[model]
        survivors = df[df["observed_hospital_death"] == 0]["predicted_mortality_probability"]
        deaths = df[df["observed_hospital_death"] == 1]["predicted_mortality_probability"]
        ax.hist(survivors, bins=bins, alpha=0.6, density=True, label="Survivors")
        ax.hist(deaths, bins=bins, alpha=0.6, density=True, label="Non-survivors")
        ax.set_title(model)
        ax.set_ylabel("Density")
        ax.legend()
    axes[-1].set_xlabel("Predicted mortality probability")
    fig.tight_layout()
    fig.savefig(figure_dir / "risk_distribution_by_outcome.png", dpi=150)
    plt.close(fig)


def run_stage2_analysis(
    *,
    report_dir: Path = REPORT_DIR,
    output_dir: Path = OUTPUT_DIR,
    final_models: list[str] | None = None,
) -> None:
    final_models = final_models or DEFAULT_FINAL_MODELS
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    ensure_dirs(table_dir, figure_dir)
    oof_by_model = load_oof_predictions(report_dir, final_models)
    apache_df = apache_predictions()
    save_final_comparison(oof_by_model, apache_df, table_dir)
    save_combined_curves(oof_by_model, apache_df, figure_dir)
    save_thresholds(oof_by_model, table_dir)
    save_calibration(oof_by_model, final_models, table_dir, figure_dir)
    save_risk_distribution(oof_by_model, final_models, figure_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--final-model",
        action="append",
        default=[],
        help="OOF model name to include. Can be repeated. Defaults to original stage 2 models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stage2_analysis(
        report_dir=args.report_dir,
        output_dir=args.output_dir,
        final_models=args.final_model or None,
    )


if __name__ == "__main__":
    main()
