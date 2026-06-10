"""Bootstrap uncertainty estimates for mortality prediction performance."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.cohort import apply_stage1_cohort
from src.data_loading import load_training_data
from src.evaluation import probability_metrics, threshold_metrics
from src.features import APACHE_BENCHMARK_COLUMN, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reduced_logistic_sklearn172" / "reports"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bootstrap_uncertainty"
DEFAULT_MODELS = [
    "reduced_logistic_elastic_net",
    "reduced_random_forest",
    "reduced_gradient_boosted_trees",
]
DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]
RECOMMENDED_MODEL = "reduced_logistic_elastic_net"
RECOMMENDED_THRESHOLD = 0.30
OBSERVED_COLUMN = "observed_hospital_death"
PREDICTED_COLUMN = "predicted_mortality_probability"
ROW_INDEX_COLUMN = "row_index"
PROBABILITY_METRICS = ["roc_auc", "pr_auc", "brier_score"]
THRESHOLD_METRICS = [
    "sensitivity_recall",
    "specificity",
    "precision",
    "f1",
    "flagged_high_risk_percent",
]


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir


def _validate_prediction_frame(df: pd.DataFrame, path: Path) -> None:
    required_columns = {ROW_INDEX_COLUMN, OBSERVED_COLUMN, PREDICTED_COLUMN}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if df[OBSERVED_COLUMN].isna().any():
        raise ValueError(f"{path} contains missing observed outcomes.")
    if df[PREDICTED_COLUMN].isna().any():
        raise ValueError(f"{path} contains missing predicted probabilities.")
    invalid_probabilities = ~df[PREDICTED_COLUMN].between(0, 1)
    if invalid_probabilities.any():
        raise ValueError(f"{path} contains probabilities outside [0, 1].")


def load_oof_predictions(report_dir: Path, models: list[str]) -> dict[str, pd.DataFrame]:
    predictions = {}
    for model in models:
        path = report_dir / f"{model}_oof_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing OOF prediction file: {path}")
        df = pd.read_csv(path)
        _validate_prediction_frame(df, path)
        predictions[model] = df
    return predictions


def load_apache_predictions() -> pd.DataFrame:
    df = apply_stage1_cohort(load_training_data()).reset_index(drop=True)
    apache = df[[TARGET_COLUMN, APACHE_BENCHMARK_COLUMN]].copy()
    apache[ROW_INDEX_COLUMN] = np.arange(len(apache))
    apache = apache.dropna()
    apache = apache[
        (apache[APACHE_BENCHMARK_COLUMN] >= 0)
        & (apache[APACHE_BENCHMARK_COLUMN] <= 1)
    ]
    return pd.DataFrame(
        {
            ROW_INDEX_COLUMN: apache[ROW_INDEX_COLUMN].to_numpy(),
            OBSERVED_COLUMN: apache[TARGET_COLUMN].astype(int).to_numpy(),
            PREDICTED_COLUMN: apache[APACHE_BENCHMARK_COLUMN].to_numpy(),
        }
    )


def safe_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    if np.unique(y_true).size < 2:
        return {metric: np.nan for metric in PROBABILITY_METRICS}
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def threshold_summary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    metrics = threshold_metrics(y_true, y_prob, threshold=threshold)
    flagged_high_risk = metrics["tp"] + metrics["fp"]
    metrics["flagged_high_risk"] = flagged_high_risk
    metrics["flagged_high_risk_percent"] = 100 * flagged_high_risk / len(y_true)
    return metrics


def percentile_interval(values: pd.Series, confidence_level: float) -> tuple[float, float]:
    alpha = 1 - confidence_level
    clean = values.dropna()
    if clean.empty:
        return np.nan, np.nan
    lower, upper = np.quantile(clean, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def summarize_bootstrap_samples(
    *,
    point_estimates: pd.DataFrame,
    bootstrap_samples: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
    confidence_level: float,
) -> pd.DataFrame:
    rows = []
    grouped_points = point_estimates.set_index(group_columns)
    grouped_bootstrap = bootstrap_samples.groupby(group_columns, dropna=False)
    for group_values, group in grouped_bootstrap:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        for metric in metric_columns:
            lower, upper = percentile_interval(group[metric], confidence_level)
            point_value = grouped_points.loc[group_values, metric]
            row = dict(zip(group_columns, group_values, strict=True))
            row.update(
                {
                    "metric": metric,
                    "point_estimate": float(point_value),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": confidence_level,
                    "bootstrap_replicates": int(group[metric].notna().sum()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def point_probability_table(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, df in predictions.items():
        metrics = probability_metrics(df[OBSERVED_COLUMN], df[PREDICTED_COLUMN])
        metrics.update({"model": model, "n": len(df)})
        rows.append(metrics)
    return pd.DataFrame(rows)


def point_threshold_table(
    predictions: dict[str, pd.DataFrame],
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    for model, df in predictions.items():
        y_true = df[OBSERVED_COLUMN].to_numpy()
        y_prob = df[PREDICTED_COLUMN].to_numpy()
        for threshold in thresholds:
            metrics = threshold_summary(y_true, y_prob, threshold)
            metrics.update({"model": model, "n": len(df)})
            rows.append(metrics)
    return pd.DataFrame(rows)


def bootstrap_model_metrics(
    predictions: dict[str, pd.DataFrame],
    *,
    thresholds: list[float],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probability_rows = []
    threshold_rows = []
    for model, df in predictions.items():
        y_true = df[OBSERVED_COLUMN].to_numpy()
        y_prob = df[PREDICTED_COLUMN].to_numpy()
        n = len(df)
        for replicate in range(1, n_bootstrap + 1):
            sample_index = rng.integers(0, n, size=n)
            sample_y = y_true[sample_index]
            sample_prob = y_prob[sample_index]

            probability_row = safe_probability_metrics(sample_y, sample_prob)
            probability_row.update({"model": model, "bootstrap_replicate": replicate, "n": n})
            probability_rows.append(probability_row)

            for threshold in thresholds:
                threshold_row = threshold_summary(sample_y, sample_prob, threshold)
                threshold_row.update(
                    {
                        "model": model,
                        "threshold": threshold,
                        "bootstrap_replicate": replicate,
                        "n": n,
                    }
                )
                threshold_rows.append(threshold_row)

    return pd.DataFrame(probability_rows), pd.DataFrame(threshold_rows)


def paired_apache_comparison(
    predictions: dict[str, pd.DataFrame],
    apache_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows = []
    bootstrap_rows = []
    apache_for_merge = apache_df.rename(
        columns={
            PREDICTED_COLUMN: "apache_predicted_mortality_probability",
            OBSERVED_COLUMN: "apache_observed_hospital_death",
        }
    )
    for model, df in predictions.items():
        paired = df.merge(apache_for_merge, on=ROW_INDEX_COLUMN, how="inner")
        if paired.empty:
            raise ValueError(f"No rows overlap between {model} and APACHE predictions.")
        if not paired[OBSERVED_COLUMN].equals(paired["apache_observed_hospital_death"]):
            raise ValueError(f"Outcome mismatch after APACHE merge for {model}.")

        y_true = paired[OBSERVED_COLUMN].to_numpy()
        model_prob = paired[PREDICTED_COLUMN].to_numpy()
        apache_prob = paired["apache_predicted_mortality_probability"].to_numpy()
        n = len(paired)

        model_metrics = safe_probability_metrics(y_true, model_prob)
        apache_metrics = safe_probability_metrics(y_true, apache_prob)
        point_row = {"model": model, "reference_model": APACHE_BENCHMARK_COLUMN, "n": n}
        for metric in PROBABILITY_METRICS:
            point_row[f"{metric}_difference"] = model_metrics[metric] - apache_metrics[metric]
        point_rows.append(point_row)

        for replicate in range(1, n_bootstrap + 1):
            sample_index = rng.integers(0, n, size=n)
            sample_y = y_true[sample_index]
            sample_model_prob = model_prob[sample_index]
            sample_apache_prob = apache_prob[sample_index]
            sample_model_metrics = safe_probability_metrics(sample_y, sample_model_prob)
            sample_apache_metrics = safe_probability_metrics(sample_y, sample_apache_prob)
            bootstrap_row = {
                "model": model,
                "reference_model": APACHE_BENCHMARK_COLUMN,
                "bootstrap_replicate": replicate,
                "n": n,
            }
            for metric in PROBABILITY_METRICS:
                bootstrap_row[f"{metric}_difference"] = (
                    sample_model_metrics[metric] - sample_apache_metrics[metric]
                )
            bootstrap_rows.append(bootstrap_row)

    return pd.DataFrame(point_rows), pd.DataFrame(bootstrap_rows)


def summarize_apache_comparison(
    point_estimates: pd.DataFrame,
    bootstrap_samples: pd.DataFrame,
    confidence_level: float,
) -> pd.DataFrame:
    difference_metrics = [f"{metric}_difference" for metric in PROBABILITY_METRICS]
    return summarize_bootstrap_samples(
        point_estimates=point_estimates,
        bootstrap_samples=bootstrap_samples,
        group_columns=["model", "reference_model"],
        metric_columns=difference_metrics,
        confidence_level=confidence_level,
    )


def save_performance_figure(summary: pd.DataFrame, figure_dir: Path) -> None:
    metrics = [
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("brier_score", "Brier Score"),
    ]
    models = list(summary["model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        data = summary[summary["metric"] == metric].set_index("model").loc[models].reset_index()
        x = np.arange(len(data))
        y = data["point_estimate"].to_numpy()
        lower_error = y - data["ci_lower"].to_numpy()
        upper_error = data["ci_upper"].to_numpy() - y
        ax.errorbar(x, y, yerr=[lower_error, upper_error], fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(data["model"], rotation=35, ha="right")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "performance_metric_confidence_intervals.png", dpi=150)
    plt.close(fig)


def save_threshold_figure(
    summary: pd.DataFrame,
    *,
    model: str,
    threshold: float,
    figure_dir: Path,
) -> None:
    data = summary[
        (summary["model"] == model)
        & np.isclose(summary["threshold"].astype(float), threshold)
        & summary["metric"].isin(THRESHOLD_METRICS)
    ].copy()
    if data.empty:
        return
    data["metric_label"] = data["metric"].replace(
        {
            "sensitivity_recall": "Sensitivity",
            "specificity": "Specificity",
            "precision": "Precision",
            "f1": "F1",
            "flagged_high_risk_percent": "Flagged %",
        }
    )
    x = np.arange(len(data))
    y = data["point_estimate"].to_numpy()
    lower_error = y - data["ci_lower"].to_numpy()
    upper_error = data["ci_upper"].to_numpy() - y
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(x, y, yerr=[lower_error, upper_error], fmt="o", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(data["metric_label"], rotation=25, ha="right")
    ax.set_title(f"{model} Threshold {threshold:.2f} Metrics")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "recommended_threshold_confidence_intervals.png", dpi=150)
    plt.close(fig)


def save_apache_comparison_figure(summary: pd.DataFrame, figure_dir: Path) -> None:
    data = summary[summary["metric"] == "roc_auc_difference"].copy()
    if data.empty:
        return
    x = np.arange(len(data))
    y = data["point_estimate"].to_numpy()
    lower_error = y - data["ci_lower"].to_numpy()
    upper_error = data["ci_upper"].to_numpy() - y
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.errorbar(x, y, yerr=[lower_error, upper_error], fmt="o", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(data["model"], rotation=30, ha="right")
    ax.set_ylabel("ROC-AUC difference vs APACHE")
    ax.set_title("Bootstrap Confidence Intervals For ROC-AUC Improvement")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "apache_roc_auc_difference_confidence_intervals.png", dpi=150)
    plt.close(fig)


def run_bootstrap_uncertainty(
    *,
    report_dir: Path = DEFAULT_REPORT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    models: list[str] | None = None,
    thresholds: list[float] | None = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    include_apache: bool = True,
) -> None:
    models = models or DEFAULT_MODELS
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    table_dir, figure_dir = ensure_dirs(output_dir)
    predictions = load_oof_predictions(report_dir, models)
    if include_apache:
        predictions[APACHE_BENCHMARK_COLUMN] = load_apache_predictions()

    rng = np.random.default_rng(random_state)
    point_probability = point_probability_table(predictions)
    point_thresholds = point_threshold_table(predictions, thresholds)
    bootstrap_probability, bootstrap_thresholds = bootstrap_model_metrics(
        predictions,
        thresholds=thresholds,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )

    probability_summary = summarize_bootstrap_samples(
        point_estimates=point_probability,
        bootstrap_samples=bootstrap_probability,
        group_columns=["model"],
        metric_columns=PROBABILITY_METRICS,
        confidence_level=confidence_level,
    )
    threshold_summary_table = summarize_bootstrap_samples(
        point_estimates=point_thresholds,
        bootstrap_samples=bootstrap_thresholds,
        group_columns=["model", "threshold"],
        metric_columns=THRESHOLD_METRICS,
        confidence_level=confidence_level,
    )

    point_probability.to_csv(table_dir / "performance_point_estimates.csv", index=False)
    point_thresholds.to_csv(table_dir / "threshold_point_estimates.csv", index=False)
    bootstrap_probability.to_csv(table_dir / "performance_bootstrap_samples.csv", index=False)
    bootstrap_thresholds.to_csv(table_dir / "threshold_bootstrap_samples.csv", index=False)
    probability_summary.to_csv(table_dir / "performance_with_bootstrap_ci.csv", index=False)
    threshold_summary_table.to_csv(
        table_dir / "threshold_metrics_with_bootstrap_ci.csv",
        index=False,
    )

    if include_apache:
        apache_df = predictions[APACHE_BENCHMARK_COLUMN]
        model_predictions = {
            model: df for model, df in predictions.items() if model != APACHE_BENCHMARK_COLUMN
        }
        point_differences, bootstrap_differences = paired_apache_comparison(
            model_predictions,
            apache_df,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        apache_summary = summarize_apache_comparison(
            point_differences,
            bootstrap_differences,
            confidence_level,
        )
        point_differences.to_csv(table_dir / "apache_comparison_point_estimates.csv", index=False)
        bootstrap_differences.to_csv(
            table_dir / "apache_comparison_bootstrap_samples.csv",
            index=False,
        )
        apache_summary.to_csv(
            table_dir / "apache_comparison_with_bootstrap_ci.csv",
            index=False,
        )
        save_apache_comparison_figure(apache_summary, figure_dir)

    save_performance_figure(probability_summary, figure_dir)
    save_threshold_figure(
        threshold_summary_table,
        model=RECOMMENDED_MODEL,
        threshold=RECOMMENDED_THRESHOLD,
        figure_dir=figure_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="OOF model name to include. Can be repeated.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=[],
        help="Decision threshold to include. Can be repeated.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-apache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_bootstrap_uncertainty(
        report_dir=args.report_dir,
        output_dir=args.output_dir,
        models=args.model or None,
        thresholds=args.threshold or None,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        random_state=args.random_state,
        include_apache=not args.no_apache,
    )


if __name__ == "__main__":
    main()
