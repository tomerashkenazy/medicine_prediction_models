"""Train and evaluate stage 2 mortality prediction models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.cohort import apply_stage1_cohort
from src.data_loading import load_solution_template, load_training_data, load_unlabeled_data
from src.evaluation import DEFAULT_THRESHOLDS, probability_metrics, summarize_thresholds
from src.features import APACHE_BENCHMARK_COLUMN, ID_COLUMN, TARGET_COLUMN, get_raw_feature_columns
from src.models import build_pipeline, load_model_config
from src.reporting import (
    ensure_output_dirs,
    save_csv,
    save_precision_recall_curve,
    save_roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _predict_mortality_probability(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _fit_params(fit_config: dict | None, y: pd.Series) -> dict:
    if not fit_config:
        return {}
    if fit_config.get("sample_weight") == "balanced":
        return {"model__sample_weight": compute_sample_weight("balanced", y)}
    raise ValueError(f"Unsupported fit configuration: {fit_config}")


def _model_feature_accounting(
    *,
    model_name: str,
    final_pipeline,
    raw_feature_count: int,
    excluded_feature_columns: list[str],
) -> dict:
    preprocessor = final_pipeline.named_steps["preprocessor"]
    estimator = final_pipeline.named_steps["model"]
    transformed_feature_count = len(preprocessor.get_feature_names_out())
    coefficients = getattr(estimator, "coef_", None)
    nonzero_coefficients = (
        int(np.count_nonzero(np.abs(coefficients.ravel()) > 1e-12))
        if coefficients is not None
        else np.nan
    )
    return {
        "model": model_name,
        "raw_feature_count": raw_feature_count,
        "excluded_feature_count": len(excluded_feature_columns),
        "excluded_features": "; ".join(excluded_feature_columns),
        "transformed_feature_count": transformed_feature_count,
        "nonzero_coefficient_count": nonzero_coefficients,
    }


def _grid_search(
    model,
    search_grid: dict,
    *,
    inner_splits: int,
    random_state: int,
    search_n_jobs: int,
) -> GridSearchCV:
    return GridSearchCV(
        estimator=model,
        param_grid=search_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=random_state,
        ),
        n_jobs=search_n_jobs,
        refit=True,
        return_train_score=False,
    )


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    random_state: int,
    fit_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rows = []
    threshold_rows = []
    oof_prob = np.zeros(len(y), dtype=float)

    for fold_index, (train_index, valid_index) in enumerate(splitter.split(X, y), start=1):
        fold_model = clone(model)
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        fold_model.fit(X_train, y_train, **_fit_params(fit_config, y_train))
        y_prob = _predict_mortality_probability(fold_model, X_valid)
        oof_prob[valid_index] = y_prob

        fold_summary = probability_metrics(y_valid, y_prob)
        fold_summary["fold"] = fold_index
        fold_rows.append(fold_summary)

        fold_thresholds = summarize_thresholds(y_valid, y_prob, DEFAULT_THRESHOLDS)
        fold_thresholds.insert(0, "fold", fold_index)
        threshold_rows.append(fold_thresholds)

    oof_predictions = pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            "observed_hospital_death": y.to_numpy(),
            "predicted_mortality_probability": oof_prob,
        }
    )
    return pd.DataFrame(fold_rows), pd.concat(threshold_rows, ignore_index=True), oof_predictions


def nested_grid_search_model(
    model,
    search_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    inner_splits: int,
    random_state: int,
    search_n_jobs: int,
    fit_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    inner_splitter = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_rows = []
    threshold_rows = []
    best_param_rows = []
    oof_prob = np.zeros(len(y), dtype=float)

    for fold_index, (train_index, valid_index) in enumerate(outer_splitter.split(X, y), start=1):
        print(f"  Outer fold {fold_index}/{n_splits}...", flush=True)
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        search = GridSearchCV(
            estimator=clone(model),
            param_grid=search_grid,
            scoring="roc_auc",
            cv=inner_splitter,
            n_jobs=search_n_jobs,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_train, y_train, **_fit_params(fit_config, y_train))
        y_prob = _predict_mortality_probability(search.best_estimator_, X_valid)
        oof_prob[valid_index] = y_prob

        fold_summary = probability_metrics(y_valid, y_prob)
        fold_summary["fold"] = fold_index
        fold_summary["inner_best_roc_auc"] = search.best_score_
        fold_rows.append(fold_summary)

        fold_thresholds = summarize_thresholds(y_valid, y_prob, DEFAULT_THRESHOLDS)
        fold_thresholds.insert(0, "fold", fold_index)
        threshold_rows.append(fold_thresholds)

        best_params = {"fold": fold_index, "inner_best_roc_auc": search.best_score_}
        best_params.update(search.best_params_)
        best_param_rows.append(best_params)

    oof_predictions = pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            "observed_hospital_death": y.to_numpy(),
            "predicted_mortality_probability": oof_prob,
        }
    )
    return (
        pd.DataFrame(fold_rows),
        pd.concat(threshold_rows, ignore_index=True),
        oof_predictions,
        pd.DataFrame(best_param_rows),
    )


def apache_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    benchmark = df[[TARGET_COLUMN, APACHE_BENCHMARK_COLUMN]].dropna()
    benchmark = benchmark[
        (benchmark[APACHE_BENCHMARK_COLUMN] >= 0)
        & (benchmark[APACHE_BENCHMARK_COLUMN] <= 1)
    ]
    metrics = probability_metrics(
        benchmark[TARGET_COLUMN],
        benchmark[APACHE_BENCHMARK_COLUMN].to_numpy(),
    )
    metrics["n"] = len(benchmark)
    return pd.DataFrame([metrics])


def train_from_configs(
    *,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
    search_n_jobs: int = -1,
    excluded_feature_columns: list[str] | None = None,
    include_models: list[str] | None = None,
) -> None:
    output_dirs = ensure_output_dirs(output_dir)
    train_df = apply_stage1_cohort(load_training_data())
    unlabeled_df = load_unlabeled_data()
    excluded_feature_columns = excluded_feature_columns or []

    feature_columns = get_raw_feature_columns(
        train_df.columns,
        additional_excluded_columns=excluded_feature_columns,
    )
    _validate_columns(train_df, feature_columns + [TARGET_COLUMN, ID_COLUMN])
    _validate_columns(unlabeled_df, feature_columns + [ID_COLUMN])

    X = train_df[feature_columns]
    y = train_df[TARGET_COLUMN].astype(int)

    save_csv(apache_benchmark(train_df), output_dirs["reports"] / "apache_benchmark_metrics.csv")

    all_fold_metrics = []
    all_threshold_metrics = []
    all_best_params_by_fold = []
    final_best_params = []
    feature_accounting_rows = []

    for config_path in sorted(config_dir.glob("*.yaml")):
        config = load_model_config(config_path)
        model_name = config["name"]
        if include_models and model_name not in include_models:
            print(f"Skipping {model_name} because it is not in --include-model.", flush=True)
            continue
        print(f"Training {model_name}...", flush=True)
        pipeline = build_pipeline(config, X)

        if config.get("tune", False):
            if "search_grid" not in config:
                raise ValueError(f"{model_name} has tune=true but no search_grid.")
            fold_metrics, threshold_metrics_df, oof_predictions, best_params_by_fold = (
                nested_grid_search_model(
                    pipeline,
                    config["search_grid"],
                    X,
                    y,
                    n_splits=n_splits,
                    inner_splits=inner_splits,
                    random_state=random_state,
                    search_n_jobs=search_n_jobs,
                    fit_config=config.get("fit"),
                )
            )
            best_params_by_fold.insert(0, "model", model_name)
            all_best_params_by_fold.append(best_params_by_fold)

            final_search = _grid_search(
                build_pipeline(config, X),
                config["search_grid"],
                inner_splits=inner_splits,
                random_state=random_state,
                search_n_jobs=search_n_jobs,
            )
            final_search.fit(X, y, **_fit_params(config.get("fit"), y))
            final_pipeline = final_search.best_estimator_
            final_params = {"model": model_name, "best_roc_auc": final_search.best_score_}
            final_params.update(final_search.best_params_)
            final_best_params.append(final_params)
        else:
            fold_metrics, threshold_metrics_df, oof_predictions = cross_validate_model(
                pipeline,
                X,
                y,
                n_splits=n_splits,
                random_state=random_state,
                fit_config=config.get("fit"),
            )
            final_pipeline = build_pipeline(config, X)
            final_pipeline.fit(X, y, **_fit_params(config.get("fit"), y))

        fold_metrics.insert(0, "model", model_name)
        threshold_metrics_df.insert(0, "model", model_name)
        oof_predictions.insert(0, "model", model_name)
        all_fold_metrics.append(fold_metrics)
        all_threshold_metrics.append(threshold_metrics_df)

        save_csv(oof_predictions, output_dirs["reports"] / f"{model_name}_oof_predictions.csv")
        save_roc_curve(
            oof_predictions["observed_hospital_death"],
            oof_predictions["predicted_mortality_probability"],
            output_dirs["figures"] / f"{model_name}_roc_curve.png",
            f"{model_name} ROC curve",
        )
        save_precision_recall_curve(
            oof_predictions["observed_hospital_death"],
            oof_predictions["predicted_mortality_probability"],
            output_dirs["figures"] / f"{model_name}_precision_recall_curve.png",
            f"{model_name} precision-recall curve",
        )

        joblib.dump(final_pipeline, output_dirs["models"] / f"{model_name}.joblib")
        feature_accounting_rows.append(
            _model_feature_accounting(
                model_name=model_name,
                final_pipeline=final_pipeline,
                raw_feature_count=len(feature_columns),
                excluded_feature_columns=excluded_feature_columns,
            )
        )

        unlabeled_predictions = load_solution_template()
        probabilities = _predict_mortality_probability(final_pipeline, unlabeled_df[feature_columns])
        prediction_by_id = pd.DataFrame(
            {
                ID_COLUMN: unlabeled_df[ID_COLUMN].to_numpy(),
                TARGET_COLUMN: probabilities,
            }
        )
        unlabeled_predictions = unlabeled_predictions[[ID_COLUMN]].merge(
            prediction_by_id,
            on=ID_COLUMN,
            how="left",
        )
        save_csv(
            unlabeled_predictions,
            output_dirs["predictions"] / f"{model_name}_unlabeled_predictions.csv",
        )
        print(f"Finished {model_name}.", flush=True)

    if not all_fold_metrics:
        raise ValueError("No configs were trained.")

    fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
    threshold_metrics_df = pd.concat(all_threshold_metrics, ignore_index=True)
    summary = (
        fold_metrics_df.groupby("model", as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            brier_score_mean=("brier_score", "mean"),
            brier_score_std=("brier_score", "std"),
        )
        .sort_values("roc_auc_mean", ascending=False)
    )

    save_csv(fold_metrics_df, output_dirs["reports"] / "cv_fold_metrics.csv")
    save_csv(summary, output_dirs["reports"] / "cv_summary_metrics.csv")
    save_csv(threshold_metrics_df, output_dirs["reports"] / "cv_threshold_metrics.csv")
    save_csv(
        pd.DataFrame(feature_accounting_rows),
        output_dirs["reports"] / "feature_accounting.csv",
    )
    if all_best_params_by_fold:
        save_csv(
            pd.concat(all_best_params_by_fold, ignore_index=True),
            output_dirs["reports"] / "best_params_by_fold.csv",
        )
    if final_best_params:
        save_csv(
            pd.DataFrame(final_best_params),
            output_dirs["reports"] / "final_best_params.csv",
        )


def tune_from_configs(
    *,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
    search_n_jobs: int = -1,
    excluded_feature_columns: list[str] | None = None,
    include_models: list[str] | None = None,
) -> None:
    output_dirs = ensure_output_dirs(output_dir)
    train_df = apply_stage1_cohort(load_training_data())
    unlabeled_df = load_unlabeled_data()
    excluded_feature_columns = excluded_feature_columns or []

    feature_columns = get_raw_feature_columns(
        train_df.columns,
        additional_excluded_columns=excluded_feature_columns,
    )
    _validate_columns(train_df, feature_columns + [TARGET_COLUMN, ID_COLUMN])
    _validate_columns(unlabeled_df, feature_columns + [ID_COLUMN])

    X = train_df[feature_columns]
    y = train_df[TARGET_COLUMN].astype(int)

    all_fold_metrics = []
    all_threshold_metrics = []
    all_best_params_by_fold = []
    final_best_params = []

    for config_path in sorted(config_dir.glob("*.yaml")):
        config = load_model_config(config_path)
        if include_models and config["name"] not in include_models:
            print(f"Skipping {config['name']} because it is not in --include-model.", flush=True)
            continue
        if "search_grid" not in config:
            print(f"Skipping {config['name']} because it has no search_grid.", flush=True)
            continue

        model_name = config["name"]
        tuned_name = f"tuned_{model_name}"
        print(f"Tuning {model_name}...", flush=True)
        pipeline = build_pipeline(config, X)

        fold_metrics, threshold_metrics_df, oof_predictions, best_params_by_fold = (
            nested_grid_search_model(
                pipeline,
                config["search_grid"],
                X,
                y,
                n_splits=n_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                search_n_jobs=search_n_jobs,
                fit_config=config.get("fit"),
            )
        )
        fold_metrics.insert(0, "model", tuned_name)
        threshold_metrics_df.insert(0, "model", tuned_name)
        oof_predictions.insert(0, "model", tuned_name)
        best_params_by_fold.insert(0, "model", tuned_name)
        all_fold_metrics.append(fold_metrics)
        all_threshold_metrics.append(threshold_metrics_df)
        all_best_params_by_fold.append(best_params_by_fold)

        save_csv(oof_predictions, output_dirs["reports"] / f"{tuned_name}_oof_predictions.csv")
        save_roc_curve(
            oof_predictions["observed_hospital_death"],
            oof_predictions["predicted_mortality_probability"],
            output_dirs["figures"] / f"{tuned_name}_roc_curve.png",
            f"{tuned_name} ROC curve",
        )
        save_precision_recall_curve(
            oof_predictions["observed_hospital_death"],
            oof_predictions["predicted_mortality_probability"],
            output_dirs["figures"] / f"{tuned_name}_precision_recall_curve.png",
            f"{tuned_name} precision-recall curve",
        )

        final_search = _grid_search(
            build_pipeline(config, X),
            config["search_grid"],
            inner_splits=inner_splits,
            random_state=random_state,
            search_n_jobs=search_n_jobs,
        )
        final_search.fit(X, y, **_fit_params(config.get("fit"), y))
        joblib.dump(final_search.best_estimator_, output_dirs["models"] / f"{tuned_name}.joblib")

        final_params = {"model": tuned_name, "best_roc_auc": final_search.best_score_}
        final_params.update(final_search.best_params_)
        final_best_params.append(final_params)

        unlabeled_predictions = load_solution_template()
        probabilities = _predict_mortality_probability(
            final_search.best_estimator_,
            unlabeled_df[feature_columns],
        )
        prediction_by_id = pd.DataFrame(
            {
                ID_COLUMN: unlabeled_df[ID_COLUMN].to_numpy(),
                TARGET_COLUMN: probabilities,
            }
        )
        unlabeled_predictions = unlabeled_predictions[[ID_COLUMN]].merge(
            prediction_by_id,
            on=ID_COLUMN,
            how="left",
        )
        save_csv(
            unlabeled_predictions,
            output_dirs["predictions"] / f"{tuned_name}_unlabeled_predictions.csv",
        )
        print(f"Finished tuning {model_name}.", flush=True)

    if not all_fold_metrics:
        raise ValueError("No configs with search_grid were found.")

    fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
    threshold_metrics_df = pd.concat(all_threshold_metrics, ignore_index=True)
    best_params_by_fold_df = pd.concat(all_best_params_by_fold, ignore_index=True)
    summary = (
        fold_metrics_df.groupby("model", as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            brier_score_mean=("brier_score", "mean"),
            brier_score_std=("brier_score", "std"),
            inner_best_roc_auc_mean=("inner_best_roc_auc", "mean"),
            inner_best_roc_auc_std=("inner_best_roc_auc", "std"),
        )
        .sort_values("roc_auc_mean", ascending=False)
    )

    save_csv(fold_metrics_df, output_dirs["reports"] / "tuned_cv_fold_metrics.csv")
    save_csv(summary, output_dirs["reports"] / "tuned_cv_summary_metrics.csv")
    save_csv(threshold_metrics_df, output_dirs["reports"] / "tuned_cv_threshold_metrics.csv")
    save_csv(
        best_params_by_fold_df,
        output_dirs["reports"] / "tuned_best_params_by_fold.csv",
    )
    save_csv(
        pd.DataFrame(final_best_params),
        output_dirs["reports"] / "tuned_final_best_params.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--search-n-jobs", type=int, default=-1)
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help="Raw feature column to exclude from model inputs. Can be repeated.",
    )
    parser.add_argument(
        "--include-model",
        action="append",
        default=[],
        help="Config model name to train. Can be repeated. Defaults to all configs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tune:
        tune_from_configs(
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            n_splits=args.n_splits,
            inner_splits=args.inner_splits,
            random_state=args.random_state,
            search_n_jobs=args.search_n_jobs,
            excluded_feature_columns=args.exclude_feature,
            include_models=args.include_model,
        )
    else:
        train_from_configs(
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            n_splits=args.n_splits,
            inner_splits=args.inner_splits,
            random_state=args.random_state,
            search_n_jobs=args.search_n_jobs,
            excluded_feature_columns=args.exclude_feature,
            include_models=args.include_model,
        )
