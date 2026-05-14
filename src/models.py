"""Model construction from YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FunctionTransformer, Pipeline

from src.preprocessing import build_preprocessor


def _to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return X


def load_model_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_estimator(config: dict[str, Any]):
    model_config = config["model"]
    model_type = model_config["type"]

    if model_type == "logistic_regression":
        params = {
            "penalty": model_config["penalty"],
            "solver": model_config["solver"],
            "C": float(model_config["C"]),
            "class_weight": model_config.get("class_weight"),
            "max_iter": int(model_config.get("max_iter", 5000)),
            "tol": float(model_config.get("tol", 0.0001)),
            "random_state": int(model_config.get("random_state", 42)),
        }
        if "n_jobs" in model_config:
            params["n_jobs"] = int(model_config["n_jobs"])
        if model_config["penalty"] == "elasticnet":
            params["l1_ratio"] = float(model_config["l1_ratio"])
        return LogisticRegression(**params)

    if model_type == "random_forest_classifier":
        return RandomForestClassifier(
            n_estimators=int(model_config.get("n_estimators", 300)),
            max_depth=model_config.get("max_depth"),
            min_samples_leaf=int(model_config.get("min_samples_leaf", 1)),
            max_features=model_config.get("max_features", "sqrt"),
            class_weight=model_config.get("class_weight"),
            n_jobs=int(model_config.get("n_jobs", -1)),
            random_state=int(model_config.get("random_state", 42)),
        )

    if model_type == "hist_gradient_boosting_classifier":
        return HistGradientBoostingClassifier(
            max_iter=int(model_config.get("max_iter", 150)),
            learning_rate=float(model_config.get("learning_rate", 0.1)),
            max_depth=int(model_config.get("max_depth", 3)),
            min_samples_leaf=int(model_config.get("min_samples_leaf", 1)),
            l2_regularization=float(model_config.get("l2_regularization", 0.0)),
            random_state=int(model_config.get("random_state", 42)),
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def build_pipeline(config: dict[str, Any], X) -> Pipeline:
    steps = [("preprocessor", build_preprocessor(X))]
    if config["model"]["type"] == "hist_gradient_boosting_classifier":
        steps.append(("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)))
    steps.append(("model", build_estimator(config)))
    return Pipeline(steps=steps)
