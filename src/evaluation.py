"""Evaluation utilities for mortality probability models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_THRESHOLDS = [0.10, 0.20, 0.30, 0.50]


def probability_metrics(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def threshold_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize_thresholds(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    return pd.DataFrame(
        [threshold_metrics(y_true, y_prob, threshold=threshold) for threshold in thresholds]
    )

