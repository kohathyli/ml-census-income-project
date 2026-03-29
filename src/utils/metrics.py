from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)),
        "roc_auc": float(roc_auc_score(y_true, y_score, sample_weight=sample_weight)),
        "average_precision": float(
            average_precision_score(y_true, y_score, sample_weight=sample_weight)
        ),
    }


def classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> str:
    return classification_report(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        target_names=["<=50K", ">50K"],
        zero_division=0,
    )


def weighted_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
