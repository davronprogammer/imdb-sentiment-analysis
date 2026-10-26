"""Evaluation metrics computation and formatting."""

from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: 1D array of ground truth binary labels (0 or 1).
        y_pred: 1D array of predicted binary labels (0 or 1).
        y_proba: Optional 1D array of predicted positive-class probabilities.

    Returns:
        Dictionary of formatted metrics.
    """
    accuracy = float(accuracy_score(y_true, y_pred))

    pos_precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    pos_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    pos_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    neg_precision = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    neg_recall = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    neg_f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Negative", "Positive"],
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "f1_score": pos_f1,
        "positive_precision": pos_precision,
        "positive_recall": pos_recall,
        "positive_f1": pos_f1,
        "negative_precision": neg_precision,
        "negative_recall": neg_recall,
        "negative_f1": neg_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "total_samples": int(len(y_true)),
    }
