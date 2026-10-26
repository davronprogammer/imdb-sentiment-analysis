"""Evaluation metrics, prediction, and plotting utilities."""

from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.evaluator import predict_dataloader, evaluate_model
from src.evaluation.plots import (
    plot_training_history,
    plot_confusion_matrix_single,
    plot_side_by_side_confusion_matrices,
)

__all__ = [
    "compute_classification_metrics",
    "predict_dataloader",
    "evaluate_model",
    "plot_training_history",
    "plot_confusion_matrix_single",
    "plot_side_by_side_confusion_matrices",
]
