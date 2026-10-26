"""Model evaluation logic on DataLoader test/validation sets."""

import time
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_classification_metrics


def predict_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device] = "cpu",
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate predictions, labels, and probabilities across a DataLoader.

    Args:
        model: PyTorch classification model.
        dataloader: PyTorch DataLoader.
        device: 'cuda' or 'cpu'.
        threshold: Decision threshold on raw logits. Default: 0.0 (equivalent to sigmoid >= 0.5).

    Returns:
        (y_true, y_pred, y_proba, elapsed_seconds)
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    all_preds = []
    all_labels = []
    all_probas = []

    start_time = time.time()

    with torch.no_grad():
        for reviews, labels in dataloader:
            reviews = reviews.to(device)
            labels = labels.to(device)

            logits = model(reviews)
            # Probability via sigmoid
            probas = torch.sigmoid(logits)

            # Prediction based on threshold on logits
            preds = (logits >= threshold).long()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())
            all_probas.extend(probas.cpu().numpy().tolist())

    elapsed = time.time() - start_time

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas),
        elapsed,
    )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device] = "cpu",
    model_name: str = "Model",
    threshold: float = 0.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a trained model on a dataset and return metrics.

    Args:
        model: PyTorch model.
        dataloader: Test or validation DataLoader.
        device: Compute device.
        model_name: Label string (e.g. 'LSTM' or 'GRU').
        threshold: Logit classification threshold (default: 0.0).
        verbose: Whether to print summary and classification report.

    Returns:
        Dictionary of evaluation results.
    """
    y_true, y_pred, y_proba, eval_time = predict_dataloader(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=threshold,
    )

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    metrics["model_name"] = model_name
    metrics["evaluation_time"] = eval_time
    metrics["logit_threshold"] = threshold

    if verbose:
        print(f"\n{'=' * 44}")
        print(f" {model_name} — Evaluation Results ")
        print(f"{'=' * 44}")
        print(f"Accuracy:         {metrics['accuracy']:.4f}")
        print(f"F1-score:         {metrics['f1_score']:.4f}")
        print(f"Macro F1-score:   {metrics['macro_f1']:.4f}")
        print(f"Negative recall:  {metrics['negative_recall']:.4f}")
        print(f"Positive recall:  {metrics['positive_recall']:.4f}")
        print(f"Evaluation time:  {eval_time:.2f} seconds")
        print("\nClassification Report:")
        print(metrics["classification_report"])

    return metrics
