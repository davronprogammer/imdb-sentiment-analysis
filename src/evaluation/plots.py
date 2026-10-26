"""Publication-quality plotting functions for training curves and confusion matrices."""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_history(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Union[str, Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot 2x2 comparison grid for Training/Validation Loss and Accuracy.

    Args:
        histories: Dict mapping model name to history dict (keys: train_loss, val_loss, train_acc, val_acc).
        save_path: Optional path to save image file (e.g. .png).
        show: Whether to display interactively.

    Returns:
        Matplotlib Figure object.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)

    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (name, history) in enumerate(histories.items()):
        color = color_cycle[idx % len(color_cycle)]
        epochs_range = range(1, len(history["train_loss"]) + 1)

        axes[0, 0].plot(
            epochs_range, history["train_loss"], marker="o", linewidth=2, label=name, color=color
        )
        axes[0, 1].plot(
            epochs_range, history["val_loss"], marker="o", linewidth=2, label=name, color=color
        )
        axes[1, 0].plot(
            epochs_range, history["train_acc"], marker="o", linewidth=2, label=name, color=color
        )
        axes[1, 1].plot(
            epochs_range, history["val_acc"], marker="o", linewidth=2, label=name, color=color
        )

    axes[0, 0].set_title("Training Loss", fontsize=14, fontweight="bold")
    axes[0, 1].set_title("Validation Loss", fontsize=14, fontweight="bold")
    axes[1, 0].set_title("Training Accuracy", fontsize=14, fontweight="bold")
    axes[1, 1].set_title("Validation Accuracy", fontsize=14, fontweight="bold")

    for ax in axes.flat:
        ax.set_xlabel("Epoch", fontsize=12)
        ax.legend(frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_confusion_matrix_single(
    cm: Union[List[List[int]], np.ndarray],
    model_name: str,
    save_path: Union[str, Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot and optionally save a single confusion matrix heatmap."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        cbar=True,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )

    ax.set_title(f"{model_name} Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_side_by_side_confusion_matrices(
    cms: Dict[str, Union[List[List[int]], np.ndarray]],
    save_path: Union[str, Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot side-by-side confusion matrices for LSTM and GRU."""
    models = list(cms.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), dpi=150)

    if len(models) == 1:
        axes = [axes]

    for ax, name in zip(axes, models):
        cm = np.array(cms[name])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            cbar=True,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(f"{name} Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
