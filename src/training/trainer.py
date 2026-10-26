"""Trainer class and training loop for PyTorch recurrent models."""

import copy
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.io import save_checkpoint


class Trainer:
    """Handles training, validation, early stopping, and checkpoint saving."""

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
        learning_rate: float = 0.001,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        early_stopping_patience: int = 0,
    ):
        """Args:
            model: PyTorch neural network model.
            device: Compute device ('cuda' or 'cpu').
            learning_rate: Learning rate for Adam optimizer (default: 0.001).
            criterion: Loss function (default: nn.BCEWithLogitsLoss).
            optimizer: Optimizer (default: Adam).
            checkpoint_dir: Directory where best checkpoint will be saved.
            early_stopping_patience: Epochs to wait without val loss improvement before stopping (0 disables).
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.early_stopping_patience = early_stopping_patience

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch over DataLoader.

        Returns:
            (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for reviews, labels in dataloader:
            reviews = reviews.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()

            logits = self.model(reviews)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            # In BCEWithLogitsLoss, logit >= 0 is mathematically identical to sigmoid(logit) >= 0.5
            preds = (logits >= 0.0).float()
            running_correct += (preds == labels).sum().item()
            total_samples += batch_size

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        epoch_acc = running_correct / total_samples if total_samples > 0 else 0.0
        return epoch_loss, epoch_acc

    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Run validation over DataLoader.

        Returns:
            (val_loss, val_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        with torch.no_grad():
            for reviews, labels in dataloader:
                reviews = reviews.to(self.device)
                labels = labels.to(self.device).float()

                logits = self.model(reviews)
                loss = self.criterion(logits, labels)

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size

                preds = (logits >= 0.0).float()
                running_correct += (preds == labels).sum().item()
                total_samples += batch_size

        val_loss = running_loss / total_samples if total_samples > 0 else 0.0
        val_acc = running_correct / total_samples if total_samples > 0 else 0.0
        return val_loss, val_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        model_name: str = "model",
        extra_metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Fit model across epochs with validation tracking and best checkpoint preservation.

        Returns:
            Dictionary containing:
                - history: dict with train_loss, val_loss, train_acc, val_acc lists
                - best_epoch: integer index of best epoch
                - best_val_loss: best validation loss
                - best_val_acc: validation accuracy at best epoch
                - training_time: elapsed seconds
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0
        best_weights = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                if self.checkpoint_dir:
                    ckpt_path = self.checkpoint_dir / f"{model_name}_best.pt"
                    metadata = {
                        "model_name": model_name,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "best_val_acc": best_val_acc,
                        **(extra_metadata or {}),
                    }
                    save_checkpoint(self.model, ckpt_path, metadata)
            else:
                patience_counter += 1

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
                    f"{' [BEST]' if is_best else ''}"
                )

            if (
                self.early_stopping_patience > 0
                and patience_counter >= self.early_stopping_patience
            ):
                if verbose:
                    print(
                        f"Early stopping triggered after {epoch} epochs "
                        f"(no validation improvement for {self.early_stopping_patience} epochs)."
                    )
                break

        training_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds. Restoring best weights from epoch {best_epoch}.")

        # Restore best model weights
        self.model.load_state_dict(best_weights)

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "training_time": training_time,
        }
