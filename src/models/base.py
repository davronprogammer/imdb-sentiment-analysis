"""Model base utilities and parameter counting."""

import torch
import torch.nn as nn
from typing import Dict, Any


def count_parameters(model: nn.Module) -> int:
    """Calculate the total number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Integer count of trainable parameters (requires_grad=True).
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Return dictionary summary of model architecture and parameter count."""
    total_params = count_parameters(model)
    all_params = sum(p.numel() for p in model.parameters())
    return {
        "model_class": model.__class__.__name__,
        "trainable_parameters": total_params,
        "total_parameters": all_params,
    }
