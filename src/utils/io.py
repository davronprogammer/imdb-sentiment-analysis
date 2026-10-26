"""I/O helpers for saving and loading JSON and PyTorch checkpoints."""

import json
from pathlib import Path
from typing import Any, Dict, Union
import torch


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save serializable data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    """Load data from JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    model: torch.nn.Module,
    path: Union[str, Path],
    metadata: Dict[str, Any] = None,
) -> None:
    """Save model weights and metadata checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module = None,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load model checkpoint."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    return checkpoint
