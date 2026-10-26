"""Configuration dataclass and default paths for the IMDB sentiment analysis project."""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    """Project configuration parameters."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: Path = field(init=False)
    raw_data_path: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    # Data processing
    max_vocab_size: int = 30000
    max_length: int = 200
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    pad_idx: int = 0
    unk_idx: int = 1

    # Splitting
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_state: int = 42

    # Model architecture
    embedding_dim: int = 128
    hidden_dim: int = 128
    output_dim: int = 1

    # Training
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.001
    early_stopping_patience: int = 2

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.raw_data_path = self.data_dir / "IMDB Dataset.csv"
        self.artifacts_dir = self.project_root / "artifacts"
        self.results_dir = self.project_root / "results"

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
