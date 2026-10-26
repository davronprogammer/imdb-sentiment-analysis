"""PyTorch Dataset and DataLoader construction for IMDB reviews."""

from typing import List, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.features.encoding import encode_text
from src.features.vocabulary import Vocabulary


class IMDBDataset(Dataset):
    """PyTorch Dataset for IMDB sentiment reviews."""

    def __init__(
        self,
        reviews: Union[List[str], pd.Series],
        labels: Optional[Union[List[int], pd.Series]] = None,
        vocab: Optional[Vocabulary] = None,
        max_length: int = 200,
    ):
        """Args:
            reviews: List or Series of cleaned review text strings.
            labels: Optional list or Series of integer labels (0 or 1).
            vocab: Vocabulary instance for mapping tokens to IDs.
            max_length: Maximum sequence length (default: 200).
        """
        if isinstance(reviews, pd.Series):
            self.reviews = reviews.tolist()
        elif isinstance(reviews, list):
            self.reviews = reviews
        else:
            self.reviews = list(reviews)

        if labels is not None:
            if isinstance(labels, pd.Series):
                self.labels = labels.tolist()
            elif isinstance(labels, list):
                self.labels = labels
            else:
                self.labels = list(labels)
        else:
            self.labels = None

        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        review = self.reviews[index]
        if self.vocab is None:
            raise ValueError("Vocabulary must be provided to encode reviews.")

        encoded = encode_text(
            text=review,
            vocab=self.vocab,
            max_length=self.max_length,
            pad_idx=self.vocab.pad_idx,
            unk_idx=self.vocab.unk_idx,
        )

        review_tensor = torch.tensor(encoded, dtype=torch.long)

        if self.labels is not None:
            label = self.labels[index]
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return review_tensor, label_tensor

        return review_tensor


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vocab: Vocabulary,
    batch_size: int = 64,
    max_length: int = 200,
    shuffle_train: bool = True,
    shuffle_eval: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and test splits.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        vocab: Vocabulary instance.
        batch_size: Batch size (default: 64).
        max_length: Sequence max length (default: 200).
        shuffle_train: Whether to shuffle train set (default: True).
        shuffle_eval: Whether to shuffle val/test sets (default: False).
        num_workers: DataLoader worker count (default: 0).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = IMDBDataset(
        reviews=train_df["review"],
        labels=train_df["label"],
        vocab=vocab,
        max_length=max_length,
    )
    val_dataset = IMDBDataset(
        reviews=val_df["review"],
        labels=val_df["label"],
        vocab=vocab,
        max_length=max_length,
    )
    test_dataset = IMDBDataset(
        reviews=test_df["review"],
        labels=test_df["label"],
        vocab=vocab,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_eval,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_eval,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
