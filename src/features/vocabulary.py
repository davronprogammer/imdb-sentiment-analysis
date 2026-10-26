"""Vocabulary management with special tokens for padding and out-of-vocabulary words."""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import json


class Vocabulary:
    """Manages token-to-index and index-to-token mappings.

    Ensures <PAD> and <UNK> are assigned deterministic indices (0 and 1 by default).
    """

    def __init__(
        self,
        token_to_idx: Optional[Dict[str, int]] = None,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        pad_idx: int = 0,
        unk_idx: int = 1,
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        if token_to_idx is not None:
            self.token_to_idx = dict(token_to_idx)
            self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        else:
            self.token_to_idx = {
                self.pad_token: self.pad_idx,
                self.unk_token: self.unk_idx,
            }
            self.idx_to_token = {
                self.pad_idx: self.pad_token,
                self.unk_idx: self.unk_token,
            }

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_idx

    def __getitem__(self, token: str) -> int:
        return self.lookup_token(token)

    def lookup_token(self, token: str) -> int:
        """Map token to index, returning unk_idx if token is not present."""
        return self.token_to_idx.get(token, self.unk_idx)

    def lookup_index(self, idx: int) -> str:
        """Map index back to token string."""
        return self.idx_to_token.get(idx, self.unk_token)

    @classmethod
    def build_from_texts(
        cls,
        texts: Iterable[str],
        max_size: int = 30000,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        pad_idx: int = 0,
        unk_idx: int = 1,
    ) -> "Vocabulary":
        """Build vocabulary from training corpus texts only (avoids data leakage).

        Args:
            texts: Iterable of cleaned training text strings.
            max_size: Maximum vocabulary size including special tokens.
            pad_token: Padding token string.
            unk_token: Unknown token string.
            pad_idx: Index reserved for padding token.
            unk_idx: Index reserved for unknown token.

        Returns:
            Configured Vocabulary instance.
        """
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Reserve slots for PAD and UNK
        num_word_slots = max(0, max_size - 2)
        most_common = counter.most_common(num_word_slots)

        token_to_idx = {
            pad_token: pad_idx,
            unk_token: unk_idx,
        }
        for word, _ in most_common:
            token_to_idx[word] = len(token_to_idx)

        return cls(
            token_to_idx=token_to_idx,
            pad_token=pad_token,
            unk_token=unk_token,
            pad_idx=pad_idx,
            unk_idx=unk_idx,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize vocabulary configuration and mapping."""
        return {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "pad_idx": self.pad_idx,
            "unk_idx": self.unk_idx,
            "vocab_size": len(self.token_to_idx),
            "token_to_idx": self.token_to_idx,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vocabulary":
        """Reconstruct Vocabulary from dictionary."""
        return cls(
            token_to_idx=data["token_to_idx"],
            pad_token=data.get("pad_token", "<PAD>"),
            unk_token=data.get("unk_token", "<UNK>"),
            pad_idx=data.get("pad_idx", 0),
            unk_idx=data.get("unk_idx", 1),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save vocabulary to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        """Load vocabulary from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
