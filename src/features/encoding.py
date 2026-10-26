"""Sequence encoding, truncation, and padding utilities."""

from typing import List
from src.features.vocabulary import Vocabulary


def encode_tokens(
    tokens: List[str],
    vocab: Vocabulary,
    max_length: int = 200,
    pad_idx: int = 0,
    unk_idx: int = 1,
) -> List[int]:
    """Map token list to integer indices with truncation and padding.

    1. Map tokens to IDs (or unk_idx if missing).
    2. Truncate to max_length if longer.
    3. Post-pad with pad_idx if shorter.

    Args:
        tokens: List of string tokens.
        vocab: Vocabulary object.
        max_length: Target sequence length.
        pad_idx: Padding index.
        unk_idx: Unknown word index.

    Returns:
        List of integer IDs of exact length max_length.
    """
    ids = [vocab.lookup_token(token) for token in tokens]

    # Truncate
    ids = ids[:max_length]

    # Post-pad
    if len(ids) < max_length:
        ids.extend([pad_idx] * (max_length - len(ids)))

    return ids


def encode_text(
    text: str,
    vocab: Vocabulary,
    max_length: int = 200,
    pad_idx: int = 0,
    unk_idx: int = 1,
) -> List[int]:
    """Tokenize and encode raw or cleaned text string."""
    tokens = text.split()
    return encode_tokens(
        tokens=tokens,
        vocab=vocab,
        max_length=max_length,
        pad_idx=pad_idx,
        unk_idx=unk_idx,
    )
