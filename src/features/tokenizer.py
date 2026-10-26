"""Text cleaning and tokenization utilities."""

import re
from html import unescape
from typing import List


def clean_text(text: str) -> str:
    """Clean review text following the educational notebook pipeline:
    1. Lowercase text.
    2. Unescape HTML entities.
    3. Replace <br> tags with spaces.
    4. Remove any remaining HTML tags.
    5. Remove URLs (http://, https://, www.).
    6. Remove numeric digits.
    7. Remove all non-letter characters (keep only [a-z] and whitespace).
    8. Collapse multiple whitespace characters into a single space and strip.

    Args:
        text: Raw review string.

    Returns:
        Cleaned, normalized text string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.lower()
    text = unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Split text into whitespace-delimited word tokens.

    Args:
        text: Input text string.

    Returns:
        List of string tokens.
    """
    return text.split()
