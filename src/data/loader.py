"""Dataset loading, preprocessing, cleaning, and train/val/test splitting."""

from pathlib import Path
from typing import Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.tokenizer import clean_text


def load_imdb_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load raw IMDB CSV dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"IMDB dataset not found at: {path}")
    return pd.read_csv(path)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess IMDB DataFrame:
    1. Create a copy.
    2. Clean text in 'review' column using clean_text.
    3. Drop duplicate rows based on ['review', 'sentiment'].
    4. Reset DataFrame index.
    5. Map sentiment string to numeric label (negative: 0, positive: 1).

    Returns:
        Cleaned, deduplicated DataFrame with 'review', 'sentiment', and 'label' columns.
    """
    df_clean = df.copy()

    # Clean review text
    df_clean["review"] = df_clean["review"].apply(clean_text)

    # Drop duplicates on cleaned text and sentiment
    df_clean = df_clean.drop_duplicates(subset=["review", "sentiment"]).reset_index(drop=True)

    # Encode sentiment labels
    label_map = {"negative": 0, "positive": 1}
    df_clean["label"] = df_clean["sentiment"].map(label_map)

    # Check for unmapped labels
    if df_clean["label"].isnull().any():
        raise ValueError("Encountered unknown sentiment label outside {'negative', 'positive'}")

    df_clean["label"] = df_clean["label"].astype(int)
    return df_clean


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministically split preprocessed dataset into train, validation, and test sets.

    Matches the original notebook's stratified two-stage split:
    - 80% train, 20% temporary test (stratified by label)
    - 50/50 split of temporary test into validation and test (each 10% of total)

    Args:
        df: Preprocessed DataFrame with 'label' column.
        train_ratio: Proportion for training set (default: 0.8).
        val_ratio: Proportion for validation set (default: 0.1).
        test_ratio: Proportion for test set (default: 0.1).
        random_state: Random state for deterministic splitting (default: 42).

    Returns:
        (train_df, val_df, test_df) tuple.
    """
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
        raise ValueError("Ratios train_ratio + val_ratio + test_ratio must sum to 1.0")

    temp_size = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_state,
        stratify=df["label"],
    )

    test_share_of_temp = test_ratio / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_share_of_temp,
        random_state=random_state,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
