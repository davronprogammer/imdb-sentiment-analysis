"""Unit and integration tests for IMDB Sentiment Classification pipeline."""

import pytest
import numpy as np
import pandas as pd
import torch

from src.features.tokenizer import clean_text, tokenize
from src.features.vocabulary import Vocabulary
from src.features.encoding import encode_tokens, encode_text
from src.data.loader import preprocess_dataframe, split_data
from src.data.dataset import IMDBDataset
from src.models.lstm import SentimentLSTM
from src.models.gru import SentimentGRU
from src.models.base import count_parameters
from src.evaluation.metrics import compute_classification_metrics


def test_text_preprocessing_html_cleaning():
    raw = "A <br /><br />great movie! Visit http://example.com & enjoy 100%!"
    cleaned = clean_text(raw)
    assert "<br" not in cleaned
    assert "http" not in cleaned
    assert "100" not in cleaned
    assert "!" not in cleaned
    assert "%" not in cleaned
    assert cleaned == "a great movie visit enjoy"


def test_vocabulary_special_tokens():
    corpus = ["movie was great", "film was bad and terrible"]
    vocab = Vocabulary.build_from_texts(corpus, max_size=10, pad_token="<PAD>", unk_token="<UNK>")

    assert vocab.pad_token == "<PAD>"
    assert vocab.unk_token == "<UNK>"
    assert vocab.pad_idx == 0
    assert vocab.unk_idx == 1
    assert vocab["<PAD>"] == 0
    assert vocab["<UNK>"] == 1
    assert vocab.lookup_index(0) == "<PAD>"
    assert vocab.lookup_index(1) == "<UNK>"


def test_unknown_token_handling():
    corpus = ["hello world"]
    vocab = Vocabulary.build_from_texts(corpus, max_size=10)

    # Word in vocabulary
    assert vocab["hello"] > 1
    # Out of vocabulary word must map to UNK index (1)
    assert vocab["extraterrestrial"] == 1
    assert vocab.lookup_token("nonexistent_word_xyz") == 1


def test_encoding_and_padding_lengths():
    vocab = Vocabulary(
        token_to_idx={"<PAD>": 0, "<UNK>": 1, "good": 2, "movie": 3},
        pad_idx=0,
        unk_idx=1,
    )

    # Short sentence should be post-padded
    short_text = "good movie"
    encoded_short = encode_text(short_text, vocab, max_length=5)
    assert len(encoded_short) == 5
    assert encoded_short == [2, 3, 0, 0, 0]

    # Long sentence should be truncated
    long_text = "good movie good movie good movie"
    encoded_long = encode_text(long_text, vocab, max_length=4)
    assert len(encoded_long) == 4
    assert encoded_long == [2, 3, 2, 3]


def test_label_mapping():
    df = pd.DataFrame({
        "review": ["awesome film", "awful waste of time"],
        "sentiment": ["positive", "negative"]
    })
    processed = preprocess_dataframe(df)
    assert "label" in processed.columns
    assert processed.loc[0, "label"] == 1
    assert processed.loc[1, "label"] == 0


def test_data_split_integrity():
    # 100 sample dataframe
    df = pd.DataFrame({
        "review": [f"review {i}" for i in range(100)],
        "sentiment": ["positive"] * 50 + ["negative"] * 50,
        "label": [1] * 50 + [0] * 50
    })

    train_df, val_df, test_df = split_data(
        df,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42
    )

    assert len(train_df) == 80
    assert len(val_df) == 10
    assert len(test_df) == 10

    # Ensure no data leakage / overlap
    train_indices = set(train_df["review"])
    val_indices = set(val_df["review"])
    test_indices = set(test_df["review"])

    assert len(train_indices.intersection(val_indices)) == 0
    assert len(train_indices.intersection(test_indices)) == 0
    assert len(val_indices.intersection(test_indices)) == 0

    # Ensure stratification
    assert train_df["label"].value_counts()[0] == 40
    assert train_df["label"].value_counts()[1] == 40
    assert val_df["label"].value_counts()[0] == 5
    assert val_df["label"].value_counts()[1] == 5
    assert test_df["label"].value_counts()[0] == 5
    assert test_df["label"].value_counts()[1] == 5


def test_model_output_shapes():
    batch_size = 8
    seq_len = 200
    vocab_size = 1000
    embedding_dim = 128
    hidden_dim = 128

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    lstm = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, padding_idx=0)
    gru = SentimentGRU(vocab_size, embedding_dim, hidden_dim, padding_idx=0)

    lstm_out = lstm(dummy_input)
    gru_out = gru(dummy_input)

    assert lstm_out.shape == torch.Size([batch_size])
    assert gru_out.shape == torch.Size([batch_size])
    assert lstm_out.dtype == torch.float32
    assert gru_out.dtype == torch.float32


def test_parameter_count_calculation():
    vocab_size = 30000
    embedding_dim = 128
    hidden_dim = 128

    lstm = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, padding_idx=0)
    gru = SentimentGRU(vocab_size, embedding_dim, hidden_dim, padding_idx=0)

    lstm_params = count_parameters(lstm)
    gru_params = count_parameters(gru)

    # Verification against theoretical PyTorch architecture:
    # Embedding: 30000 * 128 = 3,840,000
    # LSTM: 4 * (128*128 + 128*128 + 128 + 128) = 4 * (16384 + 16384 + 256) = 4 * 33024 = 132,096
    # Linear: 128 * 1 + 1 = 129
    # Total LSTM = 3,840,000 + 132,096 + 129 = 3,972,225
    assert lstm_params == 3972225

    # GRU: 3 * (128*128 + 128*128 + 128 + 128) = 3 * 33024 = 99,072
    # Total GRU = 3,840,000 + 99,072 + 129 = 3,939,201
    assert gru_params == 3939201

    # Exact parameter difference = 1 gate of hidden size: 33,024
    assert lstm_params - gru_params == 33024


def test_metric_calculations():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])  # 6 correct out of 8 (accuracy = 0.75)

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert metrics["total_samples"] == 8
    assert "confusion_matrix" in metrics
    assert "positive_f1" in metrics
    assert "macro_f1" in metrics
    assert len(metrics["confusion_matrix"]) == 2
