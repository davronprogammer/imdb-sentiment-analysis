"""PyTorch Long Short-Term Memory (LSTM) model for binary sentiment classification."""

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """LSTM-based binary sentiment classification model.

    Architecture:
    1. Embedding layer: maps token indices to dense vectors.
    2. Single-layer LSTM: processes word sequences sequentially.
    3. Fully Connected (Linear) head: maps the final hidden state to a single logit.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of token indices with shape [batch_size, sequence_length].

        Returns:
            Raw logit tensor with shape [batch_size].
        """
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # LSTM output:
        # output: [batch_size, seq_len, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell:   [num_layers, batch_size, hidden_dim]
        _, (hidden, _) = self.lstm(embedded)

        # Extract last layer's final hidden state: [batch_size, hidden_dim]
        last_hidden = hidden[-1]

        # Linear projection to single logit: [batch_size, 1]
        logits = self.fc(last_hidden)

        # Return [batch_size] 1D tensor
        return logits.view(-1)
