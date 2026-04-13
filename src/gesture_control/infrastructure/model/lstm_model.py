from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GestureLSTMClassifier(nn.Module):
    """2-layer LSTM classifier with additive temporal attention pooling."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        # Additive (Bahdanau-style scalar) attention: one score per time step.
        self.attn = nn.Linear(hidden_size, 1)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)            # (B, T, H)
        # Compute per-step attention scores and form a weighted context vector.
        scores = self.attn(output)          # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        context = (weights * output).sum(dim=1)  # (B, H)
        return self.head(self.dropout(context))
