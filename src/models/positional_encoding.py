"""
positional_encoding.py

Standard sinusoidal positional encoding, added to input embeddings
before the transformer stack. No learned position embeddings (keeps
things simple for a from-scratch smoke test); revisit if Month 2
tuning suggests learned positions help.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, hidden_dim)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)