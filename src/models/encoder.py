"""
encoder.py

Pure transformer encoder over flattened per-frame landmark features.
No CNN, no GCN anywhere in this path -- a claim differentiator, not
a style choice. Plain full self-attention only (no windowed
attention, no KV-cache), per the architecture's non-negotiable
constraints.

Input per frame: pose (33,4) + face (478,3) + left_hand (21,3) +
right_hand (21,3), flattened and concatenated into one raw feature
vector per timestep, then linearly projected into hidden_dim.

Fallback path taken for Day 8 (no compatible pretrained encoder
exists for the full holistic input without violating the no-raw-video
constraint -- see Day 8 findings doc) -- all weights here are
randomly initialized, trained from scratch starting Week 2's smoke run.
"""

import torch
import torch.nn as nn

from src.models.positional_encoding import PositionalEncoding

# Raw per-frame feature dimensionality, computed from sample_schema.py's shapes:
POSE_DIM = 33 * 4
FACE_DIM = 478 * 3
HAND_DIM = 21 * 3  # each hand
RAW_FEATURE_DIM = POSE_DIM + FACE_DIM + HAND_DIM + HAND_DIM  # = 1692


class SignEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1000,
    ):
        super().__init__()

        self.input_projection = nn.Linear(RAW_FEATURE_DIM, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            # Plain full self-attention only -- no windowed attention,
            # no KV-cache scaffolding. Do not add these, even as a
            # "future toggle" -- see architecture constraints.
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.hidden_dim = hidden_dim

    def forward(self, landmarks, attention_mask):
        """
        landmarks: (batch, seq_len, RAW_FEATURE_DIM) -- already flattened
                   and concatenated (pose+face+left_hand+right_hand) per frame.
        attention_mask: (batch, seq_len) -- True/1 for real frames, False/0
                        for padding. Padding is masked out of attention
                        entirely; presence-but-occluded real frames (per
                        the interface spec) still participate, just with
                        zero-filled coordinates.
        """
        x = self.input_projection(landmarks)
        x = self.positional_encoding(x)

        # nn.TransformerEncoder expects a mask where True means "ignore" --
        # our attention_mask uses True = "real frame, attend to this",
        # so invert before passing in.
        src_key_padding_mask = ~attention_mask.bool()

        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return encoded  # (batch, seq_len, hidden_dim)