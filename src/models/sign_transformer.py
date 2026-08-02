"""
sign_transformer.py

Combines SignEncoder + SignDecoder into the full model. This is the
Day 9 architectural skeleton -- untrained (random init, per Day 8's
fallback decision), structurally complete, respecting every
non-negotiable constraint (landmark-only input, no CNN/GCN, full
self-attention only).
"""

import torch.nn as nn

from src.models.encoder import SignEncoder
from src.models.decoder import SignDecoder


class SignTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.encoder = SignEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = SignDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )

    def forward(self, landmarks, encoder_attention_mask, target_tokens, target_padding_mask=None):
        encoder_output = self.encoder(landmarks, encoder_attention_mask)
        logits = self.decoder(
            target_tokens, encoder_output, encoder_attention_mask, target_padding_mask
        )
        return logits