"""
decoder.py

Standard transformer decoder: causal self-attention over gloss
tokens generated so far, cross-attention to the encoder's output.
Full self-attention only, matching the encoder's constraints.
"""

import torch
import torch.nn as nn

from src.models.positional_encoding import PositionalEncoding


class SignDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 100,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, target_tokens, encoder_output, encoder_padding_mask, target_padding_mask=None):
        """
        target_tokens: (batch, tgt_len) -- gloss token ids, teacher-forced
                       during training (shifted right, starting with <BOS>).
        encoder_output: (batch, src_len, hidden_dim) -- from SignEncoder.
        encoder_padding_mask: (batch, src_len) -- True/1 for real encoder
                              frames, False/0 for padding (same convention
                              as SignEncoder's attention_mask).
        target_padding_mask: (batch, tgt_len) -- True/1 for real target
                             tokens, False/0 for padding. Optional (only
                             needed for batch_size > 1 with variable-length
                             gloss sequences).
        """
        batch_size, tgt_len = target_tokens.shape
        device = target_tokens.device

        x = self.token_embedding(target_tokens)
        x = self.positional_encoding(x)

        # Causal mask: prevents attending to future tokens
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device), diagonal=1
        )

        memory_key_padding_mask = ~encoder_padding_mask.bool()
        tgt_key_padding_mask = ~target_padding_mask.bool() if target_padding_mask is not None else None

        decoded = self.transformer_decoder(
            x,
            encoder_output,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.output_projection(decoded)
        return logits  # (batch, tgt_len, vocab_size)