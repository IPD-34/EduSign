"""
sign_transformer.py

Combines SignEncoder + SignDecoder into the full model. This is the
Day 9 architectural skeleton -- untrained (random init, per Day 8's
fallback decision), structurally complete, respecting every
non-negotiable constraint (landmark-only input, no CNN/GCN, full
self-attention only).

Day 30 addition: a CTC output head, run directly off the encoder,
for ASLLRP's multi-gloss sequence supervision. This is a parallel
path, not a replacement -- the decoder path above stays intact for
WLASL-style single-gloss decoding.
"""

import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import SignEncoder
from src.models.decoder import SignDecoder
from src.models.bimanual_gate import BimanualGateHead


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
        ctc_vocab_size: int | None = None,
        use_bimanual_gate: bool = False,
        bimanual_gate_hidden_dim: int = 64,
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

        # Day 30: CTC head. +1 for the blank token, per nn.CTCLoss's
        # convention. blank_id is ctc_vocab_size (i.e. the id one past
        # the last real token) -- this must match whatever blank= value
        # is passed to nn.CTCLoss at the call site, and must NOT collide
        # with <PAD> or any other existing special token id.
        self.ctc_head = None
        if ctc_vocab_size is not None:
            self.ctc_head = nn.Linear(hidden_dim, ctc_vocab_size + 1)

        self.boundary_head = nn.Linear(hidden_dim, 1)

        # Day 44: C3 bimanual gate. Optional, off by default -- constructing
        # SignTransformer without this flag is byte-for-byte the same model
        # as before this change (every existing checkpoint/call site is
        # unaffected). See src/models/bimanual_gate.py for the mechanism:
        # a per-timestep soft gate that chooses between the encoder's
        # existing joint input_projection (symmetric case, unchanged) and
        # an independent per-hand projection (asymmetric case) -- feeding
        # into the SAME, otherwise-untouched SignEncoder self-attention.
        self.use_bimanual_gate = use_bimanual_gate
        self.bimanual_gate_head = None
        if use_bimanual_gate:
            self.bimanual_gate_head = BimanualGateHead(
                hidden_dim=hidden_dim, gate_hidden_dim=bimanual_gate_hidden_dim
            )

    def forward(self, landmarks, encoder_attention_mask, target_tokens, target_padding_mask=None):
        encoder_output = self.encoder(landmarks, encoder_attention_mask)
        logits = self.decoder(
            target_tokens, encoder_output, encoder_attention_mask, target_padding_mask
        )
        return logits

    def forward_ctc(self, landmarks, encoder_attention_mask):
        """
        Returns log-probabilities, shaped (batch, seq_len, ctc_vocab_size + 1),
        ready to feed directly into nn.CTCLoss.

        nn.CTCLoss expects log-probabilities, not raw logits and not
        softmax probabilities -- log_softmax here is required, not a
        style choice. Getting this wrong won't crash; it'll just produce
        a plausible-looking wrong loss.
        """
        if self.ctc_head is None:
            raise RuntimeError(
                "SignTransformer was constructed without ctc_vocab_size; "
                "forward_ctc() is unavailable."
            )
        encoder_output = self.encoder(landmarks, encoder_attention_mask)  # (B, T, hidden_dim)
        ctc_logits = self.ctc_head(encoder_output)  # (B, T, ctc_vocab_size + 1)
        log_probs = F.log_softmax(ctc_logits, dim=-1)
        return log_probs

    def forward_ctc_and_boundary(self, landmarks, encoder_attention_mask):
        """
        Runs the encoder ONCE and produces both CTC log-probs and raw
        boundary logits from the same encoder output. This shared
        encoder pass is itself part of the combined-system claim (see
        Month 2 plan: "the two directions share the encoder's
        confidence-estimation signal") -- do not call the encoder
        twice for these two heads.
        """
        if self.ctc_head is None:
            raise RuntimeError("SignTransformer was constructed without ctc_vocab_size.")

        encoder_output = self.encoder(landmarks, encoder_attention_mask)  # (B, T, hidden_dim)

        ctc_logits = self.ctc_head(encoder_output)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)  # (B, T, ctc_vocab_size + 1)

        boundary_logits = self.boundary_head(encoder_output).squeeze(-1)  # (B, T)

        return ctc_log_probs, boundary_logits

    def forward_ctc_boundary_and_gate(self, landmarks, encoder_attention_mask):
        """
        Day 44: three-way shared pass -- CTC, boundary, AND the C3 gate,
        off ONE encoder call, same "shared conditioning across both
        directions is itself part of the combined system claim" principle
        already used for forward_ctc_and_boundary().

        The gate computes routed_features (its own per-timestep, gated
        choice between a joint and an independent-per-hand projection of
        the raw landmarks -- see bimanual_gate.py) and hands them to
        SignEncoder as precomputed_projection, which makes SignEncoder
        skip its own input_projection call and use these instead.
        Everything after that -- positional encoding, self-attention,
        both heads -- is the exact same code path as
        forward_ctc_and_boundary(), run exactly once.
        """
        if self.bimanual_gate_head is None:
            raise RuntimeError(
                "SignTransformer was constructed without use_bimanual_gate=True; "
                "forward_ctc_boundary_and_gate() is unavailable."
            )

        routed_features, gate_logits = self.bimanual_gate_head(landmarks)

        encoder_output = self.encoder(
            landmarks, encoder_attention_mask, precomputed_projection=routed_features
        )  # (B, T, hidden_dim) -- single shared encoder pass

        ctc_logits = self.ctc_head(encoder_output)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)

        boundary_logits = self.boundary_head(encoder_output).squeeze(-1)

        return ctc_log_probs, boundary_logits, gate_logits