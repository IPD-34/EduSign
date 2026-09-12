"""
bimanual_gate.py -- Day 43: C3 bimanual gating head, standalone module.

NOT wired into SignEncoder/SignTransformer/combined_loss yet (Day 44's
task). This file is self-contained and importable in isolation, per
Month 1 Day 9's discipline of validating shapes/gradients before wiring
into the full pipeline.

--- WHAT THIS MODULE DOES ---

BimanualGate: a lightweight MLP classifier that reads ONLY the two-hand
landmark subset of the flattened per-frame feature vector (confirmed
against feature_utils.flatten_frame_features()'s real concat order --
pose, face, left_hand, right_hand -- NOT assumed) and predicts, per
timestep, whether that frame's two-hand configuration is
symmetric/coupled or asymmetric/independent.

BimanualRoutedProjection: the routing mechanism itself (the C3 claim).
See its docstring below for the exact mechanism and why it is
implemented upstream of self-attention rather than inside it.

BimanualGateHead: wraps both into the single module Day 44 will consume.

--- INTERFACE SPEC (for Day 44 integration) ---
Input:  landmarks, shape (B, T, RAW_FEATURE_DIM), dtype float32.
        RAW_FEATURE_DIM is imported from encoder.py (currently 1692),
        never redefined here -- this project's recurring-bug-class
        lesson is to reuse the one source of truth for a shared
        constant, not risk a second copy silently drifting.
Output: BimanualGateHead.forward(landmarks) -> (routed_features, gate_logits)
        routed_features: (B, T, hidden_dim), float32 -- drop-in
            replacement for what SignEncoder.input_projection(landmarks)
            currently produces, if Day 44 chooses to wire it in there.
        gate_logits: (B, T), float32, raw (pre-sigmoid) logits.
            sigmoid(gate_logits) = P(symmetric) per timestep.
            BCEWithLogitsLoss-compatible, same convention as
            SignTransformer.boundary_head.
"""

import torch
import torch.nn as nn

from src.models.encoder import POSE_DIM, FACE_DIM, HAND_DIM, RAW_FEATURE_DIM

# --- Slice indices into the RAW_FEATURE_DIM vector -----------------------
# Confirmed against the real concat order in feature_utils.flatten_frame_features()
# (pose, face, left_hand, right_hand) and the real per-landmark shapes in
# src/landmarks/constants.py (POSE_LANDMARKS=33, FACE_LANDMARKS=478,
# HAND_LANDMARKS=21) -- not assumed. See scripts/test_day43_bimanual_gate.py
# for a runtime assertion that these match RAW_FEATURE_DIM exactly.
POSE_FACE_SLICE = slice(0, POSE_DIM + FACE_DIM)                        # [0:1566]
LEFT_HAND_SLICE = slice(POSE_DIM + FACE_DIM, POSE_DIM + FACE_DIM + HAND_DIM)       # [1566:1629]
RIGHT_HAND_SLICE = slice(POSE_DIM + FACE_DIM + HAND_DIM, RAW_FEATURE_DIM)          # [1629:1692]
POSE_FACE_HAND_DIM = POSE_DIM + FACE_DIM + HAND_DIM  # dim of (pose+face+one hand) = 1629


class BimanualGate(nn.Module):
    """
    Per-timestep symmetric-vs-asymmetric classifier.

    Reads ONLY the two-hand landmark subset (HAND_DIM * 2 = 126 of the
    1692 raw dims) -- deliberately excludes pose/face, so the gate's
    decision is driven purely by hand configuration, matching the C3
    claim's stated scope (a classifier over two-hand configuration, not
    a general-purpose per-frame feature).

    Plain MLP -- no CNN, no GCN over the 21 hand landmarks' spatial
    structure, per the architecture's non-negotiable constraints. This
    intentionally does not exploit landmark adjacency/graph structure;
    it is a flat classifier over the 63-dim-per-hand feature vector,
    same "flatten first, then linear" philosophy as encoder.py's own
    input_projection.
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HAND_DIM * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """landmarks: (B, T, RAW_FEATURE_DIM) -> gate_logits: (B, T)."""
        left = landmarks[..., LEFT_HAND_SLICE]
        right = landmarks[..., RIGHT_HAND_SLICE]
        hands = torch.cat([left, right], dim=-1)  # (B, T, HAND_DIM*2)
        gate_logits = self.net(hands).squeeze(-1)  # (B, T)
        return gate_logits


class BimanualRoutedProjection(nn.Module):
    """
    C3 routing mechanism.

    THE MECHANISM (precisely, for claim language):
      - "Symmetric" path (the default): landmarks are projected through
        ONE shared nn.Linear(RAW_FEATURE_DIM, hidden_dim) -- pose, face,
        and BOTH hands are linearly mixed together in a single matrix
        multiply. This is IDENTICAL to what encoder.py's existing
        input_projection already computes today -- "no change from
        default behavior" is literal here, not approximate.
      - "Asymmetric" path (new): pose+face+left_hand and pose+face+
        right_hand are each projected through their OWN separate
        nn.Linear(POSE_FACE_HAND_DIM, hidden_dim) -- no weight is ever
        shared between the two hands' projections, so no single linear
        map mixes left- and right-hand landmark values together. The
        two independent projections are combined by elementwise mean.
      - Per-timestep output is a SOFT interpolation between the two
        paths, weighted by sigmoid(gate_logits) = P(symmetric):
            routed = p_symmetric * joint  +  (1 - p_symmetric) * independent
        A soft gate (not a hard argmax switch) keeps the module
        differentiable end-to-end and avoids a brittle discrete routing
        decision this early in training/architecture life.

    WHY THIS SITS UPSTREAM OF SELF-ATTENTION, NOT INSIDE IT:
      SignEncoder's self-attention operates over TIME -- each frame is
      already a single fused token (pose+face+both hands concatenated)
      before attention ever runs; there are no separate per-hand tokens
      for attention to route between. So "separate K/V per hand" or a
      "within-hand vs cross-hand attention mask" is not literally
      meaningful against the existing tokenization without restructuring
      it -- out of scope for today (that would mean editing encoder.py).
      This module instead changes WHAT enters each timestep's token
      (jointly-fused vs. independently-fused hand information) and
      leaves HOW attention is computed completely untouched: downstream
      self-attention remains full, dense, and un-windowed over all T
      timesteps, exactly as today. No mask, no -inf, no restricted
      connectivity is introduced anywhere by this module -- it produces
      the SAME SHAPE of dense (B, T, hidden_dim) sequence that
      input_projection already produces, so it is a drop-in replacement
      for that one call, not a change to the transformer stack itself.
      This keeps the non-negotiable "full self-attention only, no
      windowed attention" constraint trivially satisfied: the attention
      module this feeds into (if wired in on Day 44) is never modified.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.joint_proj = nn.Linear(RAW_FEATURE_DIM, hidden_dim)
        self.left_proj = nn.Linear(POSE_FACE_HAND_DIM, hidden_dim)
        self.right_proj = nn.Linear(POSE_FACE_HAND_DIM, hidden_dim)

    def forward(self, landmarks: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        landmarks: (B, T, RAW_FEATURE_DIM)
        gate_logits: (B, T) -- raw logits from BimanualGate.
        Returns: routed features, (B, T, hidden_dim).
        """
        pose_face = landmarks[..., POSE_FACE_SLICE]
        left_input = torch.cat([pose_face, landmarks[..., LEFT_HAND_SLICE]], dim=-1)
        right_input = torch.cat([pose_face, landmarks[..., RIGHT_HAND_SLICE]], dim=-1)

        joint = self.joint_proj(landmarks)  # (B, T, hidden_dim)
        independent = 0.5 * (self.left_proj(left_input) + self.right_proj(right_input))

        p_symmetric = torch.sigmoid(gate_logits).unsqueeze(-1)  # (B, T, 1)
        routed = p_symmetric * joint + (1.0 - p_symmetric) * independent
        return routed


class BimanualGateHead(nn.Module):
    """Wraps BimanualGate + BimanualRoutedProjection into the single module
    Day 44 would consume. See module docstring for the interface spec."""

    def __init__(self, hidden_dim: int = 256, gate_hidden_dim: int = 64, gate_dropout: float = 0.1):
        super().__init__()
        self.gate = BimanualGate(hidden_dim=gate_hidden_dim, dropout=gate_dropout)
        self.routed_projection = BimanualRoutedProjection(hidden_dim=hidden_dim)

    def forward(self, landmarks: torch.Tensor):
        """landmarks: (B, T, RAW_FEATURE_DIM) -> (routed_features, gate_logits)."""
        gate_logits = self.gate(landmarks)
        routed_features = self.routed_projection(landmarks, gate_logits)
        return routed_features, gate_logits
