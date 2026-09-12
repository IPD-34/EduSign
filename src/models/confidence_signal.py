"""
confidence_signal.py

Day 53: the confidence-signal spec for Person D and Person A, finalized
and delivered. See docs/confidence_signal_spec.md for the full written
spec (semantics, caveats, verification). This file is the actual,
callable implementation the spec describes -- not a document to
translate into code yourselves.

SOURCE, RE-VERIFIED (Day 53, against the live model code, not assumed):
this reads ONLY the CTC head's own output, via
SignTransformer.forward_ctc_boundary_and_gate() (src/models/
sign_transformer.py) -- self.ctc_head, a plain nn.Linear(hidden_dim,
ctc_vocab_size+1), fed by the shared encoder pass. It does NOT touch
SignTransformer.forward() / self.decoder (the older single-token
autoregressive path, kept for WLASL-style isolated-sign decoding) --
that is a structurally separate nn.Module with a different call
signature (it additionally requires target_tokens, which this
inference-time signal does not have). Confirmed directly in this
session: every ASLLRP checkpoint's decoder.* weights are left at
fresh random init (load_checkpoint_partial only ever loads
encoder./ctc_head./boundary_head./bimanual_gate_head. prefixes for
these checkpoints) -- so the decoder path is not just the wrong
choice here, its weights are untrained garbage for this task. There is
no committed record in this repo of the original Month 1 Day 6
discussion to diff against line-by-line; this file instead re-derives
the source from the current, live code and states that explicitly
rather than assuming continuity with an undocumented prior decision.
"""

import torch

from src.models.sign_transformer import SignTransformer


def compute_confidence_signal(model: SignTransformer, landmarks: torch.Tensor,
                               encoder_attention_mask: torch.Tensor) -> dict:
    """
    Parameters
    ----------
    model : SignTransformer, constructed with use_bimanual_gate=True
        (required -- see spec doc for why the gate-active path is used).
    landmarks : (B, T, RAW_FEATURE_DIM) float32.
    encoder_attention_mask : (B, T) bool, True = real (non-padded) frame.

    Returns
    -------
    dict with three same-shape (B, T) tensors:
      confidence         : float32. Per-timestep max post-softmax
                            probability over the CTC head's full
                            (ctc_vocab_size + 1)-way distribution --
                            i.e. softmax(ctc_log_probs)[b, t, :].max().
                            NOT an aggregated per-utterance score --
                            see spec doc for why per-timestep was chosen.
      predicted_token_id : int64. argmax of that same distribution --
                            the id `confidence` is a probability FOR.
                            May equal blank_id (== ctc_vocab_size).
      is_blank           : bool. predicted_token_id == blank_id.
                            MUST be checked before treating `confidence`
                            as "the model is confident about sign
                            content" -- at the current checkpoint
                            (best_val_gated.pt, step 2000, NOT
                            converged -- see spec doc), high confidence
                            in blank is the common case, not a content
                            signal.
    """
    ctc_log_probs, _boundary_logits, _gate_logits = model.forward_ctc_boundary_and_gate(
        landmarks, encoder_attention_mask
    )
    blank_id = ctc_log_probs.shape[-1] - 1

    probs = torch.softmax(ctc_log_probs, dim=-1)
    confidence, predicted_token_id = probs.max(dim=-1)
    is_blank = predicted_token_id == blank_id

    return {
        "confidence": confidence,
        "predicted_token_id": predicted_token_id,
        "is_blank": is_blank,
    }
