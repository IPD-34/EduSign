"""
combined_loss.py

Day 32: combines CTC loss (gloss-sequence recognition) and boundary
loss (per-timestep boundary prediction) into a single training step.

CTC's target format is NOT the same as target_tokens' format:
target_tokens is [BOS, gloss_1, ..., gloss_n, EOS], padded with pad_id
-- built for the autoregressive decoder path. CTC needs just the raw
gloss sequence (no BOS/EOS/padding) plus each sample's real length,
since CTCLoss internally handles alignment via the blank token and
does NOT want start/end markers polluting the target sequence.
"""

import torch
import torch.nn as nn


def strip_special_tokens(target_tokens, bos_id, eos_id, pad_id):
    """
    Given a padded batch of [BOS, g1, ..., gn, EOS, PAD, ...] sequences,
    returns (flat_targets, target_lengths) in the format nn.CTCLoss
    expects: a single 1D tensor of concatenated real gloss ids, and a
    1D tensor of each sample's real gloss count (excluding BOS/EOS/PAD).
    """
    flat_targets = []
    target_lengths = []

    for row in target_tokens:
        ids = row.tolist()
        # Strip BOS (always first), then take everything up to (not
        # including) the first EOS or PAD, whichever comes first --
        # PAD only appears after EOS in this dataset's construction,
        # but checking both is a cheap safety net against silent
        # misalignment if that ever changes.
        assert ids[0] == bos_id, f"Expected BOS at position 0, got {ids[0]}"
        gloss_ids = []
        for tok in ids[1:]:
            if tok == eos_id or tok == pad_id:
                break
            gloss_ids.append(tok)
        flat_targets.extend(gloss_ids)
        target_lengths.append(len(gloss_ids))

    flat_targets = torch.tensor(flat_targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return flat_targets, target_lengths


def compute_combined_loss(
    ctc_log_probs,        # (B, T, ctc_vocab_size + 1)
    boundary_logits,      # (B, T)
    target_tokens,        # (B, tgt_len) -- [BOS, gloss..., EOS, PAD...]
    boundary_target,      # (B, T) -- float, 0/1
    encoder_attention_mask,  # (B, T) -- bool, True = real frame
    bos_id, eos_id, pad_id,
    blank_id,
    boundary_loss_weight=1.0,
):
    """
    Returns (total_loss, ctc_loss, boundary_loss) -- all three returned
    separately so the training loop can log and inspect them
    individually, per Day 32/33's own stated discipline of checking
    each loss term's magnitude before trusting the combined number.
    """
    B, T, _ = ctc_log_probs.shape

    # --- CTC loss ---
    # nn.CTCLoss expects (T, B, C) -- time-major, not batch-major.
    ctc_log_probs_t_major = ctc_log_probs.permute(1, 0, 2)  # (T, B, C)

    input_lengths = encoder_attention_mask.sum(dim=1)  # (B,) real (non-padded) timesteps per sample

    flat_targets, target_lengths = strip_special_tokens(target_tokens, bos_id, eos_id, pad_id)

    # zero_infinity=False deliberately, per Day 33's plan: understand
    # WHY a sample produces inf (usually input_length < target_length)
    # rather than silently suppressing it. If this line raises/produces
    # inf during the Day 32 smoke test, that IS the finding Day 33's
    # corpus-wide audit is meant to catch systematically -- don't flip
    # this to True just to make an error disappear.
    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=False)
    ctc_loss = ctc_loss_fn(ctc_log_probs_t_major, flat_targets, input_lengths, target_lengths)

    # --- Boundary loss ---
    # BCEWithLogitsLoss with reduction="none" so padded positions can
    # be excluded from the mean -- a padded timestep has no real
    # boundary label and must not contribute to the loss, same
    # padding-vs-real distinction this project has enforced since Day 10.
    bce_fn = nn.BCEWithLogitsLoss(reduction="none")
    per_position_loss = bce_fn(boundary_logits, boundary_target)  # (B, T)

    mask = encoder_attention_mask.float()
    boundary_loss = (per_position_loss * mask).sum() / mask.sum().clamp(min=1.0)

    total_loss = ctc_loss + boundary_loss_weight * boundary_loss

    return total_loss, ctc_loss, boundary_loss