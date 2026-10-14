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

Fix (attention-masking follow-up): nn.CTCLoss's `input_lengths` only
supports a contiguous-prefix-of-length-N convention -- it has no API for
an arbitrary/interior invalid-position mask (checked: there isn't one;
every CTC implementation, this included, assumes real frames are
[0, input_length) and padding/invalid frames come after). That
assumption used to hold automatically because encoder_attention_mask's
only source of False was trailing padding. It no longer does:
src/models/dataset.py's collate_fn can now mark a genuine LEADING
dropout timestep invalid too (see that module's docstring), which is a
hole at the START of the mask, not the end. reorder_valid_frames_first()
below fixes this the only structurally-correct way available: reorder
each sample's timesteps (for the CTC loss computation only) so its valid
frames really do form a prefix starting at index 0, in their original
relative order -- equivalent to simply skipping the leading frame that
was never observed, which is exactly what CTC should do with it. This
does not touch the encoder's own attention masking (already correct;
src_key_padding_mask has no positional-prefix assumption) or the
boundary loss (already position-order-agnostic, see below).
"""

import torch
import torch.nn as nn


def reorder_valid_frames_first(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Reorders each batch row of `x` along the time dimension (dim=1) so
    positions where `mask` is True come first, in their original relative
    order, followed by positions where `mask` is False -- without
    changing which VALUES are valid, only WHERE in the tensor they sit.

    x: (B, T) or (B, T, C). mask: (B, T) bool, True = valid/real frame.
    Returns a new tensor, same shape as x. After this, `mask.sum(dim=1)`
    is safe to use directly as nn.CTCLoss's `input_lengths` -- the valid
    frames are now guaranteed to be a genuine prefix.

    Uses a stable sort (ties -- i.e. frames on the same side of the
    valid/invalid split -- keep their original relative order) so this is
    a pure reordering, never a reshuffling of which valid frame comes
    before which other valid frame.
    """
    sort_key = (~mask).to(torch.float32)  # 0.0 = valid (sorts first), 1.0 = invalid
    perm = torch.argsort(sort_key, dim=1, stable=True)  # (B, T)

    if x.dim() == 2:
        return torch.gather(x, dim=1, index=perm)
    index = perm.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, dim=1, index=index)


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
    # Reorder so valid frames are a genuine prefix before computing
    # input_lengths from the mask -- see reorder_valid_frames_first's
    # docstring for why this is required now (leading-dropout timesteps
    # can put a hole at the START of encoder_attention_mask).
    ctc_log_probs_reordered = reorder_valid_frames_first(ctc_log_probs, encoder_attention_mask)
    # nn.CTCLoss expects (T, B, C) -- time-major, not batch-major.
    ctc_log_probs_t_major = ctc_log_probs_reordered.permute(1, 0, 2)  # (T, B, C)

    input_lengths = encoder_attention_mask.sum(dim=1)  # (B,) real (non-padded, non-invalid) timesteps per sample

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