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


def compute_gate_entropy(gate_logits, encoder_attention_mask):
    """
    Day 44: self-supervised confidence signal for the C3 bimanual gate --
    no ground-truth symmetric/asymmetric label needed (Person D's split is
    only for the Day 45/48 ablation SCORING, not for training this gate).

    Binary entropy of sigmoid(gate_logits) per real (non-padded) timestep,
    masked-averaged. Max value ln(2)~=0.6931 at p=0.5 (gate is indecisive,
    output ~symmetric-vs-asymmetric coin flip everywhere); lower values mean
    the gate is making confident calls either way. Always computed and
    returned for logging when gate_logits is provided -- per this project's
    rule to never hide a term's parts behind a combined number -- regardless
    of whether it is weighted into total_loss.
    """
    p = torch.sigmoid(gate_logits)
    eps = 1e-7
    entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))  # (B, T)
    mask = encoder_attention_mask.float()
    return (entropy * mask).sum() / mask.sum().clamp(min=1.0)


def compute_combined_loss_with_gate(
    ctc_log_probs,        # (B, T, ctc_vocab_size + 1)
    boundary_logits,      # (B, T)
    gate_logits,          # (B, T) -- Day 44 addition
    target_tokens,        # (B, tgt_len)
    boundary_target,      # (B, T)
    encoder_attention_mask,  # (B, T)
    bos_id, eos_id, pad_id,
    blank_id,
    boundary_loss_weight=1.0,
    gate_reg_weight=0.0,
):
    """
    Day 44: three-way combined loss for the gate-enabled model. Wraps
    compute_combined_loss() UNCHANGED (does not duplicate or alter its
    CTC/boundary math, so every existing caller of that function is
    completely unaffected by this addition) and adds the gate's
    contribution as a separately-visible fourth return value.

    TRAINING SCHEME DECISION (documented per this project's standing rule
    that every fallback/deviation gets its reasoning written down before
    implementation, same as the boundary-label and vocab-split decisions):
    JOINT / UNSUPERVISED. The gate has no ground-truth label at all. Its
    only training signal is the gradient that reaches bimanual_gate_head's
    parameters through how routed_features change ctc_loss/boundary_loss
    via the shared encoder pass -- that IS the training signal, not a
    placeholder for a future supervised one. gate_reg_weight defaults to
    0.0: the entropy term above is always computed and logged, but
    contributes NOTHING to total_loss unless explicitly weighted, so a
    comparison against a pre-gate baseline isolates exactly one variable
    (the routing mechanism itself) rather than two (routing + an added
    regularization pressure). A weak-supervision term via Battison's
    Dominance/Symmetry Conditions was considered as the alternative (per
    Day 44's brief) but deliberately NOT built today: it would need real
    heuristic labeling logic, which is explicitly Day 45's task, and
    building it before checking whether pure joint training is even
    ambiguous would be solving a problem that may not exist. If Day 47's
    dedicated collapse check finds the gate is not doing anything
    non-trivial under pure joint training, weak supervision (via
    gate_reg_weight, or a real Battison's-heuristic target) is the
    documented fallback to revisit then -- not decided by default now.
    """
    total_loss, ctc_loss, boundary_loss = compute_combined_loss(
        ctc_log_probs, boundary_logits,
        target_tokens, boundary_target,
        encoder_attention_mask,
        bos_id, eos_id, pad_id, blank_id,
        boundary_loss_weight=boundary_loss_weight,
    )

    gate_entropy = compute_gate_entropy(gate_logits, encoder_attention_mask)
    total_loss = total_loss + gate_reg_weight * gate_entropy

    return total_loss, ctc_loss, boundary_loss, gate_entropy