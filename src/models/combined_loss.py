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

--- STAGE A ADDITION: BLANK DOWNWEIGHTING, OFF BY DEFAULT ---
nn.CTCLoss has no native per-class weighting -- the blank id is baked
into its forward-backward algorithm as a structural symbol, not an
exposed weight. Two options existed to work around that: (a) reweight
the blank channel's log-probability before it reaches nn.CTCLoss, or
(b) add a separate auxiliary entropy/blank-penalty term alongside the
unmodified CTC loss. Chose (a): (b) (an ad-hoc "average softmax entropy"
term) is NOT the same regularizer as the peer-reviewed EnCTC approach
to this exact known CTC pathology (that method's entropy term is
computed from the forward-backward algorithm's internal alpha/beta
marginals, not exposed by nn.CTCLoss as a black box) -- calling a
softmax-entropy proxy "the same fix" would overclaim a theoretical
grounding it doesn't have. (a) is simple, exactly what it says, and
needs no access to nn.CTCLoss internals: apply_blank_downweight() below
subtracts blank_logit_bias from the blank column of the model's
ALREADY-COMPUTED, already-normalized ctc_log_probs, then re-normalizes
via log_softmax. This is mathematically EXACTLY equivalent to
subtracting the same bias from the blank channel's PRE-softmax logit
(log_softmax is shift-invariant: log_softmax(log_probs - bias*e_blank)
== log_softmax(logits - bias*e_blank) for any per-row constant already
absorbed by the first log_softmax -- algebra: log_probs = logits -
logsumexp(logits), so log_probs with one entry shifted equals logits
with that entry shifted, minus the SAME constant logsumexp(logits)
applied to every entry, which log_softmax is invariant to). This means
no change to SignTransformer/forward_ctc_boundary_and_gate is needed --
the model's returned ctc_log_probs stay exactly what they've always
been (still what eval/decode/confidence-signal consumers see); only a
LOCAL COPY used for THIS loss computation is biased. blank_logit_bias
defaults to 0.0, an exact no-op (early-return, not just an
algebraically-zero operation) -- existing behavior is byte-identical
when this parameter is left unset, same pattern as every other
additive change this project has made (use_bimanual_gate,
precomputed_projection, gate_reg_weight, etc.).
"""

"""
--- ADDITIVE: TRANSITION-LOSS MECHANISM (boundary-mechanisms smoke test) ---
Second, additive boundary-supervision term -- see boundary_labels.
build_transition_targets' docstring for the target construction. This
mechanism deliberately reuses the EXISTING boundary_head's output
(boundary_logits) rather than adding a new head: adding a new head
would mean fresh, randomly-initialized parameters that this
mechanism's short smoke-test horizon would have to train from
scratch, confounding "does a sharper target help" with "does the
model tolerate an undertrained new head" -- reusing the existing head
keeps this an isolated one-variable change, exactly the same
resuming-from-checkpoint discipline as every other lever in this
project (blank_logit_bias, gate_reg_weight, boundary_loss_weight).

compute_transition_loss() mirrors compute_gate_entropy()'s pattern
exactly: a small standalone helper, always computable and loggable,
that a wrapper adds into total_loss ONLY when explicitly weighted.
compute_combined_loss_with_gate_and_transition() wraps
compute_combined_loss_with_gate() UNCHANGED -- does not duplicate or
alter its CTC/boundary/gate math -- so every existing caller of
compute_combined_loss_with_gate() (or compute_combined_loss()) is
completely unaffected by this addition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_blank_downweight(ctc_log_probs, blank_id, blank_logit_bias):
    """
    ctc_log_probs: (B, T, ctc_vocab_size + 1), already log_softmax'd
    (e.g. straight from SignTransformer.forward_ctc_boundary_and_gate).
    blank_logit_bias: subtracted from the blank channel, in logit space
    (see module docstring for the exact equivalence proof) -- a POSITIVE
    value makes blank LESS attractive to the loss, discouraging the
    trivial all-blank collapse. 0.0 (default) is an exact no-op.

    Returns a NEW tensor -- ctc_log_probs itself is never mutated, so
    any other consumer of the same tensor (e.g. the boundary loss call
    site, or a caller inspecting the raw model output) is unaffected.
    """
    if blank_logit_bias == 0.0:
        return ctc_log_probs
    biased = ctc_log_probs.clone()
    biased[..., blank_id] = biased[..., blank_id] - blank_logit_bias
    return F.log_softmax(biased, dim=-1)


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
    blank_logit_bias=0.0,
):
    """
    Returns (total_loss, ctc_loss, boundary_loss) -- all three returned
    separately so the training loop can log and inspect them
    individually, per Day 32/33's own stated discipline of checking
    each loss term's magnitude before trusting the combined number.

    blank_logit_bias: Stage A addition, see module docstring for the
    mechanism and equivalence proof. 0.0 (default) is an exact no-op --
    ctc_loss_for_backward below is byte-identical to before this change
    whenever this parameter is left unset.
    """
    B, T, _ = ctc_log_probs.shape

    # --- CTC loss ---
    # Stage A: bias applied to a LOCAL COPY only (apply_blank_downweight
    # never mutates its input) -- ctc_log_probs itself, and anything else
    # that reads it (there is nothing else in this function, but future
    # callers might), is completely unaffected.
    ctc_log_probs_for_loss = apply_blank_downweight(ctc_log_probs, blank_id, blank_logit_bias)

    # nn.CTCLoss expects (T, B, C) -- time-major, not batch-major.
    ctc_log_probs_t_major = ctc_log_probs_for_loss.permute(1, 0, 2)  # (T, B, C)

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
    blank_logit_bias=0.0,
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
        blank_logit_bias=blank_logit_bias,
    )

    gate_entropy = compute_gate_entropy(gate_logits, encoder_attention_mask)
    total_loss = total_loss + gate_reg_weight * gate_entropy

    return total_loss, ctc_loss, boundary_loss, gate_entropy


def compute_transition_loss(boundary_logits, transition_target, encoder_attention_mask):
    """
    Additive transition-loss mechanism -- see module docstring and
    boundary_labels.build_transition_targets. Deliberately reuses
    boundary_logits (the EXISTING boundary_head's raw output) rather
    than a new head -- see module docstring for why.

    MSE, not BCE: transition_target is a soft, continuous Gaussian
    bump in [0, 1] (a real-valued "how close is this the change-point"
    signal), not a hard 0/1 class label the way boundary_target is --
    MSE against sigmoid(boundary_logits) is the natural loss for
    matching a soft peaked shape; BCE would treat 0.5 the same
    regardless of whether it came from the peak's shoulder or its
    tail, which is not the distinction this target is trying to draw.

    Masked-averaged over real (non-padded) timesteps only, same
    padding discipline as compute_combined_loss's boundary_loss.
    """
    boundary_prob = torch.sigmoid(boundary_logits)
    per_position_loss = (boundary_prob - transition_target) ** 2  # (B, T)
    mask = encoder_attention_mask.float()
    return (per_position_loss * mask).sum() / mask.sum().clamp(min=1.0)


def compute_combined_loss_with_gate_and_transition(
    ctc_log_probs,        # (B, T, ctc_vocab_size + 1)
    boundary_logits,      # (B, T)
    gate_logits,          # (B, T)
    target_tokens,        # (B, tgt_len)
    boundary_target,      # (B, T)
    encoder_attention_mask,  # (B, T)
    bos_id, eos_id, pad_id,
    blank_id,
    boundary_loss_weight=1.0,
    gate_reg_weight=0.0,
    blank_logit_bias=0.0,
    transition_target=None,   # (B, T) or None
    transition_loss_weight=0.0,
):
    """
    Wraps compute_combined_loss_with_gate() UNCHANGED (see module
    docstring) and adds the transition-loss mechanism's contribution
    as a separately-visible fifth return value.

    True no-op when transition_target is None or
    transition_loss_weight == 0.0 (both defaults): transition_loss is
    still computed and returned (0.0 tensor if transition_target is
    None, same "always computed and logged" discipline as
    compute_gate_entropy) but total_loss is byte-identical to calling
    compute_combined_loss_with_gate() directly.
    """
    total_loss, ctc_loss, boundary_loss, gate_entropy = compute_combined_loss_with_gate(
        ctc_log_probs, boundary_logits, gate_logits,
        target_tokens, boundary_target,
        encoder_attention_mask,
        bos_id, eos_id, pad_id, blank_id,
        boundary_loss_weight=boundary_loss_weight,
        gate_reg_weight=gate_reg_weight,
        blank_logit_bias=blank_logit_bias,
    )

    if transition_target is None:
        transition_loss = torch.tensor(0.0, device=boundary_logits.device)
    else:
        transition_loss = compute_transition_loss(boundary_logits, transition_target, encoder_attention_mask)

    total_loss = total_loss + transition_loss_weight * transition_loss

    return total_loss, ctc_loss, boundary_loss, gate_entropy, transition_loss