"""
test_combined_loss.py

Tests for the CTC leading-invalid-frame fix in src/models/combined_loss.py.

nn.CTCLoss's `input_lengths` assumes valid frames are a contiguous prefix
[0, length) -- it has no API for an arbitrary invalid-position mask. Once
src/models/dataset.py's collate_fn could mark a genuine LEADING dropout
timestep invalid (not just trailing padding), that assumption broke:
CTCLoss would read starting from position 0 (an invalid frame) for
mask.sum() positions, using a shifted/wrong window instead of the real
valid frames. reorder_valid_frames_first() fixes this by moving valid
frames to a genuine prefix (stable order-preserving reorder) before
computing input_lengths.

These tests prove the fix with a hand-constructed, deterministic case
where the WRONG window is CTC-infeasible (forces zero_infinity=False's
+inf) and the CORRECT window is a clean, low-loss alignment -- not just
"doesn't crash."
"""

from __future__ import annotations

import torch

from src.models.combined_loss import compute_combined_loss, reorder_valid_frames_first

_BLANK_ID = 3  # class 3 in a 4-class toy vocab (0, 1, 2, blank=3)
_BIG = 20.0  # logit magnitude strong enough that log_softmax is ~deterministic


def _one_hot_logits(class_id: int, num_classes: int = 4) -> torch.Tensor:
    logits = torch.full((num_classes,), -_BIG)
    logits[class_id] = _BIG
    return logits


# ---------------------------------------------------------------------
# reorder_valid_frames_first: exact permutation on a known case
# ---------------------------------------------------------------------


def test_reorder_valid_frames_first_moves_leading_invalid_to_end():
    # (B=1, T=4, C=2): frame values are just markers 0,1,2,3 (broadcast
    # over the class dim) so the reordering is easy to read off directly.
    x = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    mask = torch.tensor([[False, True, True, True]])  # frame 0 invalid (leading dropout)

    out = reorder_valid_frames_first(x, mask)

    # Valid frames (1, 2, 3) first in original order, invalid frame (0) last.
    expected = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [0.0, 0.0]]])
    torch.testing.assert_close(out, expected)


def test_reorder_valid_frames_first_no_op_when_all_valid():
    x = torch.tensor([[[0.0], [1.0], [2.0]]])
    mask = torch.tensor([[True, True, True]])

    out = reorder_valid_frames_first(x, mask)

    torch.testing.assert_close(out, x)


def test_reorder_valid_frames_first_preserves_relative_order_within_each_group():
    # Two invalid frames (0, 2) interleaved with valid ones (1, 3) --
    # relative order within each group must be preserved (stable sort).
    x = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    mask = torch.tensor([[False, True, False, True]])

    out = reorder_valid_frames_first(x, mask)

    # Valid: frames 1, 3 (in that order). Invalid: frames 0, 2 (in that order).
    expected = torch.tensor([[[1.0], [3.0], [0.0], [2.0]]])
    torch.testing.assert_close(out, expected)


def test_reorder_valid_frames_first_handles_2d_input():
    x = torch.tensor([[0.0, 1.0, 2.0]])
    mask = torch.tensor([[False, True, True]])

    out = reorder_valid_frames_first(x, mask)

    torch.testing.assert_close(out, torch.tensor([[1.0, 2.0, 0.0]]))


# ---------------------------------------------------------------------
# The actual regression: a hand-constructed leading-invalid-frame case
# where the OLD (unreordered) computation reads a CTC-infeasible window
# (forcing +inf under zero_infinity=False) and the FIXED computation
# reads the correct, feasible, low-loss window.
# ---------------------------------------------------------------------


def _build_leading_invalid_case():
    """
    4 timesteps, target = [0, 1] (2 real tokens). encoder_attention_mask
    marks frame 0 invalid (leading dropout); frames 1-3 are a clean,
    confident CTC-feasible encoding of [0, 1] on their own ("0, 1, blank").

    Frame 0 (the "invalid" placeholder) is deliberately set to confidently
    predict class 1 -- chosen so that if a buggy implementation reads
    frames [0, 1, 2] as "the real sequence" (old behavior: input_lengths=3
    starting at position 0), the resulting read sequence is "1, 0, 1",
    which has NO valid CTC alignment to target [0, 1] at all (CTC paths
    for [0,1] over 3 symbols must start by emitting 0 or blank, never 1)
    -- forcing zero_infinity=False's +inf, a sharp, unambiguous signal
    that the wrong window was used.
    """
    frame_invalid = _one_hot_logits(1)  # deliberately "wrong" placeholder content
    frame_0 = _one_hot_logits(0)
    frame_1 = _one_hot_logits(1)
    frame_blank = _one_hot_logits(_BLANK_ID)

    logits = torch.stack([frame_invalid, frame_0, frame_1, frame_blank]).unsqueeze(0)  # (1, 4, 4)
    ctc_log_probs = torch.log_softmax(logits, dim=-1)

    encoder_attention_mask = torch.tensor([[False, True, True, True]])
    target_tokens = torch.tensor([[100, 0, 1, 101]])  # [BOS=100, 0, 1, EOS=101], no padding
    return ctc_log_probs, encoder_attention_mask, target_tokens


def test_old_unreordered_window_is_ctc_infeasible_confirms_the_bug_was_real():
    """
    Reproduces the PRE-FIX computation directly (no reorder) to confirm
    the bug was real, not hypothetical: reading from position 0 for
    mask.sum()=3 positions is (near-)CTC-infeasible for this construction.
    With finite (very confident, not literally -inf) logits, PyTorch's
    zero_infinity=False doesn't necessarily return literal +inf for an
    infeasible alignment -- it returns the astronomically large but
    technically finite negative-log-likelihood of the least-bad illegal
    path. Either way the signal is unambiguous: this must be a huge loss,
    orders of magnitude larger than the fixed version's near-zero loss
    (see test_fixed_compute_combined_loss_reads_correct_feasible_window).
    """
    ctc_log_probs, encoder_attention_mask, _ = _build_leading_invalid_case()

    ctc_log_probs_t_major = ctc_log_probs.permute(1, 0, 2)  # old: no reorder
    input_lengths = encoder_attention_mask.sum(dim=1)  # tensor([3])
    target_lengths = torch.tensor([2])
    targets_flat = torch.tensor([0, 1])

    ctc_loss_fn = torch.nn.CTCLoss(blank=_BLANK_ID, zero_infinity=False)
    loss = ctc_loss_fn(ctc_log_probs_t_major, targets_flat, input_lengths, target_lengths)

    assert torch.isinf(loss) or loss.item() > 10.0, (
        "expected the OLD (unreordered) window to be CTC-infeasible/near-infeasible "
        f"(huge loss) -- if this fails, the hand-constructed case no longer "
        f"demonstrates the bug (got loss={loss.item()})"
    )


def test_fixed_compute_combined_loss_reads_correct_feasible_window():
    """
    The actual fix: compute_combined_loss (which now reorders internally)
    must produce a small, FINITE loss on the exact same input that makes
    the unreordered version blow up to +inf above.
    """
    ctc_log_probs, encoder_attention_mask, target_tokens = _build_leading_invalid_case()
    T = ctc_log_probs.shape[1]
    boundary_logits = torch.zeros(1, T)
    boundary_target = torch.zeros(1, T)

    total_loss, ctc_loss, _boundary_loss = compute_combined_loss(
        ctc_log_probs, boundary_logits, target_tokens, boundary_target,
        encoder_attention_mask,
        bos_id=100, eos_id=101, pad_id=999, blank_id=_BLANK_ID,
    )

    assert torch.isfinite(ctc_loss), (
        f"expected a finite CTC loss once the leading-invalid frame is "
        f"correctly excluded via reordering, got {ctc_loss.item()}"
    )
    # The valid frames encode a maximally-confident, unambiguous CTC path
    # for the target -- loss should be very close to 0, not just "finite".
    assert ctc_loss.item() < 0.01
