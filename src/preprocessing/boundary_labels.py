"""
boundary_labels.py

Day 31: constructs per-timestep boundary-prediction targets for C2's
joint boundary-detection + recognition loss.

BOUNDARY-LABEL DEFINITION -- REVISED FROM ORIGINAL PLAN, EVIDENCE-BASED:
The Month 2 plan assumed ASLLRP only provides a single point anchor per
sign (DOMINANT_HAND START_FRAME == END_FRAME), and planned a "point +
tolerance window" scheme accordingly (option (a)). A pre-implementation
audit (scripts/audit_sign_spans.py, 30 sampled utterances, 346 signs,
seed=31) found this assumption is wrong for the majority case:

  - 88.7% of signs have a real DOMINANT_HAND span (S != E), median
    width 7 frames, mean 9.2, max 65.
  - Only 6.9% are true point anchors (S == E) -- concentrated almost
    entirely in "Lexical Signs" (7.9% within that type); every other
    SIGN_TYPE sampled (Fingerspelled, Loan, Gestures, Classifiers,
    Number Signs) had zero point anchors.
  - 4.3% have no DOMINANT_HAND element but do have NON_DOMINANT_HAND.
  - 0% had neither -- every sign in the sample had *some* usable span.

Revised scheme: use the real span as the positive boundary region
wherever available (the large majority of signs); fall back to a
tolerance window around the single frame only for genuine point
anchors. This is evidence-backed, not a synthesized approximation --
documented here explicitly per this project's standing discipline of
stating deviations from the original plan plainly rather than
smoothing them over.

DOMINANT_HAND_FINAL_HOLD (seen for signs with a held final position,
e.g. "SOME") is deliberately excluded from the primary boundary
target -- a real, separate linguistic phenomenon, not folded in here.

CRITICAL ALIGNMENT REQUIREMENT: sign spans arrive in RAW, 1-INDEXED,
ORIGINAL VIDEO frame space. The model operates on sampled_frame_indices
-- the output of frame_sampler.select_frame_indices(), which is
trimmed, resampled to TARGET_FPS, and capped at MAX_SEQUENCE_LENGTH via
uniform downsampling. Boundary labels must be re-mapped through that
exact same transformation, not the raw frame numbers, or they will be
silently misaligned against the actual landmark sequence.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def build_boundary_targets(
    sign_spans: Sequence[Optional[tuple[int, int]]],
    sampled_frame_indices: Sequence[int],
    point_anchor_tolerance: int = 2,
) -> tuple[np.ndarray, dict]:
    """
    Parameters
    ----------
    sign_spans : per-SIGN (start_frame, end_frame), raw 1-indexed,
        original video frame space -- from
        ASLLRPUtterance.sign_spans (ASLLRPReader._extract_sign_span()).
        None for signs with no usable span at all (expected to be rare;
        0/346 in the audit sample, but not assumed impossible).
    sampled_frame_indices : the exact output of
        frame_sampler.select_frame_indices() for this utterance --
        0-indexed absolute frame numbers, already trimmed/resampled/
        capped. The boundary target aligns 1:1, position-for-position,
        with this list.
    point_anchor_tolerance : half-width, in SAMPLED-SEQUENCE positions
        (not raw video frames), of the positive region applied ONLY
        when a sign's span collapses to a single frame (start == end).
        Real spans (the majority case) use their own true width
        instead -- this parameter does not affect them.

    Returns
    -------
    boundary_target : np.ndarray, shape (len(sampled_frame_indices),),
        float32, values in {0.0, 1.0}. Aligns 1:1 with the landmarks
        tensor as a per-timestep label.
    stats : dict of real, checkable counts, for the Day 31 eyeball
        verification and ongoing monitoring:
          num_signs_total
          num_signs_missing_span      -- sign_spans entry was None
          num_signs_point_anchor      -- span collapsed to start==end,
                                          tolerance window applied
          num_signs_real_span         -- true span used directly
          num_spans_dropped_frame     -- span fell entirely outside
                                          sampled_frame_indices' range
          num_spans_partially_covered -- span's start or end (or both)
                                          fell outside the sampled
                                          range but some of it overlapped
                                          -- clipped, not dropped, but
                                          worth knowing how often this
                                          happens
    """
    T = len(sampled_frame_indices)
    boundary_target = np.zeros(T, dtype=np.float32)

    stats = {
        "num_signs_total": len(sign_spans),
        "num_signs_missing_span": 0,
        "num_signs_point_anchor": 0,
        "num_signs_real_span": 0,
        "num_spans_dropped_frame": 0,
        "num_spans_partially_covered": 0,
    }

    if T == 0:
        return boundary_target, stats

    sampled_arr = np.asarray(sampled_frame_indices)
    lo_bound, hi_bound = sampled_arr.min(), sampled_arr.max()

    for span in sign_spans:
        if span is None:
            stats["num_signs_missing_span"] += 1
            continue

        start_raw, end_raw = span
        # Convert to 0-indexed to match sampled_frame_indices' space.
        start_0 = start_raw - 1
        end_0 = end_raw - 1

        if end_0 < lo_bound or start_0 > hi_bound:
            # Entire span falls outside what was actually sampled --
            # e.g. trimmed away by frame_start/frame_end. Real
            # data-quality signal, not silently ignored.
            stats["num_spans_dropped_frame"] += 1
            continue

        if start_0 < lo_bound or end_0 > hi_bound:
            stats["num_spans_partially_covered"] += 1

        if start_raw == end_raw:
            # Genuine point anchor -- snap to nearest surviving sampled
            # position, apply the tolerance window.
            stats["num_signs_point_anchor"] += 1
            nearest_pos = int(np.argmin(np.abs(sampled_arr - start_0)))
            lo = max(0, nearest_pos - point_anchor_tolerance)
            hi = min(T, nearest_pos + point_anchor_tolerance + 1)
            boundary_target[lo:hi] = 1.0
        else:
            # Real span -- mark every sampled position whose original
            # frame number falls within [start_0, end_0].
            stats["num_signs_real_span"] += 1
            mask = (sampled_arr >= start_0) & (sampled_arr <= end_0)
            boundary_target[mask] = 1.0

    return boundary_target, stats