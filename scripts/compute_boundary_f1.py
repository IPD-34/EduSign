"""
scripts/compute_boundary_f1.py

Day 50: boundary-F1 computation for Week 8's held-out (test-split)
evaluation -- how well the boundary head's predicted positive regions
line up with the REAL sign-span-derived ground truth. Built and
hand-validated today; NOT run against the full test split today -- that
is Day 51's job (see scripts/test_day50_wer_and_boundary_f1.py for the
hand-validation this script's correctness rests on).

--- CHECKPOINT OF RECORD (updated -- see docs/checkpoint_of_record_decision.md) ---
checkpoints/month2_asllrp_gated/step_0008000.pt (step 8000, gate ACTIVE
in the final architecture -- see scripts/compute_wer_beam_search.py's
header for the full switch reasoning, not repeated here; supersedes
best_val_gated.pt, which is NOT deleted and stays on disk as historical
record). boundary_logits come from forward_ctc_boundary_and_gate(), the
same shared-encoder-pass path.

--- GROUND TRUTH: reused, not reimplemented ---
build_boundary_targets() (src/preprocessing/boundary_labels.py) is
called directly with the sample's REAL full sign_spans list and real
frame_indices -- the exact same call dataset.py makes to build the
training target itself (Day 31's upgrade: real per-sign spans, not the
originally-planned point-anchor scheme). This script does not
reimplement or approximate that alignment.

--- MATCH CRITERION DECISION: IoU >= 0.5, INTERVAL-LEVEL ---
"Boundary F1" is not self-defining: build_boundary_targets() returns a
per-timestep BINARY array (1 = inside some real sign span, 0 =
background/between signs), not a set of discrete events. To score it
as detection-style F1 (what "boundary F1" conventionally means in
segmentation/boundary-detection literature), both the ground-truth and
the predicted binary arrays are first collapsed into their maximal
contiguous positive RUNS via plain run-length encoding (e.g. "sign span
#3 occupies timesteps [41,47]") -- a lossless, deterministic transform
of the same array, not a new ground-truth definition.

A predicted interval and a ground-truth interval are then a MATCH if
their IoU (intersection-over-union, in timestep counts) is >=
IOU_THRESHOLD (0.5 by default). Chosen over an arbitrary tolerance-
window width specifically because IoU>=0.5 is a single, standard,
widely-used convention for interval/segment detection F1 (the same
default used broadly in temporal-detection literature) that needs no
extra frame-count parameter to separately justify, and it directly
rewards "predicted region substantially overlaps the true span" --
exactly what matters now that spans are real intervals (Day 31), not
point events.

Matching is one-to-one and greedy: highest-IoU candidate pairs are
assigned first (standard practice for detection-style precision/recall
scoring), so a predicted interval can match at most one ground-truth
interval and vice versa.
  TP = matched pairs
  FP = predicted intervals with no match
  FN = ground-truth intervals with no match
  Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R)

Binary prediction threshold: sigmoid(boundary_logits) >= 0.5 (i.e.
boundary_logits >= 0), matching training's own BCEWithLogitsLoss
decision convention (src/models/combined_loss.py) exactly -- not a new
threshold invented for eval.

--- SECOND CRITERION, ADDITIVE: CENTER-TOLERANCE MATCH ---
Added to check how much of the ~0.10-0.15 IoU>=0.5 F1 number is metric
strictness vs. genuine failure, NOT to replace IoU -- both criteria are
fully intact and independently runnable (match_intervals/
boundary_f1_for_sample, the IoU path, are UNTOUCHED by this addition).

Real sign windows are often only 5-9 timesteps wide (point-anchor signs
are exactly 5, per Day 31's tolerance=2 window construction: 2*2+1=5),
so a prediction off by 2-3 timesteps can fail IoU>=0.5 outright even
when it is clearly in roughly the right place -- IoU penalizes width/
position error together and non-linearly for narrow intervals, which is
exactly the wrong sensitivity for asking "did the model find approximately
the right region" as a separate question from "did it also get the width
right."

CENTER_TOLERANCE_TIMESTEPS = 3: a predicted interval and a ground-truth
interval MATCH if their CENTERS (interval midpoint, (start+end)/2) are
within this many timesteps of each other -- independent of either
interval's width. Chosen at 3, not an arbitrary round number: it's
one more than the Day 31 point-anchor tolerance (2), so a point-anchor
sign's own true window (radius 2 from its center) sits inside this
band, and it's on the order of half the median real-span width
(observed ~7-8 timesteps in this project's own span audits) -- loose
enough to separate "roughly the right place" from "nowhere near it,"
tight enough that it doesn't degenerate into rewarding any prediction
that merely overlaps the same utterance.

Matching is one-to-one and greedy, same discipline as the IoU path:
closest-center pairs assigned first (analogous to highest-IoU-first),
so a predicted interval can match at most one ground-truth interval and
vice versa. TP/FP/FN/precision/recall/F1 defined identically to the
IoU path, just with "matched" redefined by this criterion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, get_device
from src.models.dataset import SignLanguageDataset
from src.preprocessing.sample_writer import load_sample
from src.preprocessing.boundary_labels import build_boundary_targets

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
TEST_NPZ_DIR = "data/processed/asllrp/test"
CHECKPOINT_PATH = "checkpoints/month2_asllrp_gated/step_0008000.pt"  # checkpoint of record, gate ACTIVE -- see docs/checkpoint_of_record_decision.md

IOU_THRESHOLD = 0.5
BOUNDARY_PROB_THRESHOLD = 0.5
CENTER_TOLERANCE_TIMESTEPS = 3


def extract_intervals(binary_array):
    """binary_array: 1D sequence of {0,1} (or bool). Returns an ordered
    list of (start, end) INCLUSIVE timestep ranges, one per maximal
    contiguous run of positives. Pure run-length encoding -- no
    reinterpretation of the ground truth."""
    intervals = []
    start = None
    for i, v in enumerate(binary_array):
        if v and start is None:
            start = i
        elif not v and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(binary_array) - 1))
    return intervals


def interval_iou(a, b):
    """a, b: (start, end) inclusive timestep ranges. IoU in timestep-count terms."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def match_intervals(gt_intervals, pred_intervals, iou_threshold=IOU_THRESHOLD):
    """Greedy one-to-one IoU matching, highest IoU assigned first.
    Returns (tp_pairs, fp_pred_indices, fn_gt_indices)."""
    candidates = []
    for gi, g in enumerate(gt_intervals):
        for pi, p in enumerate(pred_intervals):
            iou = interval_iou(g, p)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda c: -c[0])

    matched_gt, matched_pred = set(), set()
    tp_pairs = []
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        tp_pairs.append((gi, pi, iou))

    fp = [pi for pi in range(len(pred_intervals)) if pi not in matched_pred]
    fn = [gi for gi in range(len(gt_intervals)) if gi not in matched_gt]
    return tp_pairs, fp, fn


def boundary_f1_for_sample(gt_binary, pred_binary, iou_threshold=IOU_THRESHOLD):
    gt_intervals = extract_intervals(gt_binary)
    pred_intervals = extract_intervals(pred_binary)
    tp_pairs, fp, fn = match_intervals(gt_intervals, pred_intervals, iou_threshold)
    tp = len(tp_pairs)
    precision = tp / (tp + len(fp)) if (tp + len(fp)) > 0 else 0.0
    recall = tp / (tp + len(fn)) if (tp + len(fn)) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "gt_intervals": gt_intervals,
        "pred_intervals": pred_intervals,
        "tp": tp, "fp": len(fp), "fn": len(fn),
        "precision": precision, "recall": recall, "f1": f1,
        "matches": tp_pairs,
    }


def interval_center(interval):
    """interval: (start, end) inclusive. Returns the midpoint timestep (float)."""
    start, end = interval
    return (start + end) / 2.0


def match_intervals_center_tolerance(gt_intervals, pred_intervals, tolerance=CENTER_TOLERANCE_TIMESTEPS):
    """Second, additive criterion -- see module docstring. Greedy
    one-to-one matching, CLOSEST-center pairs assigned first (the
    center-tolerance analog of IoU's highest-IoU-first). Returns
    (tp_pairs, fp_pred_indices, fn_gt_indices), same shape as
    match_intervals()."""
    candidates = []
    for gi, g in enumerate(gt_intervals):
        for pi, p in enumerate(pred_intervals):
            dist = abs(interval_center(g) - interval_center(p))
            if dist <= tolerance:
                candidates.append((dist, gi, pi))
    candidates.sort(key=lambda c: c[0])

    matched_gt, matched_pred = set(), set()
    tp_pairs = []
    for dist, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        tp_pairs.append((gi, pi, dist))

    fp = [pi for pi in range(len(pred_intervals)) if pi not in matched_pred]
    fn = [gi for gi in range(len(gt_intervals)) if gi not in matched_gt]
    return tp_pairs, fp, fn


def boundary_f1_for_sample_center_tolerance(gt_binary, pred_binary, tolerance=CENTER_TOLERANCE_TIMESTEPS):
    """Center-tolerance analog of boundary_f1_for_sample() -- same shape
    of return value, IoU path completely untouched by this function."""
    gt_intervals = extract_intervals(gt_binary)
    pred_intervals = extract_intervals(pred_binary)
    tp_pairs, fp, fn = match_intervals_center_tolerance(gt_intervals, pred_intervals, tolerance)
    tp = len(tp_pairs)
    precision = tp / (tp + len(fp)) if (tp + len(fp)) > 0 else 0.0
    recall = tp / (tp + len(fn)) if (tp + len(fn)) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "gt_intervals": gt_intervals,
        "pred_intervals": pred_intervals,
        "tp": tp, "fp": len(fp), "fn": len(fn),
        "precision": precision, "recall": recall, "f1": f1,
        "matches": tp_pairs,
    }


def compute_sample_boundary_arrays(model, item, meta, device, prob_threshold=BOUNDARY_PROB_THRESHOLD,
                                    use_gate=True):
    """Returns (gt_binary, pred_binary) -- both 1D int arrays, length T.
    use_gate=False (Day 52 gate-decision recheck addition): boundary_logits
    come from forward_ctc_and_boundary() instead (gate path bypassed), for
    scoring best_val.pt as it was actually trained."""
    landmarks = item["landmarks"].unsqueeze(0).to(device)
    T = landmarks.shape[1]
    encoder_mask = torch.ones(1, T, dtype=torch.bool, device=device)
    with torch.no_grad():
        if use_gate:
            _ctc_lp, boundary_logits, _gate_logits = model.forward_ctc_boundary_and_gate(landmarks, encoder_mask)
        else:
            _ctc_lp, boundary_logits = model.forward_ctc_and_boundary(landmarks, encoder_mask)
    pred_prob = torch.sigmoid(boundary_logits[0]).cpu().numpy()
    pred_binary = (pred_prob >= prob_threshold).astype(int)

    sign_spans_as_tuples = [tuple(s) if s is not None else None for s in meta.sign_spans]
    gt_binary, stats = build_boundary_targets(
        sign_spans=sign_spans_as_tuples, sampled_frame_indices=meta.frame_indices,
    )
    return gt_binary.astype(int), pred_binary, stats


def score_sample(model, item, meta, device, iou_threshold=IOU_THRESHOLD,
                  prob_threshold=BOUNDARY_PROB_THRESHOLD, use_gate=True):
    gt_binary, pred_binary, span_stats = compute_sample_boundary_arrays(
        model, item, meta, device, prob_threshold, use_gate=use_gate
    )
    result = boundary_f1_for_sample(gt_binary, pred_binary, iou_threshold)
    result["video_id"] = meta.video_id
    result["span_stats"] = span_stats
    return result


def run_full_test_split(iou_threshold=IOU_THRESHOLD, prob_threshold=BOUNDARY_PROB_THRESHOLD,
                         checkpoint_path=CHECKPOINT_PATH, npz_dir=TEST_NPZ_DIR,
                         use_gate=True, indices=None):
    """Day 51's job (defaults reproduce it exactly). Day 52 additions
    (npz_dir, use_gate, indices) -- see compute_wer_beam_search.py's
    run_full_test_split() docstring, identical rationale."""
    device = get_device()
    import json
    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)

    dataset = SignLanguageDataset(
        npz_dir=npz_dir, vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    model = SignTransformer(
        vocab_size=len(decoder_vocab), ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
    ).to(device)
    prefixes = ("encoder.", "ctc_head.", "boundary_head.", "bimanual_gate_head.") if use_gate \
        else ("encoder.", "ctc_head.", "boundary_head.")
    loaded, skipped, new_only = load_checkpoint_partial(model, checkpoint_path, prefixes_to_load=prefixes)
    print(f"Checkpoint load: {len(loaded)} loaded, {len(skipped)} skipped, {len(new_only)} new-only")
    model.eval()

    idx_list = indices if indices is not None else range(len(dataset))
    total_tp = total_fp = total_fn = 0
    per_sample = []
    for idx in idx_list:
        item = dataset[idx]
        meta = load_sample(dataset.npz_paths[idx]).metadata
        r = score_sample(model, item, meta, device, iou_threshold, prob_threshold, use_gate=use_gate)
        total_tp += r["tp"]; total_fp += r["fp"]; total_fn += r["fn"]
        per_sample.append(r)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0.0)
    return {
        "n_samples": len(per_sample),
        "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
        "micro_precision": micro_precision, "micro_recall": micro_recall, "micro_f1": micro_f1,
        "per_sample": per_sample,
    }


def run_full_test_split_dual_criteria(iou_threshold=IOU_THRESHOLD, center_tolerance=CENTER_TOLERANCE_TIMESTEPS,
                                       prob_threshold=BOUNDARY_PROB_THRESHOLD, checkpoint_path=CHECKPOINT_PATH,
                                       npz_dir=TEST_NPZ_DIR, use_gate=True, indices=None):
    """Scores BOTH criteria (IoU>=iou_threshold, AND center-tolerance<=
    center_tolerance) per sample from the SAME gt_binary/pred_binary pair
    -- one model forward pass per sample, not two -- so this is a strict
    superset of run_full_test_split(), not a separate, possibly-drifting
    re-implementation. run_full_test_split() itself is completely
    untouched and still gives byte-identical IoU-only results."""
    device = get_device()
    import json
    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)

    dataset = SignLanguageDataset(
        npz_dir=npz_dir, vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    model = SignTransformer(
        vocab_size=len(decoder_vocab), ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
    ).to(device)
    prefixes = ("encoder.", "ctc_head.", "boundary_head.", "bimanual_gate_head.") if use_gate \
        else ("encoder.", "ctc_head.", "boundary_head.")
    loaded, skipped, new_only = load_checkpoint_partial(model, checkpoint_path, prefixes_to_load=prefixes)
    print(f"Checkpoint load: {len(loaded)} loaded, {len(skipped)} skipped, {len(new_only)} new-only")
    model.eval()

    idx_list = indices if indices is not None else range(len(dataset))
    iou_tp = iou_fp = iou_fn = 0
    ct_tp = ct_fp = ct_fn = 0
    per_sample = []
    for idx in idx_list:
        item = dataset[idx]
        meta = load_sample(dataset.npz_paths[idx]).metadata
        gt_binary, pred_binary, span_stats = compute_sample_boundary_arrays(
            model, item, meta, device, prob_threshold, use_gate=use_gate
        )
        iou_result = boundary_f1_for_sample(gt_binary, pred_binary, iou_threshold)
        ct_result = boundary_f1_for_sample_center_tolerance(gt_binary, pred_binary, center_tolerance)

        iou_tp += iou_result["tp"]; iou_fp += iou_result["fp"]; iou_fn += iou_result["fn"]
        ct_tp += ct_result["tp"]; ct_fp += ct_result["fp"]; ct_fn += ct_result["fn"]

        per_sample.append({
            "video_id": meta.video_id,
            "gt_intervals": iou_result["gt_intervals"], "pred_intervals": iou_result["pred_intervals"],
            "iou": {"tp": iou_result["tp"], "fp": iou_result["fp"], "fn": iou_result["fn"],
                    "precision": iou_result["precision"], "recall": iou_result["recall"], "f1": iou_result["f1"]},
            "center_tolerance": {"tp": ct_result["tp"], "fp": ct_result["fp"], "fn": ct_result["fn"],
                                  "precision": ct_result["precision"], "recall": ct_result["recall"], "f1": ct_result["f1"]},
        })

    def _micro(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    iou_p, iou_r, iou_f1 = _micro(iou_tp, iou_fp, iou_fn)
    ct_p, ct_r, ct_f1 = _micro(ct_tp, ct_fp, ct_fn)

    return {
        "n_samples": len(per_sample),
        "iou": {"threshold": iou_threshold, "total_tp": iou_tp, "total_fp": iou_fp, "total_fn": iou_fn,
                "micro_precision": iou_p, "micro_recall": iou_r, "micro_f1": iou_f1},
        "center_tolerance": {"tolerance_timesteps": center_tolerance, "total_tp": ct_tp, "total_fp": ct_fp, "total_fn": ct_fn,
                              "micro_precision": ct_p, "micro_recall": ct_r, "micro_f1": ct_f1},
        "per_sample": per_sample,
    }


def main():
    """Day 51: the real held-out run, full 122-sample test split, gate
    active, checkpoint of record (step_0008000.pt, step 8000 -- see
    docs/checkpoint_of_record_decision.md; supersedes best_val_gated.pt).
    The Day-50 SystemExit guard is intentionally removed here -- see Day 51's
    structural-bug checklist report before trusting these numbers."""
    import json as _json

    print(f"=== Day 51: boundary-F1, full test split, checkpoint={CHECKPOINT_PATH} ===")
    print(f"IOU_THRESHOLD={IOU_THRESHOLD} BOUNDARY_PROB_THRESHOLD={BOUNDARY_PROB_THRESHOLD}\n")

    results = run_full_test_split()

    print(f"\nn_samples={results['n_samples']}")
    print(f"total_tp={results['total_tp']}  total_fp={results['total_fp']}  total_fn={results['total_fn']}")
    print(f"MICRO Precision = {results['micro_precision']:.4f}")
    print(f"MICRO Recall    = {results['micro_recall']:.4f}")
    print(f"MICRO F1        = {results['micro_f1']:.4f}")

    per_sample_f1 = [r["f1"] for r in results["per_sample"]]
    mean_per_sample_f1 = sum(per_sample_f1) / len(per_sample_f1) if per_sample_f1 else 0.0
    n_zero_f1 = sum(1 for f in per_sample_f1 if f == 0.0)
    n_perfect_f1 = sum(1 for f in per_sample_f1 if f == 1.0)
    print(f"\nmean per-sample F1 (macro-average) = {mean_per_sample_f1:.4f}")
    print(f"samples with F1==0.0: {n_zero_f1}/{len(per_sample_f1)}")
    print(f"samples with F1==1.0: {n_perfect_f1}/{len(per_sample_f1)}")

    out_path = Path("logs") / "day51_boundary_f1_full_test_split.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump({
            "checkpoint": CHECKPOINT_PATH, "checkpoint_step": 2000, "split": "test",
            "n_test_files": 122, "iou_threshold": IOU_THRESHOLD, "prob_threshold": BOUNDARY_PROB_THRESHOLD,
            "micro_precision": results["micro_precision"], "micro_recall": results["micro_recall"],
            "micro_f1": results["micro_f1"], "mean_per_sample_f1": mean_per_sample_f1,
            "total_tp": results["total_tp"], "total_fp": results["total_fp"], "total_fn": results["total_fn"],
            "n_samples": results["n_samples"], "n_zero_f1": n_zero_f1, "n_perfect_f1": n_perfect_f1,
            "per_sample": [{k: v for k, v in r.items() if k != "span_stats"} for r in results["per_sample"]],
        }, f, indent=2)
    print(f"\nFull per-sample results saved to {out_path}")


if __name__ == "__main__":
    main()
