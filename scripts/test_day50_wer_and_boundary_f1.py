"""
scripts/test_day50_wer_and_boundary_f1.py

Day 50 hand-validation of scripts/compute_wer_beam_search.py and
scripts/compute_boundary_f1.py -- same bar as Day 10's dataloader
validation (exact match against an independently hand-computed answer,
not "looks about right").

Two-part validation, same split used for the C3 ablation's own
discipline (validate the algorithm on a controlled case, then validate
the scoring on real data):

  PART A -- synthetic, fully hand-traceable cases (tiny vocab, tiny T)
  that prove the DECODE / MATCHING ALGORITHMS themselves are correct.
  Full CTC beam search over the REAL ~2223-symbol vocab cannot be
  hand-traced at all (no human traces which of 2223 per-timestep
  probabilities dominate by eye) regardless of utterance length -- so
  algorithm correctness is proven here, on cases small enough to
  enumerate every path by hand, not on real model output.

  PART B -- two real ASLLRP test-split samples (asllrp_63579,
  asllrp_97998228 -- the two shortest test-split utterances, picked so
  the resulting numbers stay hand-checkable), run through the real
  checkpoint of record (best_val_gated.pt, gate active), with the
  EDIT-DISTANCE and IOU-MATCHING arithmetic on that real output
  independently hand-computed and compared for exact match. This
  proves the SCORING/aggregation logic is correct end-to-end on real
  data, given the beam search's own output as an (already-proven-
  correct, per Part A) black box.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, get_device
from src.models.dataset import SignLanguageDataset
from src.preprocessing.sample_writer import load_sample
from src.preprocessing.boundary_labels import build_boundary_targets

from scripts.compute_wer_beam_search import ctc_prefix_beam_search, edit_distance, CHECKPOINT_PATH
from scripts.compute_boundary_f1 import (
    extract_intervals, interval_iou, match_intervals, boundary_f1_for_sample,
    compute_sample_boundary_arrays,
)

FAILURES = []


def check(name, got, expected):
    ok = got == expected
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got={got!r} expected={expected!r}")
    if not ok:
        FAILURES.append(name)
    return ok


def check_close(name, got, expected, tol=1e-6):
    ok = abs(got - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got={got!r} expected={expected!r} (tol={tol})")
    if not ok:
        FAILURES.append(name)
    return ok


# =====================================================================
# PART A1 -- CTC beam search, hand-traced Case A (merge-and-sum ranking)
# =====================================================================
def part_a1_beam_search_case_a():
    print("\n=== Part A1: ctc_prefix_beam_search, synthetic Case A ===")
    print("T=2, vocab={sym0, sym1, blank=2}. P(t0)=[0.6,0.1,0.3], P(t1)=[0.5,0.2,0.3].")
    print("Hand enumeration of all 9 raw paths, aggregated by CTC-collapsed output:")
    print('  "0"    <- (0,0)=0.30 + (0,blank)=0.18 + (blank,0)=0.15 = 0.63')
    print('  "0,1"  <- (0,1)=0.12')
    print('  "1"    <- (1,1)=0.02 + (1,blank)=0.03 + (blank,1)=0.06 = 0.11')
    print('  ""     <- (blank,blank)=0.09')
    print('  "1,0"  <- (1,0)=0.05')
    print("  (sums to 1.00 exactly -- sanity check on the hand enumeration itself)")
    total = 0.63 + 0.12 + 0.11 + 0.09 + 0.05
    check_close("Case A hand-enumeration sums to 1.0", total, 1.0)
    print('  Winner: "0" at p=0.63 (beats "0,1" at 0.12) -- expected decode = [0]')

    log_probs = torch.log(torch.tensor([
        [0.6, 0.1, 0.3],
        [0.5, 0.2, 0.3],
    ]))
    decoded = ctc_prefix_beam_search(log_probs, blank_id=2, beam_width=10, expand_top_k=3)
    check("Case A decoded_ids", decoded, [0])


# =====================================================================
# PART A2 -- CTC beam search, hand-traced Case B (collapse-rule test)
# =====================================================================
def part_a2_beam_search_case_b():
    print("\n=== Part A2: ctc_prefix_beam_search, synthetic Case B (collapse rule) ===")
    print("T=3, vocab={a=0, blank=1}. P(a)=0.9, P(blank)=0.1 at every timestep.")
    print("Tests the core CTC subtlety: same symbol repeated WITHOUT an intervening")
    print("blank does NOT extend the sequence (still one occurrence of 'a'); a blank")
    print("in between WOULD make it two occurrences ('a,a').")
    print("Hand enumeration of all 8 raw paths, aggregated by CTC-collapsed output:")
    print('  "a"    <- (a,a,a)=0.729 + (a,a,bl)=0.081 + (a,bl,bl)=0.009')
    print('           + (bl,a,a)=0.081 + (bl,a,bl)=0.009 + (bl,bl,a)=0.009 = 0.918')
    print('  "a,a"  <- (a,bl,a)=0.081   [blank SEPARATES two real occurrences of a]')
    print('  ""     <- (bl,bl,bl)=0.001')
    total = 0.918 + 0.081 + 0.001
    check_close("Case B hand-enumeration sums to 1.0", total, 1.0)
    print('  Winner: "a" at p=0.918 (beats "a,a" at 0.081) -- expected decode = [0]')

    log_probs = torch.log(torch.tensor([
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1],
    ]))
    decoded = ctc_prefix_beam_search(log_probs, blank_id=1, beam_width=10, expand_top_k=2)
    check("Case B decoded_ids", decoded, [0])


# =====================================================================
# PART A3 -- boundary-F1 matching engine, hand-traced synthetic case
# =====================================================================
def part_a3_boundary_f1_synthetic():
    print("\n=== Part A3: boundary_f1_for_sample, synthetic case ===")
    gt =   [0,1,1,1,0,0,1,1,0,0,1,1,1,1,0]
    pred = [0,1,1,1,1,0,0,0,1,1,1,0,1,1,0]
    print(f"  gt_binary:   {gt}")
    print(f"  pred_binary: {pred}")
    print("  Hand-extracted GT intervals:   G1=(1,3) G2=(6,7) G3=(10,13)")
    print("  Hand-extracted pred intervals: P1=(1,4) P2=(8,10) P3=(12,13)")
    check("GT intervals", extract_intervals(gt), [(1, 3), (6, 7), (10, 13)])
    check("Pred intervals", extract_intervals(pred), [(1, 4), (8, 10), (12, 13)])

    print("  Hand IoU: IoU(G1,P1)=3/4=0.75 (match, >=0.5)")
    print("            IoU(G3,P3)=2/4=0.5  (match, >=0.5, boundary case)")
    print("            IoU(G3,P2)=1/6=0.167 (below threshold)")
    print("            all other pairs = 0.0 (no overlap)")
    check_close("IoU(G1,P1)", interval_iou((1, 3), (1, 4)), 0.75)
    check_close("IoU(G3,P3)", interval_iou((10, 13), (12, 13)), 0.5)
    check_close("IoU(G3,P2)", interval_iou((10, 13), (8, 10)), 1 / 6)

    print("  Hand greedy matching (highest IoU first): G1-P1 (0.75), then G3-P3 (0.5).")
    print("  G2 unmatched -> FN. P2 unmatched -> FP.")
    print("  => TP=2, FP=1, FN=1, Precision=2/3, Recall=2/3, F1=2/3")

    result = boundary_f1_for_sample(gt, pred, iou_threshold=0.5)
    check("TP", result["tp"], 2)
    check("FP", result["fp"], 1)
    check("FN", result["fn"], 1)
    check_close("Precision", result["precision"], 2 / 3)
    check_close("Recall", result["recall"], 2 / 3)
    check_close("F1", result["f1"], 2 / 3)


# =====================================================================
# Shared setup for Part B (real samples)
# =====================================================================
def _load_model_and_dataset():
    device = get_device()
    with open("data/vocab_merged.json") as f:
        vocab_merged = json.load(f)
    with open("data/vocab.json") as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)
    blank_id = ctc_vocab_size

    dataset = SignLanguageDataset(
        npz_dir="data/processed/asllrp/test", vocab_path="data/vocab_merged.json",
        asllrp_token_to_id_path="data/asllrp_token_to_id.json",
    )
    model = SignTransformer(
        vocab_size=len(decoder_vocab), ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
    ).to(device)
    loaded, skipped, new_only = load_checkpoint_partial(
        model, CHECKPOINT_PATH,
        prefixes_to_load=("encoder.", "ctc_head.", "boundary_head.", "bimanual_gate_head."),
    )
    print(f"  Checkpoint load: {len(loaded)} loaded, {len(skipped)} skipped, {len(new_only)} new-only")
    model.eval()
    return model, dataset, device, blank_id


# =====================================================================
# PART B1 -- WER, real samples, hand-computed edit distance
# =====================================================================
# (ref, hyp, hand_edits) -- edit distance hand-derived via Levenshtein
# table, independent of the script's own edit_distance() implementation.
REAL_WER_HAND_CHECKS = {
    "asllrp_63579": {
        "true_ids": [1853, 1917],
        "hand_edits": 2,   # ref len 2, hyp [] -> 2 deletions
    },
    "asllrp_97998228": {
        "true_ids": [2105, 1522, 3],
        "hand_edits": 3,   # ref len 3, hyp [1853] (len 1, no overlap with ref)
                           # hand Levenshtein table (ref=[2105,1522,3], hyp=[1853]):
                           #   dp[0]=[0,1]; dp[1]=[1,1]; dp[2]=[2,2]; dp[3]=[3,3]
                           #   -> dp[3][1] = 3
    },
}


def part_b1_wer_real_samples():
    print("\n=== Part B1: WER on real test-split samples ===")
    model, dataset, device, blank_id = _load_model_and_dataset()
    video_id_to_idx = {load_sample(p).metadata.video_id: i for i, p in enumerate(dataset.npz_paths)}

    for vid, expected in REAL_WER_HAND_CHECKS.items():
        idx = video_id_to_idx[vid]
        item = dataset[idx]
        landmarks = item["landmarks"].unsqueeze(0).to(device)
        T = landmarks.shape[1]
        encoder_mask = torch.ones(1, T, dtype=torch.bool, device=device)
        true_tokens = item["target_tokens"].tolist()
        true_ids = true_tokens[1:-1]
        check(f"{vid}: true_ids", true_ids, expected["true_ids"])

        with torch.no_grad():
            ctc_log_probs, _bl, _gl = model.forward_ctc_boundary_and_gate(landmarks, encoder_mask)
        log_probs = ctc_log_probs[0].cpu()

        decoded_ids = ctc_prefix_beam_search(log_probs, blank_id, beam_width=10, expand_top_k=20)
        print(f"  {vid}: decoded_ids (beam_width=10, expand_top_k=20) = {decoded_ids}")

        script_edits = edit_distance(true_ids, decoded_ids)
        # Independent hand re-derivation, not calling the module under test:
        hand_edits = _hand_levenshtein(true_ids, decoded_ids)
        check(f"{vid}: script edit_distance == pre-registered hand value", script_edits, expected["hand_edits"])
        check(f"{vid}: script edit_distance == independently re-derived hand value", script_edits, hand_edits)
        check_close(f"{vid}: sample_wer", script_edits / max(len(true_ids), 1),
                    expected["hand_edits"] / max(len(expected["true_ids"]), 1))


def _hand_levenshtein(ref, hyp):
    """Independent re-implementation (deliberately separate code from
    compute_wer_beam_search.edit_distance) used ONLY to cross-check that
    function's output on real data -- not imported by the production
    script."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[n][m]


# =====================================================================
# PART B2 -- boundary-F1, real samples, hand-computed IoU matching
# =====================================================================
def part_b2_boundary_f1_real_samples():
    print("\n=== Part B2: boundary-F1 on real test-split samples ===")
    model, dataset, device, _blank_id = _load_model_and_dataset()
    video_id_to_idx = {load_sample(p).metadata.video_id: i for i, p in enumerate(dataset.npz_paths)}

    # ---- asllrp_63579 ----
    vid = "asllrp_63579"
    idx = video_id_to_idx[vid]
    item = dataset[idx]
    meta = load_sample(dataset.npz_paths[idx]).metadata
    gt_binary, pred_binary, stats = compute_sample_boundary_arrays(model, item, meta, device)

    print(f"\n  {vid}: sign_spans={meta.sign_spans}, frame_indices[0..{len(meta.frame_indices)-1}]="
          f"{meta.frame_indices[0]}..{meta.frame_indices[-1]} (contiguous, so timestep i <-> raw frame {meta.frame_indices[0]}+i)")
    print("  Hand-derived GT intervals directly from sign_spans (bypassing build_boundary_targets):")
    print(f"    sign1 [382,383] 1-idx -> 0-idx [381,382] -> frame_indices offset "
          f"[{381-meta.frame_indices[0]},{382-meta.frame_indices[0]}] = (8,9)")
    print(f"    sign2 [390,408] 1-idx -> 0-idx [389,407] -> frame_indices offset "
          f"[{389-meta.frame_indices[0]},{407-meta.frame_indices[0]}] = (16,34)")
    check("GT intervals (hand-derived from raw spans) == build_boundary_targets output",
          extract_intervals(gt_binary), [(8, 9), (16, 34)])

    print("  Hand-extracted pred intervals from pred_binary (threshold>=0.5):")
    pred_intervals_hand = [(14, 29), (32, 37), (39, 40), (42, 42)]
    print(f"    {pred_intervals_hand}")
    check("Pred intervals", extract_intervals(pred_binary), pred_intervals_hand)

    print("  Hand IoU: IoU(G1=(8,9), any pred) = 0 for all (no overlap) -> G1 is FN")
    print("            IoU(G2=(16,34), P1=(14,29)) = 14/21 = 0.6667 -> ONLY candidate >= 0.5")
    print("            IoU(G2, P2=(32,37)) = 3/22 = 0.1364 (below threshold)")
    print("  => TP=1 (G2-P1), FP=3 (P2,P3,P4 unmatched), FN=1 (G1 unmatched)")
    print("     Precision = 1/4 = 0.25, Recall = 1/2 = 0.5, F1 = 2*.25*.5/.75 = 0.3333")

    result = boundary_f1_for_sample(gt_binary, pred_binary)
    check(f"{vid}: TP", result["tp"], 1)
    check(f"{vid}: FP", result["fp"], 3)
    check(f"{vid}: FN", result["fn"], 1)
    check_close(f"{vid}: Precision", result["precision"], 0.25)
    check_close(f"{vid}: Recall", result["recall"], 0.5)
    check_close(f"{vid}: F1", result["f1"], 1/3)

    # ---- asllrp_97998228 ----
    vid = "asllrp_97998228"
    idx = video_id_to_idx[vid]
    item = dataset[idx]
    meta = load_sample(dataset.npz_paths[idx]).metadata
    gt_binary, pred_binary, stats = compute_sample_boundary_arrays(model, item, meta, device)

    print(f"\n  {vid}: sign_spans={meta.sign_spans}, frame_indices[0]={meta.frame_indices[0]} (contiguous)")
    print("  Hand-derived GT intervals directly from sign_spans:")
    print(f"    sign1 [3842,3847] -> 0-idx [3841,3846] -> offset (13,18)")
    print(f"    sign2 [3851,3859] -> 0-idx [3850,3858] -> offset (22,30)")
    print(f"    sign3 [3862,3871] -> 0-idx [3861,3870] -> offset (33,42)")
    check("GT intervals (hand-derived from raw spans) == build_boundary_targets output",
          extract_intervals(gt_binary), [(13, 18), (22, 30), (33, 42)])

    print("  Hand-extracted pred intervals: model predicts ONE big merged blob covering all 3 signs:")
    pred_intervals_hand = [(12, 50)]
    print(f"    {pred_intervals_hand}")
    check("Pred intervals", extract_intervals(pred_binary), pred_intervals_hand)

    print("  Hand IoU (each small GT interval fully inside the one big pred interval,")
    print("  but diluted by its size -> all below 0.5 threshold):")
    print("    IoU(G1=(13,18)[6], P1=(12,50)[39]) = 6/39 = 0.1538")
    print("    IoU(G2=(22,30)[9], P1) = 9/39 = 0.2308")
    print("    IoU(G3=(33,42)[10], P1) = 10/39 = 0.2564")
    print("  => no candidate reaches 0.5 -> TP=0, FP=1 (P1 unmatched), FN=3 (all GT unmatched)")
    print("     Precision=0.0, Recall=0.0, F1=0.0")
    print("  (Real, informative finding, not a bug: the boundary head at step 2000 detects THAT")
    print("   signing is happening in roughly the right broad region, but does not yet separate")
    print("   the 3 individual signs -- a strict interval-level IoU>=0.5 criterion correctly")
    print("   scores that as a miss. Reported here as an observation about what the hand-check")
    print("   surfaced, not a Day 51 conclusion.)")

    result = boundary_f1_for_sample(gt_binary, pred_binary)
    check(f"{vid}: TP", result["tp"], 0)
    check(f"{vid}: FP", result["fp"], 1)
    check(f"{vid}: FN", result["fn"], 3)
    check_close(f"{vid}: Precision", result["precision"], 0.0)
    check_close(f"{vid}: Recall", result["recall"], 0.0)
    check_close(f"{vid}: F1", result["f1"], 0.0)


def main():
    part_a1_beam_search_case_a()
    part_a2_beam_search_case_b()
    part_a3_boundary_f1_synthetic()
    part_b1_wer_real_samples()
    part_b2_boundary_f1_real_samples()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    else:
        print("RESULT: ALL CHECKS PASSED (exact match against hand computation).")


if __name__ == "__main__":
    main()
