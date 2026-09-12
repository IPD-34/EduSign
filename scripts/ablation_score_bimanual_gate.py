"""
scripts/ablation_score_bimanual_gate.py

Day 45 (SCOPED): the C3 ablation SCORING script only. Does NOT build the
Battison's-conditions heuristic labeling function -- that is explicitly
deferred (see "PENDING" section below). Person D's real symmetric/
asymmetric split has not arrived as of this writing.

--- LABEL-SOURCE-AGNOSTIC BY DESIGN ---
This script takes a {sample_id -> "symmetric"|"asymmetric"} mapping as
INPUT -- a JSON file (or, if calling run_ablation() directly, a plain
dict). It has no idea and does not care whether that mapping came from
a person or a heuristic. This is deliberate: the scoring logic below is
needed identically regardless of which label source eventually feeds it,
so building it now (while the label source is still undecided) is not
wasted work, unlike the heuristic itself would be if Person D's split
lands first.

PENDING -- next steps depend on what happens with Person D by the Friday
2026-08-28 trigger date (per the Day 41 escalation):
  (a) If her real split arrives: write a small adapter that converts her
      split's file format into this script's {id -> label} JSON shape.
      Nothing else changes -- feed the adapter's output straight into
      --labels below.
  (b) If it does not arrive by the trigger date: build the Battison's
      Dominance/Symmetry Conditions heuristic labeling function next,
      have it emit the same {id -> label} JSON shape, and feed ITS
      output into this same script, unchanged.
Do not build (b) preemptively -- see Day 45's brief.

--- METRIC ---
Per Day 40's finding (checkpoints/month2_asllrp/best_val.pt, step 4000,
greedy CTC decode collapses to blank at ~100% of timesteps -- expected
early-training blank-dominance, not a bug), top-1/greedy WER is not a
usable signal yet. This script scores rank/probability of the TRUE gloss
sequence instead, mirroring channel_check_score_v2.py's rank/probability-
sensitive approach (itself adopted after Day 17 found top-1 accuracy was
the wrong lens at this training stage) -- adapted for CTC sequences via
the exact forward-algorithm probability and forced-alignment per-token
rank already built and validated in diagnose_ctc_decode.py (imported
here, not re-implemented, to avoid a second copy of that math silently
drifting from the first).

--- CONFIGURATION ---
Evaluates ONE loaded model two ways -- gate active
(forward_ctc_boundary_and_gate) vs. gate forcibly disabled
(forward_ctc_and_boundary, the pre-Day-44 code path) -- same model
instance and weights both times, same "one trained model, toggle how
it's invoked at inference" method channel_check_score_v2.py already
uses for its pose-masking probe. Each of the two gate configs is scored
separately on whichever symmetric subset and asymmetric subset the input
mapping defines.

No checkpoint with TRAINED gate weights exists yet -- Day 44 was a local
smoke test only, no real Kaggle run has trained SignTransformer with
use_bimanual_gate=True. This script loads checkpoints/month2_asllrp/
best_val.pt (the real checkpoint of record) via load_checkpoint_partial
for encoder/ctc_head/boundary_head, leaving bimanual_gate_head at fresh
random init -- stated explicitly in the load report, not hidden. The
"gate active" numbers today reflect an UNTRAINED gate; that is expected
and is exactly why today's run is a plumbing test, not a finding.

--- TODAY'S TEST RUN ---
Called with --dummy-n, this script generates a RANDOM {id -> label}
mapping over N real ASLLRP samples (fixed seed, clearly separate code
path, written to logs/day45_dummy_labels_PLUMBING_TEST.json and then
read back through the same file-loading path a real mapping would use)
purely to confirm the plumbing works end-to-end: correct per-subset
shapes, gate-on/gate-off mechanics, no crashes. The resulting numbers
are NOT a real finding -- the labels are random noise, and the gate is
untrained. Do not cite this run's numbers as evidence of anything.

Usage:
    # plumbing test (today):
    uv run python scripts/ablation_score_bimanual_gate.py --dummy-n 20

    # real use (once a real or heuristic label file exists):
    uv run python scripts/ablation_score_bimanual_gate.py --labels path/to/labels.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, get_device
from src.models.dataset import SignLanguageDataset, collate_fn
from src.preprocessing.sample_writer import load_sample
from diagnose_ctc_decode import true_sequence_nll, ctc_forced_alignment

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"  # 1850 entries -- old WLASL decoder vocab
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
ASLLRP_NPZ_DIR = "data/processed/asllrp/train"
CHECKPOINT_PATH = "checkpoints/month2_asllrp/best_val.pt"
DUMMY_LABELS_PATH = Path("logs/day45_dummy_labels_PLUMBING_TEST.json")

VALID_LABELS = {"symmetric", "asymmetric"}

SIGN_LEVEL_KEY_RE = None  # not needed, we split on "::" directly
VALID_SIGN_CATEGORIES = {"symmetric", "asymmetric"}  # "ambiguous" is always excluded


# ---------------------------------------------------------------------
# Label mapping: file-based, source-agnostic
# ---------------------------------------------------------------------
def load_label_mapping(path) -> dict:
    """path -> {video_id: "symmetric"|"asymmetric"}. The ONLY assumption
    this script makes about where labels came from: a flat JSON object
    with those two string values. Works identically for Person D's real
    split (via a future adapter), a future Battison's-heuristic output,
    or today's dummy plumbing-test labels."""
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    bad = {v for v in mapping.values() if v not in VALID_LABELS}
    if bad:
        raise ValueError(f"Label file {path} has invalid label value(s): {bad} "
                          f"(only {VALID_LABELS} allowed)")
    return mapping


def load_sign_level_split(path, reliability_filter="reliable", eval_split="all"):
    """
    Loads Person D's data/eval_splits/symmetric_asymmetric_split_sign_level.json
    format: {"windows": {"video_id::signNN": {category, gloss, sign_index,
    reliability, split, ...}}, "meta": {...}}.

    reliability_filter: "reliable" (default) or "all" -- applied to BOTH
      symmetric and asymmetric subsets, since the reliability field spans
      all categories (262 reliable / 56 low_confidence_short_window total),
      not just asymmetric.
    eval_split: "all" (default), "train", or "val" -- see WARNING below.

    Returns: (by_video, raw_expected_signs) --
      by_video: dict[video_id] -> list of {sign_index, gloss, category,
        reliability, split} dicts, already filtered. "ambiguous" category
        windows are always dropped (93 windows) -- scoring against a
        category the labeler itself couldn't confidently place would
        muddy the ablation.
      raw_expected_signs: dict[video_id] -> max(sign_index)+1 computed
        from the RAW, UNFILTERED window set (all categories, all
        reliability levels) for that video. Exists specifically so the
        downstream count-mismatch check compares against the TRUE total
        sign count D produced for the utterance, not a count that's
        already had ambiguous/low-confidence signs stripped out --
        computing it from by_video (post-filter) undercounts whenever a
        video has any such signs, producing false mismatches even when
        D's real total agrees exactly with the model's token count.
        Caught empirically: all 10 "mismatches" in the first real run
        against D's actual file turned out to be this artifact -- 0/20
        videos found in train showed a genuine count disagreement once
        checked against the true unfiltered count.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    windows = raw["windows"]
    meta = raw.get("meta", {})

    raw_expected_signs = {}
    for key, w in windows.items():
        video_id, _ = key.split("::")
        prev = raw_expected_signs.get(video_id, 0)
        raw_expected_signs[video_id] = max(prev, w["sign_index"] + 1)

    print(f"Loaded sign-level split: {len(windows)} windows "
          f"({meta.get('samples_processed', '?')} source samples)")

    if eval_split != "all":
        print(f"  *** WARNING: filtering to split=={eval_split!r}. D's train/val "
              f"partition is NOT confirmed to be the same partition your "
              f"CTC/boundary/gate model was trained on (990/122/122 ASLLRP "
              f"split). Filtering on the wrong assumption could either hide "
              f"leakage or silently discard good data. If unconfirmed, prefer "
              f"eval_split='all' and inspect the by-split breakdown below "
              f"instead of trusting a filter blindly. ***")

    by_video = {}
    dropped_ambiguous = dropped_reliability = dropped_split = 0
    for key, w in windows.items():
        video_id, sign_tag = key.split("::")
        cat = w.get("category")
        if cat not in VALID_SIGN_CATEGORIES:
            dropped_ambiguous += 1
            continue
        if reliability_filter == "reliable" and w.get("reliability") != "reliable":
            dropped_reliability += 1
            continue
        if eval_split != "all" and w.get("split") != eval_split:
            dropped_split += 1
            continue
        by_video.setdefault(video_id, []).append({
            "sign_index": w["sign_index"],
            "gloss": w.get("gloss"),
            "category": cat,
            "reliability": w.get("reliability"),
            "split": w.get("split"),
        })

    n_sym = sum(1 for v in by_video.values() for s in v if s["category"] == "symmetric")
    n_asym = sum(1 for v in by_video.values() for s in v if s["category"] == "asymmetric")
    print(f"  after filtering (reliability={reliability_filter!r}, eval_split={eval_split!r}): "
          f"{n_sym} symmetric signs, {n_asym} asymmetric signs, across {len(by_video)} videos")
    print(f"  dropped: {dropped_ambiguous} ambiguous, {dropped_reliability} low-confidence "
          f"(reliability filter), {dropped_split} wrong-split")

    # Report the train/val breakdown regardless of eval_split, since this is
    # exactly the kind of thing that should never be silently assumed.
    split_counts = {}
    for v in by_video.values():
        for s in v:
            split_counts[s["split"]] = split_counts.get(s["split"], 0) + 1
    print(f"  by D's split field (informational, not necessarily your model's "
          f"train/held-out split): {split_counts}")

    return by_video, raw_expected_signs


def generate_dummy_label_mapping(dataset, n, seed=45) -> dict:
    """PLUMBING-TEST ONLY. Randomly assigns n real ASLLRP video_ids to
    symmetric/asymmetric. NOT a labeling methodology -- do not reuse
    this function's output as anything other than a mechanism check."""
    rng = random.Random(seed)
    all_video_ids = []
    for p in dataset.npz_paths:
        all_video_ids.append(load_sample(p).metadata.video_id)
    n = min(n, len(all_video_ids))
    chosen = rng.sample(all_video_ids, n)
    return {vid: rng.choice(["symmetric", "asymmetric"]) for vid in chosen}


# ---------------------------------------------------------------------
# Per-sample scoring: exact CTC forward-algorithm probability +
# forced-alignment per-token rank (imported from diagnose_ctc_decode.py)
# ---------------------------------------------------------------------
def score_sample(model, item, blank_id, device, use_gate):
    landmarks = item["landmarks"].unsqueeze(0).to(device)
    T = landmarks.shape[1]
    encoder_mask = torch.ones(1, T, dtype=torch.bool, device=device)

    true_tokens = item["target_tokens"].tolist()
    true_ids = true_tokens[1:-1]  # strip BOS/EOS, same convention as eval_holdout.py

    with torch.no_grad():
        if use_gate:
            ctc_log_probs, _boundary_logits, _gate_logits = model.forward_ctc_boundary_and_gate(
                landmarks, encoder_mask
            )
        else:
            ctc_log_probs, _boundary_logits = model.forward_ctc_and_boundary(
                landmarks, encoder_mask
            )
    log_probs = ctc_log_probs[0].cpu()  # (T, V+1)

    nll = true_sequence_nll(log_probs, true_ids, blank_id)
    n = max(len(true_ids), 1)
    nll_per_token = nll / n
    prob_per_token = torch.exp(torch.tensor(-nll_per_token)).item()

    alignment = ctc_forced_alignment(log_probs, true_ids, blank_id)
    ranks = []
    if alignment is not None:
        step_probs = torch.softmax(log_probs, dim=-1)
        for gloss_id, t in alignment:
            dist = step_probs[t]
            true_p = dist[gloss_id].item()
            ranks.append(int((dist > true_p).sum().item()) + 1)

    return {
        "video_id": item.get("video_id"),
        "n_glosses": len(true_ids),
        "nll_per_token": nll_per_token,
        "prob_per_token": prob_per_token,
        "mean_rank": (sum(ranks) / len(ranks)) if ranks else None,
        "mean_reciprocal_rank": (sum(1.0 / r for r in ranks) / len(ranks)) if ranks else None,
        "alignment_failed": alignment is None,
    }


def score_subset(model, dataset, indices, blank_id, device, use_gate):
    results = []
    for idx in indices:
        item = dataset[idx]
        item["video_id"] = load_sample(dataset.npz_paths[idx]).metadata.video_id
        results.append(score_sample(model, item, blank_id, device, use_gate))

    n = len(results)
    if n == 0:
        return {"n_samples": 0}, results

    ranked = [r for r in results if r["mean_rank"] is not None]
    summary = {
        "n_samples": n,
        "n_alignment_failed": n - len(ranked),
        "mean_nll_per_token": sum(r["nll_per_token"] for r in results) / n,
        "mean_prob_per_token": sum(r["prob_per_token"] for r in results) / n,
        "mean_rank": (sum(r["mean_rank"] for r in ranked) / len(ranked)) if ranked else None,
        "mean_reciprocal_rank": (sum(r["mean_reciprocal_rank"] for r in ranked) / len(ranked)) if ranked else None,
    }
    return summary, results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run_ablation(label_mapping, checkpoint_path=CHECKPOINT_PATH, max_per_subset=None):
    device = get_device()
    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)
    blank_id = ctc_vocab_size
    random_baseline_prob_per_token = 1.0 / (ctc_vocab_size + 1)
    random_baseline_nll_per_token = torch.log(torch.tensor(float(ctc_vocab_size + 1))).item()

    dataset = SignLanguageDataset(
        npz_dir=ASLLRP_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )

    print(f"Scanning {len(dataset)} dataset samples against {len(label_mapping)} mapped IDs...")
    video_id_to_idx = {load_sample(p).metadata.video_id: i for i, p in enumerate(dataset.npz_paths)}

    symmetric_idxs, asymmetric_idxs, unmatched = [], [], []
    for vid, label in label_mapping.items():
        if vid not in video_id_to_idx:
            unmatched.append(vid)
            continue
        (symmetric_idxs if label == "symmetric" else asymmetric_idxs).append(video_id_to_idx[vid])

    if max_per_subset:
        symmetric_idxs = symmetric_idxs[:max_per_subset]
        asymmetric_idxs = asymmetric_idxs[:max_per_subset]

    print(f"  symmetric subset:  {len(symmetric_idxs)} samples found in dataset")
    print(f"  asymmetric subset: {len(asymmetric_idxs)} samples found in dataset")
    print(f"  mapped IDs not found in dataset: {len(unmatched)}"
          + (f"  (e.g. {unmatched[:5]})" if unmatched else ""))
    if not symmetric_idxs or not asymmetric_idxs:
        print("  WARNING: at least one subset is empty -- comparison below will be incomplete.")

    model = SignTransformer(
        vocab_size=len(decoder_vocab), ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
    ).to(device)
    loaded, skipped, new_only = load_checkpoint_partial(
        model, checkpoint_path, prefixes_to_load=("encoder.", "ctc_head.", "boundary_head.")
    )
    print(f"Checkpoint load: {len(loaded)} loaded, {len(skipped)} skipped, {len(new_only)} new "
          f"(bimanual_gate_head is expected to be in 'new' -- no gate-trained checkpoint exists yet)")
    model.eval()

    results = {}
    for gate_name, use_gate in [("gate_off", False), ("gate_on", True)]:
        results[gate_name] = {}
        for subset_name, idxs in [("symmetric", symmetric_idxs), ("asymmetric", asymmetric_idxs)]:
            summary, per_sample = score_subset(model, dataset, idxs, blank_id, device, use_gate)
            summary["random_baseline_prob_per_token"] = random_baseline_prob_per_token
            summary["random_baseline_nll_per_token"] = random_baseline_nll_per_token
            results[gate_name][subset_name] = {"summary": summary, "per_sample": per_sample}

    return results, {"unmatched_ids": unmatched, "n_symmetric": len(symmetric_idxs),
                      "n_asymmetric": len(asymmetric_idxs)}


# ---------------------------------------------------------------------
# Sign-level scoring: Person D's per-sign windows, indexed into one
# forced-alignment pass per utterance
# ---------------------------------------------------------------------
def score_video_signs(model, item, sign_entries, n_expected, blank_id, device, use_gate, video_id):
    """
    One forward pass for the whole utterance; CTC forced-alignment gives a
    (gloss_id, timestep) pair per true token in order. sign_index in D's
    file is ASSUMED to correspond 1:1 to token position in true_ids --
    checked here, not trusted blindly. Any mismatch is flagged per-video
    and those signs are skipped (not silently mis-scored).

    sign_entries: this call's category-filtered signs (what gets scored).
    n_expected: the RAW, unfiltered sign count for this video (from
    load_sign_level_split's raw_expected_signs, computed before any
    category/reliability/split filtering). Two bugs already found and
    fixed on this exact line: (1) originally computed from sign_entries
    alone (one category), so max(sign_index) depended on which category
    held the highest index; (2) then computed from by_video[vid], which
    is post-reliability/ambiguous-filter, so it still undercounted for
    any video with dropped signs. Verified empirically against D's real
    file: with n_expected passed in raw like this, 0/20 videos found in
    train show a genuine mismatch -- all 10 previously-reported
    "mismatches" were filtering-before-counting artifacts, not real
    tokenization drift.
    """
    landmarks = item["landmarks"].unsqueeze(0).to(device)
    T = landmarks.shape[1]
    encoder_mask = torch.ones(1, T, dtype=torch.bool, device=device)
    true_tokens = item["target_tokens"].tolist()
    true_ids = true_tokens[1:-1]

    with torch.no_grad():
        if use_gate:
            ctc_log_probs, _bl, _gl = model.forward_ctc_boundary_and_gate(landmarks, encoder_mask)
        else:
            ctc_log_probs, _bl = model.forward_ctc_and_boundary(landmarks, encoder_mask)
    log_probs = ctc_log_probs[0].cpu()

    alignment = ctc_forced_alignment(log_probs, true_ids, blank_id)
    results = []

    if alignment is None:
        print(f"  [{video_id}] alignment FAILED -- skipping {len(sign_entries)} sign(s)")
        return results
    if len(alignment) != len(true_ids):
        print(f"  [{video_id}] alignment length {len(alignment)} != true_ids length "
              f"{len(true_ids)} -- skipping (unexpected, shouldn't happen)")
        return results
    if n_expected != len(true_ids):
        print(f"  [{video_id}] COUNT MISMATCH: D's file implies {n_expected} signs, "
              f"model's true_ids has {len(true_ids)} tokens. Sign-index-to-token "
              f"assumption may not hold for this video -- skipping its {len(sign_entries)} sign(s).")
        return results

    step_probs = torch.softmax(log_probs, dim=-1)
    for s in sign_entries:
        idx = s["sign_index"]
        if idx >= len(alignment):
            print(f"  [{video_id}] sign_index {idx} out of range for alignment "
                  f"(len {len(alignment)}) -- skipping this sign")
            continue
        gloss_id, t = alignment[idx]
        dist = step_probs[t]
        true_p = dist[gloss_id].item()
        rank = int((dist > true_p).sum().item()) + 1
        results.append({
            "video_id": video_id,
            "sign_index": idx,
            "gloss": s["gloss"],
            "reliability": s["reliability"],
            "split": s["split"],
            "prob": true_p,
            "nll": -torch.log(torch.tensor(true_p + 1e-12)).item(),
            "rank": rank,
            "reciprocal_rank": 1.0 / rank,
        })
    return results


def score_sign_level_subset(model, dataset, video_id_to_idx, by_video, raw_expected_signs,
                             category, blank_id, device, use_gate):
    videos_for_category = {
        vid: [s for s in signs if s["category"] == category]
        for vid, signs in by_video.items()
        if any(s["category"] == category for s in signs)
    }
    all_results = []
    for vid, sign_entries in videos_for_category.items():
        if vid not in video_id_to_idx:
            print(f"  [{vid}] not found in loaded dataset ({ASLLRP_NPZ_DIR}) -- skipping")
            continue
        item = dataset[video_id_to_idx[vid]]
        all_results.extend(
            score_video_signs(model, item, sign_entries, raw_expected_signs[vid],
                               blank_id, device, use_gate, vid)
        )

    n = len(all_results)
    if n == 0:
        return {"n_signs": 0}, all_results
    summary = {
        "n_signs": n,
        "n_videos": len(set(r["video_id"] for r in all_results)),
        "mean_nll": sum(r["nll"] for r in all_results) / n,
        "mean_prob": sum(r["prob"] for r in all_results) / n,
        "mean_rank": sum(r["rank"] for r in all_results) / n,
        "mean_reciprocal_rank": sum(r["reciprocal_rank"] for r in all_results) / n,
    }
    return summary, all_results


def run_sign_level_ablation(by_video, raw_expected_signs, checkpoint_path=CHECKPOINT_PATH):
    device = get_device()
    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)
    blank_id = ctc_vocab_size

    dataset = SignLanguageDataset(
        npz_dir=ASLLRP_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )
    video_id_to_idx = {load_sample(p).metadata.video_id: i for i, p in enumerate(dataset.npz_paths)}

    model = SignTransformer(
        vocab_size=len(decoder_vocab), ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
    ).to(device)
    loaded, skipped, new_only = load_checkpoint_partial(
        model, checkpoint_path, prefixes_to_load=("encoder.", "ctc_head.", "boundary_head.")
    )
    print(f"Checkpoint load: {len(loaded)} loaded, {len(skipped)} skipped, {len(new_only)} new "
          f"(bimanual_gate_head expected in 'new' -- still untrained as of this run)")
    model.eval()

    results = {}
    for gate_name, use_gate in [("gate_off", False), ("gate_on", True)]:
        results[gate_name] = {}
        for category in ["symmetric", "asymmetric"]:
            summary, per_sign = score_sign_level_subset(
                model, dataset, video_id_to_idx, by_video, raw_expected_signs,
                category, blank_id, device, use_gate
            )
            results[gate_name][category] = {"summary": summary, "per_sign": per_sign}
    return results


def print_sign_level_comparison(results):
    print("\n==========================")
    print("C3 Ablation (SIGN-LEVEL): gate-off vs gate-on, per category")
    print("==========================")
    header = f"{'metric':<20} {'off/sym':>12} {'off/asym':>12} {'on/sym':>12} {'on/asym':>12}"
    print(header)
    for key in ["n_signs", "n_videos", "mean_nll", "mean_prob", "mean_rank", "mean_reciprocal_rank"]:
        row = [key]
        for gate_name in ["gate_off", "gate_on"]:
            for cat in ["symmetric", "asymmetric"]:
                v = results[gate_name][cat]["summary"].get(key)
                row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print(f"{row[0]:<20} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12}")


def print_comparison(results):
    print("\n==========================")
    print("C3 Ablation Scoring: gate-off vs gate-on, per subset")
    print("(rank/probability-sensitive metrics -- top-1/greedy WER not usable yet, per Day 40)")
    print("==========================")
    header = f"{'metric':<26} {'off/sym':>12} {'off/asym':>12} {'on/sym':>12} {'on/asym':>12}"
    print(header)
    for key in ["n_samples", "mean_nll_per_token", "mean_prob_per_token", "mean_rank", "mean_reciprocal_rank"]:
        row = [key]
        for gate_name in ["gate_off", "gate_on"]:
            for subset_name in ["symmetric", "asymmetric"]:
                v = results[gate_name][subset_name]["summary"].get(key)
                row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print(f"{row[0]:<26} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12}")
    rb = results["gate_off"]["symmetric"]["summary"].get("random_baseline_prob_per_token")
    if rb:
        print(f"\n(random_baseline_prob_per_token = {rb:.6f}, for scale)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default=None,
                         help="Path to a {video_id: 'symmetric'|'asymmetric'} JSON file. "
                              "Real usage -- from Person D's split (via adapter) or the "
                              "Battison's heuristic, once either exists.")
    parser.add_argument("--dummy-n", type=int, default=None,
                         help="PLUMBING TEST ONLY: generate N random labels over real ASLLRP "
                              "samples instead of reading --labels. Numbers from this mode are "
                              "not a finding.")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--max-per-subset", type=int, default=None,
                         help="Cap samples scored per subset (for a faster local run).")
    parser.add_argument("--sign-level-labels", type=str, default=None,
                         help="Path to Person D's symmetric_asymmetric_split_sign_level.json. "
                              "Mutually exclusive with --labels/--dummy-n.")
    parser.add_argument("--reliability", choices=["reliable", "all"], default="reliable",
                         help="Sign-level only. 'reliable' (default) excludes the 56 "
                              "low_confidence_short_window signs.")
    parser.add_argument("--eval-split", choices=["all", "train", "val"], default="all",
                         help="Sign-level only. Filters by D's split field. Default 'all' "
                              "since D's train/val partition is not confirmed to match your "
                              "model's own train/held-out split -- see printed warning.")
    args = parser.parse_args()

    if args.labels and args.dummy_n:
        parser.error("--labels and --dummy-n are mutually exclusive.")
    if not args.labels and not args.dummy_n and not args.sign_level_labels:
        parser.error("Pass either --labels <file> (real use), --dummy-n <N> (plumbing test), "
                      "or --sign-level-labels <file> (Person D's sign-level split).")
    if args.sign_level_labels and (args.labels or args.dummy_n):
        parser.error("--sign-level-labels is mutually exclusive with --labels/--dummy-n.")

    if args.sign_level_labels:
        by_video, raw_expected_signs = load_sign_level_split(
            args.sign_level_labels, reliability_filter=args.reliability, eval_split=args.eval_split
        )
        results = run_sign_level_ablation(by_video, raw_expected_signs, checkpoint_path=args.checkpoint)
        print_sign_level_comparison(results)
        out_path = Path("logs") / "ablation_score_bimanual_gate_sign_level.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "is_plumbing_test_do_not_cite": False,
                "checkpoint": args.checkpoint,
                "reliability_filter": args.reliability,
                "eval_split": args.eval_split,
                "results": results,
            }, f, indent=2)
        print(f"\nDetailed results saved to {out_path}")
        return  # skip the rest of main() (old utterance-level path)

    is_dummy_run = bool(args.dummy_n)

    if is_dummy_run:
        print("*** PLUMBING TEST MODE (--dummy-n): labels are RANDOM, gate is UNTRAINED. ***")
        print("*** Numbers below confirm the script runs correctly. They are NOT a finding. ***\n")
        dataset_for_dummy = SignLanguageDataset(
            npz_dir=ASLLRP_NPZ_DIR, vocab_path=VOCAB_MERGED_PATH,
            asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
        )
        mapping = generate_dummy_label_mapping(dataset_for_dummy, args.dummy_n)
        DUMMY_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DUMMY_LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        print(f"Wrote dummy labels to {DUMMY_LABELS_PATH} ({len(mapping)} IDs)")
        label_mapping = load_label_mapping(DUMMY_LABELS_PATH)  # exercise the real file-load path
    else:
        label_mapping = load_label_mapping(args.labels)
        print(f"Loaded {len(label_mapping)} labels from {args.labels}")

    results, coverage = run_ablation(label_mapping, checkpoint_path=args.checkpoint,
                                      max_per_subset=args.max_per_subset)
    print_comparison(results)

    out_path = Path("logs") / (
        "day45_ablation_PLUMBING_TEST.json" if is_dummy_run else "ablation_score_bimanual_gate.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "is_plumbing_test_do_not_cite": is_dummy_run,
            "checkpoint": args.checkpoint,
            "coverage": coverage,
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")

    if is_dummy_run:
        print("\n*** REMINDER: this was a plumbing test with random labels and an untrained ***")
        print("*** gate. Do not report these numbers as any kind of real finding.          ***")


if __name__ == "__main__":
    main()
