"""
scripts/ablation_score_bimanual_gate_stage3.py

Stage 3 of the Week 7/Day 48 C3 ablation. Builds on Stage 1 (gate trained
-> checkpoints/month2_asllrp/best_val_gated.pt, step 2000, val_loss=5.7116)
and Stage 2 (data/eval_splits/symmetric_asymmetric_split_sign_level.json,
318 sign-level windows mapped onto encoder timesteps).

--- LABEL SOURCE, CORRECTED ---
The 318-window split file's own "source" field reads "heuristic_sign_level"
for all 318 windows, not a human annotation. Confirmed by tracing
scripts/battison_heuristic_labels.py's docstring: Person D's real split
never arrived (trigger date 2026-08-28 passed with no response), so the
Battison geometric-heuristic fallback became the only split that exists.
That script's own validation is a documented FAILURE: ~15% recall on
expected-asymmetric samples, ~55% balanced accuracy at best across a
9-way threshold sweep ("barely better than chance"). An independent
marker-based check run specifically against THIS sign-level file (not
assumed from the utterance-level number) found only 8/318 windows carry
an inline ASLLRP (1h)/(2h) handedness marker at all (all 8 are "(1h)",
zero "(2h)" -- the symmetric direction is untestable this way here), and
agreement on those 8 is 50-60% depending on how ties/ambiguous cases are
counted -- same "barely better than chance" neighborhood, on far too few
markers to be a real validation by itself.

Bottom line, stated plainly rather than smoothed over: every number this
script produces is against a HEURISTIC, GEOMETRY-DERIVED proxy label of
known-poor accuracy, not ground truth. Treat the whole ablation below as
"does the gate's behavior correlate with this heuristic proxy", not
"does the gate correctly detect true Battison symmetry/asymmetry".

--- WHAT THIS SCRIPT FIXES FROM ablation_score_bimanual_gate.py ---
1. load_sign_level_split() (imported unchanged from that file) already
   handles reliability/category/split filtering correctly -- reused
   as-is. What's fixed here is *dataset loading*: the original script's
   ASLLRP_NPZ_DIR = ".../train" only, silently dropping the 112/187
   reliable windows whose video lives in .../val. This script loads BOTH
   directories via two SignLanguageDataset instances and a merged
   video_id -> (split_dir, idx) index, so no window is silently dropped
   because of which split its video happens to sit in.

2. score_video_signs()'s CTC-forced-alignment single-timestep approach
   (Viterbi's own most-confident timestep for a token) is replaced here
   with the REAL, ground-truth-anchored per-window timestep RANGE from
   Stage 2: build_boundary_targets() (src/preprocessing/boundary_labels.py,
   imported directly, not reimplemented) is called with a sign_spans list
   that is all-None except the ONE sign_index being scored, against that
   video's real meta.frame_indices. This is the exact same alignment
   machinery src/models/dataset.py uses to build real training targets,
   just isolated to one sign at a time -- so a window's score reflects
   the model's output across the ACTUAL annotated frame span for that
   sign (evidence-grounded), not wherever the model's own decode
   happens to commit probability mass.
   Per-timestep rank/probability at each timestep in that range uses the
   same softmax-over-CTC-log-probs primitive already established and
   validated in diagnose_ctc_decode.py / ablation_score_bimanual_gate.py
   (rank = 1 + count of vocab entries with higher probability than the
   true gloss at that timestep) -- not a new metric, just applied over a
   different (ground-truth, not self-aligned) set of timesteps, and then
   averaged within-window (arithmetic mean of per-timestep nll/rank,
   matching this project's existing per-token averaging convention).

3. Gate-on vs gate-off comparison: uses ONE loaded model, ONE checkpoint
   (best_val_gated.pt, fully loaded including the TRAINED gate weights --
   encoder./ctc_head./boundary_head./bimanual_gate_head. all loaded),
   toggling only which forward method is called
   (forward_ctc_boundary_and_gate vs forward_ctc_and_boundary). This is
   the cleaner controlled comparison versus best_val.pt-vs-best_val_gated.pt:
   that pair would confound the gate's effect with the encoder/ctc/boundary
   weight drift from the gated run's additional joint-training steps
   (best_val_gated.pt continued training from best_val.pt's step 4000, so
   its non-gate weights are NOT identical to best_val.pt's). Same
   model instance and weights both times isolates the gate's own marginal
   effect, exactly as this project's channel_check_score_v2.py already
   does for its pose-masking probe.

4. Gate's own P(symmetric)/entropy per subset (new ask, not in the
   original Day 45 script): computed directly from
   model.bimanual_gate_head.gate(landmarks) -- a property of the trained
   gate and the input landmarks only, independent of which forward path
   is used for CTC scoring, so computed once per video and averaged over
   each window's own real timestep range (same windowing as the CTC
   metrics, for an apples-to-apples per-subset comparison).

Usage:
    uv run python scripts/ablation_score_bimanual_gate_stage3.py
    uv run python scripts/ablation_score_bimanual_gate_stage3.py --reliability all
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint_partial, get_device
from src.models.dataset import SignLanguageDataset
from src.preprocessing.sample_writer import load_sample
from src.preprocessing.boundary_labels import build_boundary_targets
from scripts.ablation_score_bimanual_gate import load_sign_level_split

VOCAB_MERGED_PATH = "data/vocab_merged.json"
DECODER_VOCAB_PATH = "data/vocab.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"
ASLLRP_NPZ_DIRS = {"train": "data/processed/asllrp/train", "val": "data/processed/asllrp/val"}
SIGN_LEVEL_SPLIT_PATH = "data/eval_splits/symmetric_asymmetric_split_sign_level.json"
CHECKPOINT_PATH = "checkpoints/month2_asllrp/best_val_gated.pt"  # TRAINED gate, step 2000

EPS = 1e-12


# ---------------------------------------------------------------------
# Merged train+val dataset index -- Stage 2's caveat fix.
# ---------------------------------------------------------------------
def build_merged_index():
    """Returns (datasets, video_id_to_loc) where video_id_to_loc[vid] =
    (split_name, idx_within_that_split's_dataset). Two separate
    SignLanguageDataset instances (unchanged class) so gloss-ID mapping
    and feature flattening stay exactly the validated per-item logic --
    nothing about __getitem__ is reimplemented here."""
    datasets = {}
    for split_name, npz_dir in ASLLRP_NPZ_DIRS.items():
        if not Path(npz_dir).exists():
            print(f"  WARNING: {npz_dir} does not exist -- skipping split {split_name!r}")
            continue
        datasets[split_name] = SignLanguageDataset(
            npz_dir=npz_dir, vocab_path=VOCAB_MERGED_PATH,
            asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
        )

    video_id_to_loc = {}
    collisions = []
    for split_name, ds in datasets.items():
        for idx, p in enumerate(ds.npz_paths):
            vid = load_sample(p).metadata.video_id
            if vid in video_id_to_loc:
                collisions.append(vid)
            video_id_to_loc[vid] = (split_name, idx)

    print(f"Merged dataset index: {sum(len(d) for d in datasets.values())} total npz files "
          f"across {list(datasets.keys())} ({', '.join(f'{k}={len(v)}' for k, v in datasets.items())})")
    if collisions:
        print(f"  *** WARNING: {len(collisions)} video_id(s) appear in more than one split dir: "
              f"{collisions[:5]} -- data integrity issue, investigate before trusting results. ***")
    return datasets, video_id_to_loc


# ---------------------------------------------------------------------
# Per-window scoring using Stage 2's real timestep-range mapping.
# ---------------------------------------------------------------------
def score_video_signs_windowed(model, item, meta, sign_entries, n_expected, blank_id, device, use_gate, video_id):
    """
    Mirrors ablation_score_bimanual_gate.py's score_video_signs(), but
    scores each sign over its REAL sign_spans-derived timestep range
    (via build_boundary_targets, called once per sign with every other
    sign's span masked to None) instead of the model's own CTC
    forced-alignment timestep.
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
    step_probs = torch.softmax(log_probs, dim=-1)

    if n_expected != len(true_ids):
        print(f"  [{video_id}] COUNT MISMATCH: split file implies {n_expected} signs, "
              f"model's true_ids has {len(true_ids)} tokens -- skipping its "
              f"{len(sign_entries)} sign(s) (sign_index-to-token-position assumption "
              f"doesn't hold for this video).")
        return []

    sign_spans_raw = meta.sign_spans
    if sign_spans_raw is None or meta.frame_indices is None:
        print(f"  [{video_id}] no sign_spans/frame_indices in metadata -- skipping "
              f"{len(sign_entries)} sign(s).")
        return []
    sign_spans_as_tuples = [tuple(s) if s is not None else None for s in sign_spans_raw]
    n_signs_total = len(sign_spans_as_tuples)

    results = []
    for s in sign_entries:
        idx = s["sign_index"]
        if idx >= n_signs_total or idx >= len(true_ids):
            print(f"  [{video_id}] sign_index {idx} out of range "
                  f"(n_signs_total={n_signs_total}, len(true_ids)={len(true_ids)}) -- skipping.")
            continue

        single_span_list = [None] * n_signs_total
        single_span_list[idx] = sign_spans_as_tuples[idx]
        window_mask, span_stats = build_boundary_targets(
            sign_spans=single_span_list, sampled_frame_indices=meta.frame_indices,
        )
        window_timesteps = np.where(window_mask == 1.0)[0]
        if len(window_timesteps) == 0:
            print(f"  [{video_id}] sign_index {idx} ({s['gloss']!r}): real span mapped to "
                  f"ZERO encoder timesteps (dropped_frame={span_stats['num_spans_dropped_frame']}, "
                  f"missing_span={span_stats['num_signs_missing_span']}) -- skipping this window.")
            continue

        gloss_id = true_ids[idx]
        nlls, ranks = [], []
        for t in window_timesteps:
            dist = step_probs[int(t)]
            true_p = dist[gloss_id].item()
            rank = int((dist > true_p).sum().item()) + 1
            nlls.append(-np.log(true_p + EPS))
            ranks.append(rank)

        mean_nll = float(np.mean(nlls))
        results.append({
            "video_id": video_id,
            "sign_index": idx,
            "gloss": s["gloss"],
            "category": s["category"],
            "reliability": s["reliability"],
            "split": s["split"],
            "window_len_timesteps": int(len(window_timesteps)),
            "window_t_start": int(window_timesteps.min()),
            "window_t_end": int(window_timesteps.max()),
            "was_point_anchor": bool(span_stats["num_signs_point_anchor"] == 1),
            "mean_nll": mean_nll,
            "mean_prob": float(np.mean([np.exp(-n) for n in nlls])),  # arithmetic mean of per-t prob, matches existing convention
            "mean_rank": float(np.mean(ranks)),
            "reciprocal_rank": float(np.mean([1.0 / r for r in ranks])),
        })
    return results


# ---------------------------------------------------------------------
# Gate's own P(symmetric)/entropy per window, independent of use_gate.
# ---------------------------------------------------------------------
def gate_stats_windowed(model, item, meta, sign_entries, n_expected, device, video_id):
    if model.bimanual_gate_head is None:
        return []
    true_tokens = item["target_tokens"].tolist()
    true_ids = true_tokens[1:-1]
    if n_expected != len(true_ids):
        return []  # same mismatch already reported by the CTC-side pass

    landmarks = item["landmarks"].unsqueeze(0).to(device)
    with torch.no_grad():
        gate_logits = model.bimanual_gate_head.gate(landmarks)  # (1, T)
    p_sym = torch.sigmoid(gate_logits[0]).cpu().numpy()
    eps = 1e-7
    entropy = -(p_sym * np.log(p_sym + eps) + (1 - p_sym) * np.log(1 - p_sym + eps))

    sign_spans_raw = meta.sign_spans
    if sign_spans_raw is None or meta.frame_indices is None:
        return []
    sign_spans_as_tuples = [tuple(sp) if sp is not None else None for sp in sign_spans_raw]
    n_signs_total = len(sign_spans_as_tuples)

    results = []
    for s in sign_entries:
        idx = s["sign_index"]
        if idx >= n_signs_total or idx >= len(true_ids):
            continue
        single_span_list = [None] * n_signs_total
        single_span_list[idx] = sign_spans_as_tuples[idx]
        window_mask, _ = build_boundary_targets(
            sign_spans=single_span_list, sampled_frame_indices=meta.frame_indices,
        )
        window_timesteps = np.where(window_mask == 1.0)[0]
        if len(window_timesteps) == 0:
            continue
        results.append({
            "video_id": video_id,
            "sign_index": idx,
            "category": s["category"],
            "reliability": s["reliability"],
            "split": s["split"],
            "mean_p_symmetric": float(np.mean(p_sym[window_timesteps])),
            "mean_gate_entropy": float(np.mean(entropy[window_timesteps])),
        })
    return results


# ---------------------------------------------------------------------
# Subset aggregation
# ---------------------------------------------------------------------
def aggregate_subset(per_sign_results):
    n = len(per_sign_results)
    if n == 0:
        return {"n_signs": 0}
    n_videos = len(set(r["video_id"] for r in per_sign_results))
    return {
        "n_signs": n,
        "n_videos": n_videos,
        "mean_nll": mean(r["mean_nll"] for r in per_sign_results),
        "mean_prob": mean(r["mean_prob"] for r in per_sign_results),
        "mean_rank": mean(r["mean_rank"] for r in per_sign_results),
        "mean_reciprocal_rank": mean(r["reciprocal_rank"] for r in per_sign_results),
    }


def aggregate_gate_stats(gate_results):
    n = len(gate_results)
    if n == 0:
        return {"n_signs": 0}
    return {
        "n_signs": n,
        "mean_p_symmetric": mean(r["mean_p_symmetric"] for r in gate_results),
        "mean_gate_entropy": mean(r["mean_gate_entropy"] for r in gate_results),
    }


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def run_stage3(reliability_filter, checkpoint_path=CHECKPOINT_PATH):
    device = get_device()
    with open(VOCAB_MERGED_PATH) as f:
        vocab_merged = json.load(f)
    with open(DECODER_VOCAB_PATH) as f:
        decoder_vocab = json.load(f)
    ctc_vocab_size = len(vocab_merged)
    blank_id = ctc_vocab_size

    print("=== Loading merged train+val dataset index (Stage 2 caveat fix) ===")
    datasets, video_id_to_loc = build_merged_index()

    print(f"\n=== Loading sign-level split: {SIGN_LEVEL_SPLIT_PATH} (reliability={reliability_filter!r}) ===")
    by_video, raw_expected_signs = load_sign_level_split(
        SIGN_LEVEL_SPLIT_PATH, reliability_filter=reliability_filter, eval_split="all",
    )

    n_not_found = sum(1 for vid in by_video if vid not in video_id_to_loc)
    print(f"  videos in split not found in merged dataset: {n_not_found}/{len(by_video)}")

    print(f"\n=== Loading model + checkpoint: {checkpoint_path} (ALL prefixes incl. trained gate) ===")
    model = SignTransformer(
        vocab_size=len(decoder_vocab), ctc_vocab_size=ctc_vocab_size, use_bimanual_gate=True,
    ).to(device)
    loaded, skipped, new_only = load_checkpoint_partial(
        model, checkpoint_path,
        prefixes_to_load=("encoder.", "ctc_head.", "boundary_head.", "bimanual_gate_head."),
    )
    print(f"Checkpoint load: {len(loaded)} loaded, {len(skipped)} skipped, {len(new_only)} new-only "
          f"(expect new_only to be empty or decoder-only -- gate weights ARE trained in this checkpoint)")
    model.eval()

    categories = ["symmetric", "asymmetric"]
    gate_modes = [("gate_off", False), ("gate_on", True)]

    all_ctc_results = {gm: {cat: [] for cat in categories} for gm, _ in gate_modes}
    gate_stat_results = {cat: [] for cat in categories}

    for cat in categories:
        videos_for_cat = {
            vid: [s for s in signs if s["category"] == cat]
            for vid, signs in by_video.items()
            if any(s["category"] == cat for s in signs)
        }
        for vid, sign_entries in videos_for_cat.items():
            if vid not in video_id_to_loc:
                print(f"  [{vid}] not found in merged train+val dataset -- skipping")
                continue
            split_name, idx = video_id_to_loc[vid]
            ds = datasets[split_name]
            item = ds[idx]
            meta = load_sample(ds.npz_paths[idx]).metadata

            for gate_mode_name, use_gate in gate_modes:
                res = score_video_signs_windowed(
                    model, item, meta, sign_entries, raw_expected_signs[vid],
                    blank_id, device, use_gate, vid,
                )
                all_ctc_results[gate_mode_name][cat].extend(res)

            gate_stat_results[cat].extend(
                gate_stats_windowed(model, item, meta, sign_entries, raw_expected_signs[vid], device, vid)
            )

    summary = {gm: {cat: aggregate_subset(all_ctc_results[gm][cat]) for cat in categories} for gm, _ in gate_modes}
    gate_summary = {cat: aggregate_gate_stats(gate_stat_results[cat]) for cat in categories}

    return {
        "summary": summary,
        "gate_summary": gate_summary,
        "per_sign": all_ctc_results,
        "gate_per_sign": gate_stat_results,
    }


def print_table(summary, reliability_label):
    print(f"\n========== C3 Ablation (Stage 3, {reliability_label}) ==========")
    print("gate-off = trained checkpoint, forward_ctc_and_boundary (gate bypassed)")
    print("gate-on  = SAME checkpoint/weights, forward_ctc_boundary_and_gate (trained gate active)")
    header = f"{'metric':<24} {'off/sym':>12} {'off/asym':>12} {'on/sym':>12} {'on/asym':>12}"
    print(header)
    for key in ["n_signs", "n_videos", "mean_nll", "mean_prob", "mean_rank", "mean_reciprocal_rank"]:
        row = [key]
        for gate_name in ["gate_off", "gate_on"]:
            for cat in ["symmetric", "asymmetric"]:
                v = summary[gate_name][cat].get(key)
                row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print(f"{row[0]:<24} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12}")


def print_gate_table(gate_summary, reliability_label):
    print(f"\n---- Gate's own P(symmetric)/entropy per subset ({reliability_label}) ----")
    print("(overall training-run range was P(symmetric) 0.37-0.64, mean 0.52, std 0.054, "
          "entropy near ln(2)=0.693 -- compare these subset numbers against that baseline)")
    header = f"{'metric':<22} {'symmetric':>14} {'asymmetric':>14}"
    print(header)
    for key in ["n_signs", "mean_p_symmetric", "mean_gate_entropy"]:
        row = [key]
        for cat in ["symmetric", "asymmetric"]:
            v = gate_summary[cat].get(key)
            row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print(f"{row[0]:<22} {row[1]:>14} {row[2]:>14}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    all_results = {}
    for reliability_filter in ["reliable", "all"]:
        label = "RELIABLE-ONLY" if reliability_filter == "reliable" else "FULL SET (incl. low_confidence_short_window)"
        print(f"\n\n################ {label} ################")
        results = run_stage3(reliability_filter, checkpoint_path=args.checkpoint)
        print_table(results["summary"], label)
        print_gate_table(results["gate_summary"], label)
        all_results[reliability_filter] = results

    out_path = Path("logs") / "ablation_score_bimanual_gate_stage3.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "label_source_caveat": "heuristic_sign_level (Battison geometric proxy), NOT human-annotated ground truth",
            "gate_off_method": "same checkpoint/weights as gate_on, forward_ctc_and_boundary (gate bypassed)",
            "results": {
                rel: {
                    "summary": r["summary"],
                    "gate_summary": r["gate_summary"],
                    "per_sign": r["per_sign"],
                    "gate_per_sign": r["gate_per_sign"],
                }
                for rel, r in all_results.items()
            },
        }, f, indent=2)
    print(f"\n\nFull detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
