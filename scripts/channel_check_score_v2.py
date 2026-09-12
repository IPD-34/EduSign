"""
channel_check_score_v2.py

Day 17 (revised): run both channel configs through the Day 12(+)
checkpoint, compare using RANK/PROBABILITY-SENSITIVE metrics instead
of top-1 argmax accuracy.

WHY THIS EXISTS: channel_check_score.py's top-1 accuracy came back
0/144 for both configs. diagnose_gloss_position.py confirmed this was
NOT a bug -- on a known training sample, the model ranked the true
gloss 12/1850 with ~12x random-baseline probability mass, i.e. it is
genuinely learning, just not yet reliable enough for top-1 argmax to
be a meaningful signal at this training stage. Top-1 accuracy is the
wrong lens for a model at this point in training; this script swaps
in metrics that are sensitive to partial/ranked correctness instead,
per the same "verify beyond doesn't-crash" discipline used throughout
this project (see Day 10's exact-mask-sum check, Day 11's
random-baseline loss check).

METRICS REPORTED (per config, and per-clip in the saved JSON):
  - true-gloss probability   -- softmax probability mass on the correct
                                 token. Directly comparable to the
                                 random baseline (1/vocab_size).
  - true-gloss rank          -- 1-indexed rank of the correct token
                                 among all vocab entries, by probability.
                                 Lower is better; vocab_size/2 is the
                                 expected rank of a token under a
                                 uniform (untrained) model.
  - top-5 accuracy           -- fraction of clips where the true gloss
                                 is in the model's top 5 predictions.
  - top-1 accuracy           -- still reported for reference/comparison
                                 against the original script, but not
                                 treated as the decision metric while
                                 the model is under-trained.

METHOD (unchanged from channel_check_score.py): lightweight probe.
Config B (hands+face+body) evaluates the trained model as-is. Config A
(hands+face only) zeros the pose channel of the SAME model's input at
inference time. This is a probe of degradation-without-pose on an
already-trained model, not a from-scratch separate-training comparison
-- document this alongside the numbers, same as before.

Usage:
    uv run python channel_check_score_v2.py
"""

import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.config import PROCESSED_DATA_DIR
from src.preprocessing.sample_writer import load_sample
from src.models.feature_utils import flatten_frame_features
from src.models.sign_transformer import SignTransformer
from src.models.train_utils import load_checkpoint, get_device

CHANNEL_CHECK_PROCESSED_DIR = PROCESSED_DATA_DIR / "channel_check"
MANIFEST_PATH = CHANNEL_CHECK_PROCESSED_DIR / "manifest.csv"
CHECKPOINT_PATH = Path("checkpoints") / "day12_latest.pt"

TOP_K = 5

# Pose is the first 132 values of the flattened per-frame feature vector
# (33 landmarks x 4 = 132). VERIFY against feature_utils.py's actual
# concatenation order before trusting this.
POSE_DIM_SLICE = slice(0, 132)


def score_sample(model, landmarks, true_gloss, token_to_id, id_to_token, bos_id, device, top_k=TOP_K):
    """Single-step decode, but instead of only taking argmax, compute
    the full probability distribution so rank/probability/top-k can
    all be read off the same forward pass."""
    landmarks = landmarks.unsqueeze(0).to(device)          # (1, T, feat_dim)
    encoder_mask = torch.ones(1, landmarks.shape[1], dtype=torch.bool, device=device)
    target_tokens = torch.tensor([[bos_id]], device=device)  # (1, 1)

    with torch.no_grad():
        logits = model(landmarks, encoder_mask, target_tokens, target_padding_mask=None)
        probs = torch.softmax(logits[0, -1, :], dim=-1)

    true_id = token_to_id.get(true_gloss, token_to_id["<UNK>"])
    true_prob = probs[true_id].item()
    # rank: 1 = model's top prediction. Count how many tokens have
    # strictly higher probability than the true one, +1.
    rank = int((probs > true_prob).sum().item()) + 1

    top_k_ids = torch.topk(probs, top_k).indices.tolist()
    top_k_tokens = [id_to_token.get(i, "<UNK>") for i in top_k_ids]
    top1_correct = (top_k_tokens[0] == true_gloss)
    topk_correct = (true_gloss in top_k_tokens)

    return {
        "true_prob": true_prob,
        "rank": rank,
        "top1_correct": top1_correct,
        f"top{top_k}_correct": topk_correct,
        f"top{top_k}_predictions": top_k_tokens,
    }


def run_config(model, samples, token_to_id, id_to_token, bos_id, device, mask_pose, vocab_size):
    per_sample_results = []

    for video_id, gloss in samples:
        npz_path = CHANNEL_CHECK_PROCESSED_DIR / f"{video_id}.npz"
        if not npz_path.exists():
            continue

        sample = load_sample(npz_path)
        landmarks = flatten_frame_features(sample)
        landmarks = torch.from_numpy(landmarks).float()  # (T, 1692)

        if mask_pose:
            landmarks = landmarks.clone()
            landmarks[:, POSE_DIM_SLICE] = 0.0

        result = score_sample(model, landmarks, gloss, token_to_id, id_to_token, bos_id, device)
        result["video_id"] = video_id
        result["true_gloss"] = gloss
        per_sample_results.append(result)

    n = len(per_sample_results)
    if n == 0:
        return None

    mean_true_prob = sum(r["true_prob"] for r in per_sample_results) / n
    mean_rank = sum(r["rank"] for r in per_sample_results) / n
    mean_reciprocal_rank = sum(1.0 / r["rank"] for r in per_sample_results) / n
    top1_acc = sum(r["top1_correct"] for r in per_sample_results) / n
    topk_acc = sum(r[f"top{TOP_K}_correct"] for r in per_sample_results) / n

    summary = {
        "n_samples": n,
        "mean_true_gloss_probability": mean_true_prob,
        "random_baseline_probability": 1.0 / vocab_size,
        "mean_rank": mean_rank,
        "mean_reciprocal_rank": mean_reciprocal_rank,
        "top1_accuracy": top1_acc,
        f"top{TOP_K}_accuracy": topk_acc,
    }
    return summary, per_sample_results


def main():
    if not MANIFEST_PATH.exists():
        print(f"No processed manifest at {MANIFEST_PATH} -- run channel_check_extract.py first.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "success"]
    samples = [(r["video_id"], r["gloss"]) for r in rows]
    print(f"Scoring {len(samples)} successfully extracted clips")

    with open("data/vocab.json", "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    id_to_token = {v: k for k, v in token_to_id.items()}
    bos_id = token_to_id["<BOS>"]
    vocab_size = len(token_to_id)

    device = get_device()
    print(f"Using device: {device}")

    model = SignTransformer(vocab_size=vocab_size, pad_token_id=token_to_id["<PAD>"])
    model.to(device)

    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint found at {CHECKPOINT_PATH} -- run training first.")
        return

    dummy_optimizer = torch.optim.AdamW(model.parameters())
    step = load_checkpoint(model, dummy_optimizer, CHECKPOINT_PATH, device=device)
    print(f"Loaded checkpoint from step {step}")
    model.eval()

    print("\n--- Config B: hands + face + body (as trained) ---")
    summary_b, results_b = run_config(
        model, samples, token_to_id, id_to_token, bos_id, device, mask_pose=False, vocab_size=vocab_size
    )
    for k, v in summary_b.items():
        print(f"  {k}: {v}")

    print("\n--- Config A: hands + face only (pose masked at inference) ---")
    summary_a, results_a = run_config(
        model, samples, token_to_id, id_to_token, bos_id, device, mask_pose=True, vocab_size=vocab_size
    )
    for k, v in summary_a.items():
        print(f"  {k}: {v}")

    print("\n==========================")
    print("Channel Check Result (Rank/Probability-Sensitive Metrics)")
    print("==========================")
    print(f"{'Metric':<32} {'Config A':>12} {'Config B':>12} {'B - A':>10}")
    for key in ["mean_true_gloss_probability", "mean_rank", "mean_reciprocal_rank",
                "top1_accuracy", f"top{TOP_K}_accuracy"]:
        a_val = summary_a[key]
        b_val = summary_b[key]
        diff = b_val - a_val
        print(f"{key:<32} {a_val:>12.6f} {b_val:>12.6f} {diff:>+10.6f}")

    print(
        "\nMETHOD: lightweight probe (pose masked at inference on the "
        "trained model), not a full separate-retrain comparison. Metrics "
        "are rank/probability-based rather than top-1 argmax because "
        "diagnose_gloss_position.py confirmed the model is under-trained "
        "for top-1 accuracy to be a meaningful signal yet (see checkpoint "
        f"step {step}). Document both the method and this metric choice "
        "alongside the numbers per the Week 3 plan."
    )

    out_path = Path("logs") / "channel_check_results_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "lightweight_probe_rank_sensitive",
            "checkpoint_step": step,
            "config_a_summary": summary_a,
            "config_b_summary": summary_b,
            "config_a_results": results_a,
            "config_b_results": results_b,
        }, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()