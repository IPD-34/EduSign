"""
channel_check_score.py

Day 17: run both channel configs through the Day 12 checkpoint, compare
accuracy on real, self-recorded footage.

METHOD USED: lightweight probe (per the Week 3 plan's documented options,
not the full separate-retrain comparison). The Day 12 model was already
trained on the full holistic input (pose+face+hands, 1692-dim). Config B
(hands+face+body) is evaluated as-is. Config A (hands+face only) is
evaluated by zeroing the pose channel of the SAME trained model's input
at inference time -- this tests how much the model's predictions degrade
without pose information actually present, which is a fast, honest signal,
but NOT the same as training a model from scratch without pose available.
If this result is close/ambiguous, per the plan's pre-agreed tie-breaker,
default to hands+face-only (simpler scope) and note this was a probe,
not a full retrain comparison, when writing up the decision.

Pose occupies the first 132 dims of the flattened 1692-dim per-frame
feature vector (pose: 33 landmarks x 4 = 132), per feature_utils.py's
flatten order (pose, face, left_hand, right_hand). Verify this ordering
against feature_utils.py before trusting the mask indices below --
if flatten_frame_features() concatenates in a different order, the
POSE_DIM_SLICE below needs to change to match.

Usage:
    uv run python scripts/channel_check_score.py
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

# Pose is the first 132 values of the flattened per-frame feature vector
# (33 landmarks x 4 = 132). VERIFY against feature_utils.py's actual
# concatenation order before trusting this.
POSE_DIM_SLICE = slice(0, 132)


def predict_gloss(model, landmarks, id_to_token, bos_id, eos_id, pad_id, device):
    """Greedy single-step decode: feed <BOS>, take the model's argmax
    prediction as the recognized gloss token. Isolated-sign glosses are
    a single token, so one decode step is sufficient (per Day 9/11's
    target_tokens shape of [BOS, gloss_id, EOS])."""
    landmarks = landmarks.unsqueeze(0).to(device)          # (1, T, feat_dim)
    encoder_mask = torch.ones(1, landmarks.shape[1], dtype=torch.bool, device=device)
    target_tokens = torch.tensor([[bos_id]], device=device)  # (1, 1)

    with torch.no_grad():
        logits = model(landmarks, encoder_mask, target_tokens, target_padding_mask=None)
        # logits: (1, 1, vocab_size) -- prediction for the token after <BOS>
        predicted_id = logits[0, -1, :].argmax().item()

    return id_to_token.get(predicted_id, "<UNK>")


def run_config(model, samples, gloss_col, id_to_token, bos_id, eos_id, pad_id, device, mask_pose):
    correct = 0
    total = 0
    per_sample_results = []

    for video_id, gloss in samples:
        npz_path = CHANNEL_CHECK_PROCESSED_DIR / f"{video_id}.npz"
        if not npz_path.exists():
            continue

        sample = load_sample(npz_path)
        landmarks = flatten_frame_features(sample)
        
        landmarks = torch.from_numpy(landmarks).float()  # (T, 1692) -- adjust if your
                                                      # actual function signature differs

        if mask_pose:
            landmarks = landmarks.clone()
            landmarks[:, POSE_DIM_SLICE] = 0.0

        predicted_gloss = predict_gloss(model, landmarks, id_to_token, bos_id, eos_id, pad_id, device)

        is_correct = predicted_gloss == gloss
        correct += int(is_correct)
        total += 1
        per_sample_results.append({
            "video_id": video_id,
            "true_gloss": gloss,
            "predicted_gloss": predicted_gloss,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total, per_sample_results


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
    pad_id = token_to_id["<PAD>"]
    bos_id = token_to_id["<BOS>"]
    eos_id = token_to_id["<EOS>"]
    vocab_size = len(token_to_id)

    device = get_device()
    print(f"Using device: {device}")

    model = SignTransformer(vocab_size=vocab_size, pad_token_id=pad_id)
    model.to(device)

    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint found at {CHECKPOINT_PATH} -- run Day 12 training first.")
        return

    dummy_optimizer = torch.optim.AdamW(model.parameters())  # load_checkpoint expects one
    step = load_checkpoint(model, dummy_optimizer, CHECKPOINT_PATH, device=device)
    print(f"Loaded checkpoint from step {step}")
    model.eval()

    print("\n--- Config B: hands + face + body (as trained) ---")
    acc_b, total_b, results_b = run_config(
        model, samples, "gloss", id_to_token, bos_id, eos_id, pad_id, device, mask_pose=False
    )
    print(f"Accuracy: {acc_b:.4f} ({int(acc_b*total_b)}/{total_b})")

    print("\n--- Config A: hands + face only (pose masked at inference) ---")
    acc_a, total_a, results_a = run_config(
        model, samples, "gloss", id_to_token, bos_id, eos_id, pad_id, device, mask_pose=True
    )
    print(f"Accuracy: {acc_a:.4f} ({int(acc_a*total_a)}/{total_a})")

    print("\n==========================")
    print("Channel Check Result (Lightweight Probe)")
    print("==========================")
    print(f"Config A (hands+face only):      {acc_a:.4f}")
    print(f"Config B (hands+face+body):      {acc_b:.4f}")
    print(f"Difference (B - A):              {acc_b - acc_a:+.4f}")
    print("\nMETHOD: lightweight probe (pose masked at inference on the")
    print("already-trained Day 12 model), not a full separate-retrain comparison.")
    print("Document this method alongside the numbers per the Week 3 plan.")

    # Save detailed results for the writeup
    out_path = Path("logs") / "channel_check_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "lightweight_probe",
            "config_a_accuracy": acc_a,
            "config_b_accuracy": acc_b,
            "config_a_results": results_a,
            "config_b_results": results_b,
        }, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()