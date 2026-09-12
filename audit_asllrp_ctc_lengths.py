"""
audit_asllrp_ctc_lengths.py

Day 36: corpus-wide CTC length-mismatch audit, deferred from Day 33.

CTC requires each sample's input sequence (encoder timesteps, T) to be
long enough to align to its target gloss sequence. The simple
necessary condition is T >= target_length (BOS/EOS stripped), but
CTC's alignment also needs an explicit blank between any two
consecutive REPEATED gloss labels (e.g. gloss sequence [X, X] cannot
align to a 2-frame input without a blank between the two X's) -- so
the real minimum required length is:

    required_T = target_length + num_consecutive_repeats

This mirrors combined_loss.py's own note: with zero_infinity=False, an
under-length sample raises/produces an inf CTC loss -- "usually
input_length < target_length" (the simple case), but the repeat-aware
case is a real, if rarer, second cause and worth checking explicitly
rather than assumed away.

Runs across all three splits (train/val/test = 990/122/122 = 1234
total, matching the Day 33/36 "0/1234" figure) using the exact same
SignLanguageDataset + gloss-to-ID path real training uses -- not a
separate ad hoc length computation -- so a "0 violations" result here
is trustworthy evidence about the real training pipeline, not just a
side calculation that could silently drift from it.

ASSUMPTION, stated explicitly: this treats encoder_attention_mask's
real-frame count (== landmarks.shape[0] per sample, pre-batching) as
the CTC input length, matching how combined_loss.py and
compute_val_ctc_loss() both compute input_lengths. If forward_ctc /
the encoder does any temporal downsampling (stride, pooling) between
raw frames and the CTC head's output timesteps, this assumption is
wrong and T below needs to be replaced with that downsampled length
instead. Worth a quick check against sign_transformer.py before
trusting a "0 violations" result.

Usage:
    uv run python audit_asllrp_ctc_lengths.py
"""

from pathlib import Path

from src.models.dataset import SignLanguageDataset

VOCAB_MERGED_PATH = "data/vocab_merged.json"
ASLLRP_TOKEN_TO_ID_PATH = "data/asllrp_token_to_id.json"

SPLITS = {
    "train": "data/processed/asllrp/train",
    "val": "data/processed/asllrp/val",
    "test": "data/processed/asllrp/test",
}


def required_length(gloss_ids):
    """Minimum CTC input length for this target sequence: one timestep
    per real token, plus one extra (blank) timestep for every place two
    consecutive tokens repeat -- CTC cannot collapse two adjacent
    identical labels without a blank between them."""
    if not gloss_ids:
        return 0
    repeats = sum(
        1 for i in range(1, len(gloss_ids)) if gloss_ids[i] == gloss_ids[i - 1]
    )
    return len(gloss_ids) + repeats


def audit_split(name, npz_dir):
    if not Path(npz_dir).exists():
        print(f"[SKIP] {name}: {npz_dir} does not exist")
        return 0, 0

    dataset = SignLanguageDataset(
        npz_dir=npz_dir,
        vocab_path=VOCAB_MERGED_PATH,
        asllrp_token_to_id_path=ASLLRP_TOKEN_TO_ID_PATH,
    )

    violations = []
    for i in range(len(dataset)):
        item = dataset[i]
        T = item["landmarks"].shape[0]
        # target_tokens is [BOS, gloss_ids..., EOS] -- strip both ends,
        # same convention combined_loss.py's strip_special_tokens() uses.
        gloss_ids = item["target_tokens"][1:-1].tolist()
        req = required_length(gloss_ids)

        if T < req:
            violations.append({
                "video_id": item["video_id"],
                "T": T,
                "target_length": len(gloss_ids),
                "required_length": req,
                "deficit": req - T,
            })

    total = len(dataset)
    print(f"\n=== {name}: {len(violations)}/{total} violations ===")
    for v in violations[:20]:
        print(
            f"  {v['video_id']}: T={v['T']}, target_length={v['target_length']}, "
            f"required(with repeats)={v['required_length']}, deficit={v['deficit']}"
        )
    if len(violations) > 20:
        print(f"  ... and {len(violations) - 20} more")

    return len(violations), total


def main():
    total_violations = 0
    total_samples = 0

    for name, npz_dir in SPLITS.items():
        v, n = audit_split(name, npz_dir)
        total_violations += v
        total_samples += n

    print("\n==========================")
    print("CTC LENGTH-MISMATCH AUDIT -- SUMMARY")
    print("==========================")
    print(f"Total violations: {total_violations}/{total_samples}")
    if total_violations == 0:
        print("Clean -- every sample's input length is sufficient for its "
              "target sequence, including the repeat-aware minimum.")
    else:
        print("Non-zero violations found. These samples produce inf CTC "
              "loss under zero_infinity=False and need an explicit handling "
              "decision (exclude from training vs. raise "
              "MAX_SEQUENCE_LENGTH vs. something else) before trusting "
              "further training on the affected samples.")


if __name__ == "__main__":
    main()