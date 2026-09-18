"""
scripts/asllrp_isolate_signs.py

Converts ASLLRP's existing continuous-utterance landmark data
(data/processed/asllrp/*, 1,234 files, 1,214/1,215 being multi-sign
sequences -- "876 canonical words" there was utterance co-occurrence,
NOT usable per-word training data) into real isolated per-sign clips,
using data ALREADY ON DISK -- no raw video/XML needed, no external
drive required for this step.

WHY THIS WORKS (verified against 40 real files before writing this,
not assumed): SampleMetadata already carries sign_spans (per-sign
frame boundaries, ORIGINAL video frame numbers) and frame_indices
(which original frame number each STORED array position corresponds
to) for every file checked. 39/40 had spans fully within the stored
frame_indices range with a matching sign_spans-to-gloss-token count;
the 1 exception was a span ending exactly 1 frame past the last stored
frame (a benign boundary rounding difference between annotation
systems) -- handled by clamping, not a real gap in the data.

Each isolated sign becomes its own ProcessedSample: metadata.gloss is
the single sign token (not the "|"-joined utterance string), split is
inherited from the parent utterance (same signer/session, so this
doesn't introduce new leakage beyond whatever the utterance-level split
already had), video_id is "{parent}_sign{i}" so isolated clips trace
back to their source utterance.

Output: data/processed/asllrp_isolated/{train,val,test}/*.npz -- a NEW
directory, does not touch or overwrite the existing continuous-utterance
data.
"""
import sys
import json
import csv
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.sample_writer import load_sample, write_sample
from src.preprocessing.sample_schema import SampleMetadata, ProcessedSample, LandmarkPresence

SOURCE_DIR = Path("data/processed/asllrp")
OUT_DIR = Path("data/processed/asllrp_isolated")


def isolate_signs_from_utterance(sample):
    m = sample.metadata
    if m.sign_spans is None or m.frame_indices is None:
        return []
    gloss_tokens = m.gloss.split("|")
    if len(gloss_tokens) != len(m.sign_spans):
        return []  # count mismatch -- skip this utterance rather than guess alignment

    fi = m.frame_indices
    frame_to_idx = {f: i for i, f in enumerate(fi)}
    fi_min, fi_max = fi[0], fi[-1]

    results = []
    for j, (token, span) in enumerate(zip(gloss_tokens, m.sign_spans)):
        if span is None:
            continue
        start, end = span
        start = max(start, fi_min)
        end = min(end, fi_max)  # clamp: verified benign 1-frame-over case, see module docstring
        if start > end:
            continue
        idxs = [frame_to_idx[f] for f in range(start, end + 1) if f in frame_to_idx]
        if not idxs:
            continue
        i0, i1 = min(idxs), max(idxs) + 1

        new_meta = SampleMetadata(
            video_id=f"{m.video_id}_sign{j}",
            gloss=token,
            split=m.split,
            source="asllrp_isolated",
            fps=m.fps,
            num_frames=i1 - i0,
            translation=None,
            frame_indices=fi[i0:i1],
            sign_spans=None,
        )
        new_presence = LandmarkPresence(
            pose=sample.presence.pose[i0:i1],
            left_hand=sample.presence.left_hand[i0:i1],
            right_hand=sample.presence.right_hand[i0:i1],
            face=sample.presence.face[i0:i1],
        )
        results.append(ProcessedSample(
            metadata=new_meta,
            pose=sample.pose[i0:i1], face=sample.face[i0:i1],
            left_hand=sample.left_hand[i0:i1], right_hand=sample.right_hand[i0:i1],
            presence=new_presence,
        ))
    return results


def main():
    all_paths = []
    for split in ("train", "val", "test"):
        all_paths.extend((SOURCE_DIR / split).glob("*.npz"))

    print(f"Source utterances: {len(all_paths)}")

    n_isolated = 0
    n_skipped_utterances = 0
    gloss_counts = Counter()
    manifest_rows = []

    for p in all_paths:
        sample = load_sample(p)
        isolated = isolate_signs_from_utterance(sample)
        if not isolated:
            n_skipped_utterances += 1
            continue
        for iso in isolated:
            out_path = OUT_DIR / iso.metadata.split / f"{iso.metadata.video_id}.npz"
            write_sample(iso, out_path)
            n_isolated += 1
            gloss_counts[iso.metadata.gloss] += 1
            manifest_rows.append({
                "video_id": iso.metadata.video_id, "gloss": iso.metadata.gloss,
                "split": iso.metadata.split, "num_frames": iso.metadata.num_frames,
                "parent_utterance": sample.metadata.video_id,
            })
        if len(manifest_rows) % 2000 < len(isolated):
            print(f"  ...{n_isolated} isolated signs written so far")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n=== DONE ===")
    print(f"Utterances processed: {len(all_paths)}")
    print(f"Utterances skipped (missing spans/frame_indices/count mismatch): {n_skipped_utterances}")
    print(f"Isolated sign clips written: {n_isolated}")
    print(f"Distinct isolated-sign vocabulary: {len(gloss_counts)}")

    counts = list(gloss_counts.values())
    print(f"Per-word counts: min={min(counts)} max={max(counts)} mean={sum(counts)/len(counts):.2f}")
    n_at_least_10 = sum(1 for c in counts if c >= 10)
    print(f"Words with >=10 real isolated examples: {n_at_least_10}/{len(counts)}")

    with open("logs/asllrp_isolated_gloss_counts.json", "w") as f:
        json.dump(dict(gloss_counts), f)
    print("Saved logs/asllrp_isolated_gloss_counts.json")


if __name__ == "__main__":
    main()
