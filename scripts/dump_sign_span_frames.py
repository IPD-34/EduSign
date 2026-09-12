"""
dump_sign_span_frames.py

Day 31 visual verification: dump real video frames spanning a sign's
DOMINANT_HAND span, so the boundary-label decision can be checked by
eye, not just trusted from XML numbers -- same discipline as Day 4's
check_frame_indexing_batch.py.

Uses cv2.VideoCapture directly rather than the project's internal
video_reader.py, since that module's exact interface wasn't confirmed
here -- this is a one-off inspection script, not part of the training
pipeline, so a standalone implementation is lower-risk than guessing
at an internal API.
"""

import cv2
from pathlib import Path

from src.preprocessing.metadata_readers.asllrp_reader import ASLLRPReader

XML_DIR = "data/raw/asllrp/xml"
VIDEO_DIR = "data/raw/asllrp/videos"
TARGET_UTTERANCE_ID = "8400279"
TARGET_SIGN_LABEL = "ARTICLE"
CONTEXT_FRAMES = 3  # extra frames before/after the span, for context --
                     # so you can see the transition IN and OUT, not
                     # just the span itself

OUTPUT_DIR = Path("data/debug/sign_span_frames")


def main():
    reader = ASLLRPReader(xml_dir=XML_DIR, video_dir=VIDEO_DIR)

    target_utt = None
    for utt in reader:
        if utt.utterance_id == TARGET_UTTERANCE_ID:
            target_utt = utt
            break

    if target_utt is None:
        raise RuntimeError(f"Utterance {TARGET_UTTERANCE_ID} not found")

    if TARGET_SIGN_LABEL not in target_utt.glosses:
        raise RuntimeError(f"Sign {TARGET_SIGN_LABEL} not found in utterance {TARGET_UTTERANCE_ID}")

    sign_idx = target_utt.glosses.index(TARGET_SIGN_LABEL)
    span = target_utt.sign_spans[sign_idx]
    if span is None:
        raise RuntimeError(f"No span found for {TARGET_SIGN_LABEL}")

    start_raw, end_raw = span
    print(f"Utterance: {target_utt.utterance_id}")
    print(f"Sign: {TARGET_SIGN_LABEL} (index {sign_idx} of {len(target_utt.glosses)})")
    print(f"Raw span (1-indexed, as stored in XML): {start_raw}-{end_raw}")
    print(f"Video: {target_utt.video_path}")

    # Convert to 0-indexed for cv2 frame seeking -- same -1 adjustment
    # used everywhere else in this project for ASLLRP's 1-indexed
    # frame numbers (Day 4's empirically-confirmed convention).
    start_0 = start_raw - 1
    end_0 = end_raw - 1

    dump_start = max(0, start_0 - CONTEXT_FRAMES)
    dump_end = end_0 + CONTEXT_FRAMES

    cap = cv2.VideoCapture(str(target_utt.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {target_utt.video_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = []
    for frame_idx in range(dump_start, dump_end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  [WARNING] Could not read frame {frame_idx}")
            continue

        # Mark whether this frame is inside the span, for filename clarity
        in_span = start_0 <= frame_idx <= end_0
        tag = "SPAN" if in_span else "context"
        out_path = OUTPUT_DIR / f"{TARGET_SIGN_LABEL}_frame{frame_idx:04d}_{tag}.png"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)

    cap.release()

    print(f"\nSaved {len(saved)} frames to {OUTPUT_DIR}/")
    print(f"Frames {dump_start}-{start_0 - 1}: context (before span)")
    print(f"Frames {start_0}-{end_0}: SPAN ('{TARGET_SIGN_LABEL}' articulation, per XML)")
    print(f"Frames {end_0 + 1}-{dump_end}: context (after span)")
    print("\nOpen these in order and confirm: does the hand configuration")
    print("actually change/move meaningfully across the SPAN frames,")
    print("settling into a held shape by the last SPAN frame, compared")
    print("to the context frames before/after? That's the real check --")
    print("not just that frames extracted without error.")


if __name__ == "__main__":
    main()