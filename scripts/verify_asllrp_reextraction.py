"""
verify_asllrp_reextraction.py

Day 32 verification: after re-running run_asllrp_preprocessing_parallel.py
with the frame_indices / sign_spans schema addition, confirm the new
fields actually landed correctly on real, re-extracted .npz files --
not just that the run finished without an exception.

Checks, per sample:
  1. frame_indices exists, is a real list, and len(frame_indices) == num_frames
     (frame_indices is the raw video frame numbers kept after
     select_frame_indices()'s trim/resample/cap -- it must be exactly
     one entry per sampled timestep, or Day 32's boundary-target
     remapping will silently misalign).
  2. sign_spans exists, is a real list, and its length matches the
     number of glosses (gloss.split("|")) -- one entry per sign,
     same order, same as the in-memory ASLLRPUtterance contract.
  3. Each sign_spans entry is either null (JSON) / None (Python), or a
     2-element [start, end] with start <= end and both > 0 -- catches
     a malformed span silently surviving the JSON round-trip.
  4. frame_indices is monotonically non-decreasing and within
     plausible bounds (>= 0) -- a basic sanity check that this wasn't
     accidentally populated with garbage.

Reports real counts across all sampled files, not just pass/fail on
one -- same discipline as every other verification pass this project
has run (Day 20's 144-clip channel check, Day 31's 346-sign audit).
"""

import json
import random
from pathlib import Path

import numpy as np

ASLLRP_PROCESSED_DIR = Path("data/processed/asllrp")
SAMPLE_SIZE = 20  # matches the --limit 20 smoke test; bump this after
                   # the full corpus re-run to spot-check more broadly

random.seed(32)


def check_one(npz_path: Path) -> dict:
    result = {
        "path": npz_path.name,
        "ok": True,
        "issues": [],
    }

    with np.load(npz_path, allow_pickle=False) as data:
        if "metadata" not in data:
            result["ok"] = False
            result["issues"].append("no metadata key in .npz at all")
            return result

        meta = json.loads(str(data["metadata"]))

    num_frames = meta.get("num_frames")
    gloss = meta.get("gloss", "")
    num_glosses = len(gloss.split("|")) if gloss else 0

    frame_indices = meta.get("frame_indices")
    sign_spans = meta.get("sign_spans")

    # --- Check 1: frame_indices ---
    if frame_indices is None:
        result["ok"] = False
        result["issues"].append("frame_indices is None (field missing or not populated)")
    elif not isinstance(frame_indices, list):
        result["ok"] = False
        result["issues"].append(f"frame_indices is not a list: {type(frame_indices)}")
    elif len(frame_indices) != num_frames:
        result["ok"] = False
        result["issues"].append(
            f"len(frame_indices)={len(frame_indices)} != num_frames={num_frames}"
        )
    else:
        # Check 4: monotonic non-decreasing, non-negative
        if any(f < 0 for f in frame_indices):
            result["ok"] = False
            result["issues"].append("frame_indices contains negative values")
        if any(a > b for a, b in zip(frame_indices, frame_indices[1:])):
            result["ok"] = False
            result["issues"].append("frame_indices is not monotonically non-decreasing")

    # --- Check 2: sign_spans ---
    if sign_spans is None:
        result["ok"] = False
        result["issues"].append("sign_spans is None (field missing or not populated)")
    elif not isinstance(sign_spans, list):
        result["ok"] = False
        result["issues"].append(f"sign_spans is not a list: {type(sign_spans)}")
    elif len(sign_spans) != num_glosses:
        result["ok"] = False
        result["issues"].append(
            f"len(sign_spans)={len(sign_spans)} != num_glosses={num_glosses} "
            f"(gloss field: '{gloss}')"
        )
    else:
        # Check 3: each entry is null or a valid [start, end]
        for i, span in enumerate(sign_spans):
            if span is None:
                continue
            if not isinstance(span, list) or len(span) != 2:
                result["ok"] = False
                result["issues"].append(f"sign_spans[{i}] malformed: {span!r}")
                continue
            s, e = span
            if not (isinstance(s, int) and isinstance(e, int)):
                result["ok"] = False
                result["issues"].append(f"sign_spans[{i}] non-integer values: {span!r}")
                continue
            if s <= 0 or e < s:
                result["ok"] = False
                result["issues"].append(f"sign_spans[{i}] invalid range: {span!r}")

    return result


def main():
    all_npz = list(ASLLRP_PROCESSED_DIR.rglob("*.npz"))
    print(f"{len(all_npz)} total .npz files found under {ASLLRP_PROCESSED_DIR}")

    if not all_npz:
        print("Nothing to check -- did the re-extraction run finish? "
              "Confirm the old folder was moved aside first.")
        return

    sample = random.sample(all_npz, min(SAMPLE_SIZE, len(all_npz)))
    print(f"Checking {len(sample)} sampled files...\n")

    results = [check_one(p) for p in sample]

    passed = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    print(f"=== Results: {len(passed)}/{len(results)} passed ===\n")

    if failed:
        print(f"{len(failed)} FAILED:")
        for r in failed:
            print(f"  {r['path']}:")
            for issue in r["issues"]:
                print(f"    - {issue}")
        print()

    # Print one real, fully-detailed example so you can eyeball actual
    # values, not just pass/fail -- same as every prior "verified by
    # eye" step in this project.
    if passed:
        example_path = ASLLRP_PROCESSED_DIR.rglob("*.npz")
        example_path = next(p for p in sample if any(r["path"] == p.name and r["ok"] for r in results))
        with np.load(example_path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
        print("=== Example passing file, full detail ===")
        print(f"video_id: {meta['video_id']}")
        print(f"gloss: {meta['gloss']}")
        print(f"num_frames: {meta['num_frames']}")
        print(f"frame_indices (first 10): {meta['frame_indices'][:10]}")
        print(f"frame_indices (last 5): {meta['frame_indices'][-5:]}")
        print(f"sign_spans: {meta['sign_spans']}")

    if len(passed) == len(results):
        print("\n[OK] All sampled files have valid frame_indices and sign_spans.")
    else:
        print(f"\n[ATTENTION] {len(failed)}/{len(results)} files failed -- "
              f"do not proceed to boundary-target wiring until this is resolved.")


if __name__ == "__main__":
    main()