"""
Check WLASL per-gloss download coverage.

Cross-references WLASL_v0.3.json (full metadata: every gloss + every video_id
it should have) against what's actually sitting in data/raw/wlasl/videos/
(or whatever your video dir is) to find glosses that got hit hardest by
missing downloads — not just the aggregate %.

Adjust the two paths below to match your layout, then:
    uv run python scripts/wlasl_acquisition/gloss_coverage_check.py
"""

import json
from collections import defaultdict
from pathlib import Path

# --- adjust these if your paths differ ---
WLASL_JSON = Path("data/raw/wlasl/metadata/WLASL_v0.3.json")
VIDEO_DIRS = [
    Path("data/raw/wlasl/videos"),      # holds files named like 69241.mp4
    Path("data/raw/wlasl/hf_mirror"),   # in case some recovered videos aren't merged in yet
]
VIDEO_EXT = ".mp4"
# ----------------------------------------------

def main():
    if not WLASL_JSON.exists():
        raise SystemExit(f"Can't find {WLASL_JSON} — fix the WLASL_JSON path at top of script.")

    existing_dirs = [d for d in VIDEO_DIRS if d.exists()]
    if not existing_dirs:
        raise SystemExit(f"None of {VIDEO_DIRS} exist — fix VIDEO_DIRS at top of script.")

    with open(WLASL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # set of video_ids we actually have on disk, across all video dirs (fast lookup)
    have_ids = set()
    for d in existing_dirs:
        found = {p.stem for p in d.glob(f"*{VIDEO_EXT}")}
        print(f"  {d}: {len(found)} files")
        have_ids |= found
    print()

    gloss_total = {}
    gloss_have = {}

    for entry in data:
        gloss = entry["gloss"]
        instances = entry.get("instances", [])
        gloss_total[gloss] = len(instances)
        have_count = sum(1 for inst in instances if inst["video_id"] in have_ids)
        gloss_have[gloss] = have_count

    total_expected = sum(gloss_total.values())
    total_have = sum(gloss_have.values())

    buckets = defaultdict(list)  # have_count -> [gloss, ...]
    for gloss, have in gloss_have.items():
        buckets[have].append(gloss)

    zero = buckets.get(0, [])
    one = buckets.get(1, [])
    two = buckets.get(2, [])

    print(f"Total glosses           : {len(gloss_total)}")
    print(f"Total expected videos   : {total_expected}")
    print(f"Total videos on disk    : {total_have}")
    print(f"Overall missing %       : {100 * (1 - total_have / total_expected):.1f}%")
    print()
    print(f"Glosses with 0 videos   : {len(zero)}")
    print(f"Glosses with 1 video    : {len(one)}")
    print(f"Glosses with 2 videos   : {len(two)}")
    print(f"Glosses with 3+ videos  : {len(gloss_total) - len(zero) - len(one) - len(two)}")
    print()

    if zero:
        print(f"--- Glosses with ZERO videos (unusable, {len(zero)} total) ---")
        for g in sorted(zero)[:30]:
            print(f"  {g}  (expected {gloss_total[g]})")
        if len(zero) > 30:
            print(f"  ... and {len(zero) - 30} more")
        print()

    if one:
        print(f"--- Glosses with only 1 video (too thin for train/val split, {len(one)} total) ---")
        for g in sorted(one)[:15]:
            print(f"  {g}  (expected {gloss_total[g]})")
        if len(one) > 15:
            print(f"  ... and {len(one) - 15} more")

    # dump full per-gloss csv for anything you want to inspect further
    out_csv = Path("logs/gloss_coverage.csv")
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("gloss,expected,have,missing_pct\n")
        for gloss in sorted(gloss_total):
            exp = gloss_total[gloss]
            have = gloss_have[gloss]
            pct = 100 * (1 - have / exp) if exp else 0
            f.write(f"{gloss},{exp},{have},{pct:.1f}\n")
    print(f"\nFull per-gloss breakdown written to {out_csv}")


if __name__ == "__main__":
    main()