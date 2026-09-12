"""
check_low_coverage_glosses.py

Day 26 follow-up: firms up the HFMirrorReader-drop decision with real
evidence instead of "nothing's changed since Week 1."

Two checks:
  1. Which specific glosses sit at the lowest video coverage (per the
     original Day 3 finding: 0 glosses with 0 videos, 6 glosses with
     only 2 videos, 1994 with 3+)? Named, not just counted.
  2. Do any of those low-coverage glosses look STEM-relevant? A rough,
     manual-list-based check -- not a claim of completeness, just a
     fast signal on whether the coverage gap intersects the project's
     actual motivation (STEM vocabulary support) or is confined to
     everyday/unrelated words.

This does NOT settle whether coverage is "genuinely insufficient" --
that's a Month 2, real-accuracy question. It only answers: is there a
concrete, specific reason to worry sooner, or does the gap look
low-stakes based on what's checkable today?

Usage:
    uv run python check_low_coverage_glosses.py
"""

import json
from collections import defaultdict
from pathlib import Path

from src.preprocessing.config import WLASL_METADATA, WLASL_VIDEO_DIR

# Rough, non-exhaustive STEM-signal wordlist. This is a fast heuristic,
# not a real taxonomy -- if a gloss contains/matches one of these, or
# looks like a technical term on inspection, flag it for a human look
# rather than trusting the keyword match alone.
STEM_HINT_WORDS = {
    "math", "science", "biology", "chemistry", "physics", "engineer",
    "algebra", "geometry", "equation", "formula", "atom", "molecule",
    "cell", "gene", "dna", "computer", "algorithm", "data", "code",
    "circuit", "voltage", "energy", "force", "gravity", "reaction",
    "experiment", "hypothesis", "theorem", "calculus", "vector",
    "matrix", "protein", "enzyme", "bacteria", "virus", "neuron",
    "planet", "orbit", "electron", "proton", "neutron", "carbon",
    "oxygen", "hydrogen", "acid", "base", "solution", "compound",
}


def main():
    with open(WLASL_METADATA, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    coverage = []  # (gloss, n_videos_on_disk, n_total_instances)

    for entry in metadata:
        gloss = entry["gloss"]
        instances = entry["instances"]
        n_total = len(instances)

        n_on_disk = 0
        for inst in instances:
            video_id = str(inst["video_id"])
            video_path = WLASL_VIDEO_DIR / f"{video_id}.mp4"
            if video_path.exists():
                n_on_disk += 1

        coverage.append((gloss, n_on_disk, n_total))

    coverage.sort(key=lambda x: x[1])  # lowest coverage first

    zero = [c for c in coverage if c[1] == 0]
    low = [c for c in coverage if c[1] > 0 and c[1] <= 2]

    print("==========================")
    print("WLASL Low-Coverage Gloss Check")
    print("==========================")
    print(f"Total glosses           : {len(coverage)}")
    print(f"Glosses with 0 videos   : {len(zero)}")
    print(f"Glosses with 1-2 videos : {len(low)}")
    print()

    if zero:
        print("Zero-coverage glosses (should be none, per Day 3 finding -- flag if any appear):")
        for g, n_disk, n_total in zero:
            print(f"  {g}  (0/{n_total} on disk)")
        print()

    print("Low-coverage glosses (<=2 videos on disk), with STEM-relevance flag:")
    print(f"{'Gloss':<25} {'On disk':>8} {'Total':>8}  STEM-flagged?")
    print("-" * 60)

    any_stem_flagged = False
    for gloss, n_disk, n_total in low:
        gloss_lower = gloss.lower()
        is_flagged = any(hint in gloss_lower for hint in STEM_HINT_WORDS)
        if is_flagged:
            any_stem_flagged = True
        flag_str = "*** YES ***" if is_flagged else "no"
        print(f"{gloss:<25} {n_disk:>8} {n_total:>8}  {flag_str}")

    print()
    if any_stem_flagged:
        print("RESULT: At least one low-coverage gloss matched the STEM heuristic.")
        print("        -> Do NOT close the HFMirrorReader decision on 'nothing changed' alone.")
        print("           Manually inspect the flagged gloss(es) above and consider whether")
        print("           HF mirror supplementation is worth it specifically for those terms.")
    else:
        print("RESULT: No low-coverage gloss matched the STEM heuristic (keyword-based, not exhaustive).")
        print("        -> Consistent with keeping HFMirrorReader dropped for now.")
        print("        -> NOTE: this is a rough keyword check, not a guarantee -- if any gloss")
        print("           above looks STEM-relevant on manual read despite not matching a keyword,")
        print("           treat it the same as a flagged match.")


if __name__ == "__main__":
    main()