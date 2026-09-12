"""
scripts/battison_heuristic_labels.py

Day 45 (fallback path): Person D's real symmetric/asymmetric split has
NOT arrived as of this writing (confirmed via direct search of
data/processed/asllrp/{train,val,test} -- no separate annotation split
exists anywhere in the repo; escalated to the team on Day 41, trigger
date EOD Fri 2026-08-28 has passed with no response). Per Day 41's
pre-agreed fallback and scripts/ablation_score_bimanual_gate.py's own
"PENDING (b)" note, this builds the Battison's Dominance/Symmetry
Conditions heuristic labeling function next.

--- WHAT THIS ACTUALLY MEASURES (read before trusting the output) ---
Real Battison's conditions are defined over HANDSHAPE:
  - Symmetry Condition: if both hands move independently, they must
    share handshape, orientation, and (mirrored/in-phase) movement.
  - Dominance Condition: if handshapes differ, the non-dominant hand is
    passive and drawn from a small unmarked-handshape inventory.
This pipeline has NO handshape classifier -- only raw MediaPipe landmark
coordinates and per-frame hand-presence flags. This script is therefore
a GEOMETRIC PROXY, not a real implementation of Battison's conditions:
it approximates "symmetric" as "both hands substantially present AND
contributing comparable motion energy", and "asymmetric" as "one hand
is mostly absent, or present but contributing much less motion than the
other" -- a movement/presence-based stand-in for the passivity half of
Battison's conditions, with no handshape comparison at all. Document
this as provisional everywhere its output is used.

--- METHOD ---
Per sample (read directly from the real ProcessedSample -- no vocab or
tokenization needed for a purely geometric label, so this does not go
through SignLanguageDataset):
  1. left/right hand presence fraction, from LandmarkPresence.left_hand/
     right_hand -- reused directly from the real preprocessing pipeline,
     not re-derived from zero-coordinate guessing.
  2. If either hand's presence fraction is below PRESENCE_FLOOR: label
     "asymmetric" (a hand barely in frame cannot be co-equally
     articulating -- the cheapest, clearest proxy for the Dominance
     Condition).
  3. Otherwise, per-hand motion energy = summed frame-to-frame Euclidean
     displacement of the hand centroid (mean of its 21 landmarks),
     computed only across consecutive PRESENT frames (a presence gap is
     an occlusion artifact, not real movement, and is not counted).
     balance_ratio = min(L,R) / max(L,R). balance_ratio >=
     BALANCE_THRESHOLD -> "symmetric" (both hands comparably active);
     below -> "asymmetric" (one hand passive despite being present).

PRESENCE_FLOOR=0.5 and BALANCE_THRESHOLD=0.5 are reasonable, inspectable
defaults -- but see "KNOWN LIMITATION" below before trusting this
script's output for anything beyond a plumbing test.

--- KNOWN LIMITATION, NOT RESOLVED (investigated this session) ---
Validated against the (1h)/(2h) marker sanity check below: recall on
expected-asymmetric samples is only ~15% (out of 260 usable samples),
vs. ~95% recall on expected-symmetric -- i.e. this heuristic almost
never predicts "asymmetric" when it plausibly should. A presence-floor
x balance-threshold sweep (9 combinations) failed to find any setting
above ~55% balanced accuracy -- barely better than chance, and the
ORIGINAL defaults above were tied for best in that sweep. Root cause
not isolated: it may be that ordinary MediaPipe tracking jitter on a
genuinely passive hand is large enough, relative to often-modest real
sign movement, that raw motion-energy ratio cannot cleanly separate
"passive" from "active" without denoising (e.g. a low-pass filter, or
an absolute rather than relative motion floor) -- or it may be that the
marker-based check itself is a weak proxy (a single (1h)-marked gloss
inside a 20-gloss utterance doesn't necessarily describe that
utterance's overall bimanual behavior). Do NOT treat this script's
output as a usable provisional labeling until one of those is resolved
-- it is currently a validated-negative result, not a hedge-with-
caveats-and-ship result.

--- SANITY CHECK (a directional signal, NOT a validation -- no ground
truth exists for this heuristic to be checked against) ---
Many ASLLRP raw gloss strings self-report handedness inline, e.g.
"(1h)next-to" (one-handed) vs "(2h)alt.see" (two-handed). This script
reports the agreement rate between this heuristic's label and the
majority (1h)/(2h) marker across each sample's gloss sequence, where
such markers exist. This is a soft, partial signal (one utterance
contains many glosses, not all carry a marker, and a "(2h)" marker
alone does not establish the Symmetry Condition's handshape requirement)
-- reported for honesty, not treated as proof the heuristic is correct.

Output: {video_id: "symmetric"|"asymmetric"} JSON, matching
ablation_score_bimanual_gate.py's load_label_mapping() shape exactly --
that script needs no changes to consume this file via --labels.

Usage:
    uv run python scripts/battison_heuristic_labels.py
    uv run python scripts/battison_heuristic_labels.py --npz-dir data/processed/asllrp/train --out logs/battison_heuristic_labels_PROVISIONAL.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.preprocessing.sample_writer import load_sample

ASLLRP_NPZ_DIR = "data/processed/asllrp/train"  # matches ablation_score_bimanual_gate.py's default
OUT_PATH = "logs/battison_heuristic_labels_PROVISIONAL.json"

PRESENCE_FLOOR = 0.5      # a hand present <50% of frames is not meaningfully engaged
BALANCE_THRESHOLD = 0.5   # min/max motion-energy ratio below this -> one hand passive
EPS = 1e-8

HANDEDNESS_MARKER_RE = re.compile(r"\((1h|2h)\)")


def hand_motion_energy(hand_xyz: np.ndarray, present: np.ndarray) -> float:
    """hand_xyz: (T, 21, 3). present: (T,) bool. Returns MEAN (not summed)
    Euclidean displacement of the hand centroid across consecutive PRESENT
    frames only -- a presence gap (hand briefly out of frame) is not
    counted as motion, since it's an occlusion artifact, not real
    movement. Mean, not sum: two hands with different present-frame
    counts (both above PRESENCE_FLOOR) must be compared per-frame, or the
    hand present for more frames wins on raw accumulated total regardless
    of true activity level -- caught via the (1h)/(2h) sanity check
    scoring WORSE than random chance with a raw sum, root-caused to this."""
    present_idxs = np.flatnonzero(present)
    if len(present_idxs) < 2:
        return 0.0
    centroids = hand_xyz[present_idxs].mean(axis=1)  # (n_present, 3)
    diffs = np.diff(centroids, axis=0)
    return float(np.linalg.norm(diffs, axis=1).mean())


def battison_label(sample) -> str:
    """Returns 'symmetric' or 'asymmetric'. See module docstring for what
    this actually measures -- a movement/presence-based geometric proxy,
    NOT a real handshape-based Battison's-conditions classifier."""
    presence = sample.presence
    T = sample.pose.shape[0]
    left_present_frac = presence.left_hand.mean() if T else 0.0
    right_present_frac = presence.right_hand.mean() if T else 0.0

    if left_present_frac < PRESENCE_FLOOR or right_present_frac < PRESENCE_FLOOR:
        return "asymmetric"

    left_energy = hand_motion_energy(sample.left_hand, presence.left_hand)
    right_energy = hand_motion_energy(sample.right_hand, presence.right_hand)
    balance_ratio = min(left_energy, right_energy) / (max(left_energy, right_energy) + EPS)
    return "symmetric" if balance_ratio >= BALANCE_THRESHOLD else "asymmetric"


def handedness_marker_sanity_check(npz_paths, labels):
    """Directional-only sanity signal -- see module docstring. Not a
    validation; there is no ground truth for this heuristic to be
    checked against. Reports PER-CLASS recall, not just a blended
    agree/disagree rate -- a blended rate is exactly what hid this
    heuristic's actual failure mode (near-zero recall on the asymmetric
    class) when this script was first built and validated."""
    tp = tn = fp = fn = 0  # 'positive' = asymmetric
    no_marker = 0
    for p in npz_paths:
        sample = load_sample(p)
        vid = sample.metadata.video_id
        label = labels.get(vid)
        if label is None:
            continue
        markers = HANDEDNESS_MARKER_RE.findall(sample.metadata.gloss)
        if not markers:
            no_marker += 1
            continue
        n_2h = markers.count("2h")
        n_1h = markers.count("1h")
        if n_2h == n_1h:
            continue  # tied / not decisive for this utterance
        expected = "symmetric" if n_2h > n_1h else "asymmetric"
        if expected == "asymmetric" and label == "asymmetric":
            tp += 1
        elif expected == "symmetric" and label == "symmetric":
            tn += 1
        elif expected == "symmetric" and label == "asymmetric":
            fp += 1
        elif expected == "asymmetric" and label == "symmetric":
            fn += 1
    n_asym, n_sym = tp + fn, tn + fp
    recall_asym = tp / n_asym if n_asym else None
    recall_sym = tn / n_sym if n_sym else None
    balanced_acc = (
        (recall_asym + recall_sym) / 2 if recall_asym is not None and recall_sym is not None else None
    )
    return {
        "n_expected_asymmetric": n_asym, "n_expected_symmetric": n_sym,
        "recall_asymmetric": recall_asym, "recall_symmetric": recall_sym,
        "balanced_accuracy": balanced_acc, "no_usable_marker": no_marker,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-dir", type=str, default=ASLLRP_NPZ_DIR)
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()

    npz_paths = sorted(Path(args.npz_dir).glob("*.npz"))
    if not npz_paths:
        raise RuntimeError(f"No .npz files found in {args.npz_dir}")

    print("=== Battison's-conditions heuristic labeling (PROVISIONAL) ===")
    print(f"Scanning {len(npz_paths)} samples in {args.npz_dir}\n")

    labels = {}
    n_sym, n_asym = 0, 0
    for p in npz_paths:
        sample = load_sample(p)
        label = battison_label(sample)
        labels[sample.metadata.video_id] = label
        if label == "symmetric":
            n_sym += 1
        else:
            n_asym += 1

    print(f"Labeled {len(labels)} samples: {n_sym} symmetric, {n_asym} asymmetric "
          f"({100*n_sym/len(labels):.1f}% / {100*n_asym/len(labels):.1f}%)")

    sanity = handedness_marker_sanity_check(npz_paths, labels)
    print("\nDirectional sanity check vs. inline (1h)/(2h) gloss markers "
          "(soft signal, NOT a validation -- see docstring):")
    print(f"  expected-asymmetric: {sanity['n_expected_asymmetric']}  "
          f"expected-symmetric: {sanity['n_expected_symmetric']}  "
          f"no usable marker: {sanity['no_usable_marker']}")
    if sanity["balanced_accuracy"] is not None:
        print(f"  recall on expected-asymmetric: {sanity['recall_asymmetric']:.3f}")
        print(f"  recall on expected-symmetric:  {sanity['recall_symmetric']:.3f}")
        print(f"  balanced accuracy:             {sanity['balanced_accuracy']:.3f}")
        if sanity["balanced_accuracy"] < 0.65:
            print("\n  *** WARNING: balanced accuracy is close to chance (0.5). A presence-floor x")
            print("  balance-threshold sweep (see module docstring) found no configuration above")
            print("  ~0.55 -- this is NOT a calibration problem fixable by adjusting the constants")
            print("  above. Do not treat this run's labels as a usable provisional signal. ***")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"\nWrote {len(labels)} labels to {args.out} (schema-valid, but see the sanity-check "
          f"warning above before treating this as a usable provisional labeling).")


if __name__ == "__main__":
    main()
