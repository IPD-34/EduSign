"""
audit_asllrp_vocab.py

Day 29 follow-up: the merged-vocab build reported ZERO overlap between
WLASL and ASLLRP vocabularies, and a spot-check of ASLLRP-only tokens
showed SignStream linguistic annotation conventions (classifier
predicates with embedded descriptive strings, fingerspelling prefixes,
morphological markers, numbered variants) -- not plain word glosses
like WLASL's. Before trusting vocab_merged.json for Day 30's CTC head,
this script checks two things:

  1. Is the zero-overlap result explained by simple case mismatch
     (WLASL lowercase vs. ASLLRP uppercase), or does real overlap
     stay at zero even case-insensitively?
  2. What's the actual frequency distribution of ASLLRP gloss tokens
     across the corpus? If a large fraction appear only once or twice,
     they cannot be meaningfully learned via CTC supervision and
     inflating the vocab with them is close to pure noise.

Usage:
    uv run python audit_asllrp_vocab.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessing.config import ASLLRP_XML_DIR, ASLLRP_VIDEO_DIR
from src.preprocessing.metadata_readers.asllrp_reader import ASLLRPReader

EXISTING_VOCAB_PATH = Path("data") / "vocab.json"
SPECIAL_TOKENS = {"<PAD>", "<BOS>", "<EOS>", "<UNK>"}

# Rough patterns to flag "non-plain-word" ASLLRP tokens -- annotation
# conventions rather than simple glosses. Not exhaustive, just enough
# to get a real count for a decision.
ANNOTATION_PATTERNS = [
    (r'^DCL:', "classifier predicate (DCL:)"),
    (r'^[A-Z]{1,3}CL:', "classifier predicate (*CL:)"),
    (r'^fs-', "fingerspelling prefix (fs-)"),
    (r'_\d+$', "numbered variant suffix (_N)"),
    (r'^\(\d+\)', "leading numeric marker ((N))"),
    (r'-\d?p$', "person/possessive marker (-Np)"),
    (r'"', "embedded quoted description"),
]


def classify_token(tok):
    for pattern, label in ANNOTATION_PATTERNS:
        if re.search(pattern, tok):
            return label
    return None


def main():
    with open(EXISTING_VOCAB_PATH, "r", encoding="utf-8") as f:
        wlasl_vocab = json.load(f)
    wlasl_tokens_raw = set(wlasl_vocab.keys()) - SPECIAL_TOKENS
    wlasl_tokens_lower = {t.lower() for t in wlasl_tokens_raw}

    reader = ASLLRPReader(xml_dir=ASLLRP_XML_DIR, video_dir=ASLLRP_VIDEO_DIR)

    token_counter = Counter()  # token -> utterance occurrence count
    for sample in reader.samples():
        for tok in sample.gloss.split("|"):
            tok = tok.strip()
            if tok:
                token_counter[tok] += 1

    asllrp_tokens_raw = set(token_counter.keys())
    asllrp_tokens_lower = {t.lower() for t in asllrp_tokens_raw}

    print("==========================")
    print("1. Case-Sensitivity Check")
    print("==========================")
    overlap_raw = wlasl_tokens_raw & asllrp_tokens_raw
    overlap_lower = wlasl_tokens_lower & asllrp_tokens_lower

    print(f"Overlap (case-sensitive)  : {len(overlap_raw)}")
    print(f"Overlap (case-INsensitive): {len(overlap_lower)}")

    if len(overlap_lower) > len(overlap_raw):
        print(f"\n-> Case mismatch confirmed as a real contributing factor: "
              f"{len(overlap_lower)} tokens overlap once case is ignored.")
        sample_case_overlap = sorted(overlap_lower)[:10]
        print(f"   Sample case-insensitive overlaps: {sample_case_overlap}")
    else:
        print(f"\n-> Case is NOT the explanation -- overlap stays at "
              f"{len(overlap_lower)} even case-insensitively. The vocabularies "
              f"are genuinely almost entirely disjoint in surface form.")

    print()
    print("==========================")
    print("2. ASLLRP Token Frequency Distribution")
    print("==========================")
    total_unique = len(token_counter)
    freq_values = list(token_counter.values())

    once = sum(1 for c in freq_values if c == 1)
    twice_or_fewer = sum(1 for c in freq_values if c <= 2)
    five_or_fewer = sum(1 for c in freq_values if c <= 5)

    print(f"Total unique ASLLRP tokens     : {total_unique}")
    print(f"Appearing exactly once         : {once} ({100*once/total_unique:.1f}%)")
    print(f"Appearing <= 2 times            : {twice_or_fewer} ({100*twice_or_fewer/total_unique:.1f}%)")
    print(f"Appearing <= 5 times            : {five_or_fewer} ({100*five_or_fewer/total_unique:.1f}%)")

    most_common = token_counter.most_common(10)
    print(f"\nTop 10 most frequent ASLLRP tokens:")
    for tok, count in most_common:
        print(f"  {tok!r}: {count}")

    print()
    print("==========================")
    print("3. Annotation-Convention Token Check")
    print("==========================")
    annotation_counts = Counter()
    plain_tokens = []

    for tok in asllrp_tokens_raw:
        label = classify_token(tok)
        if label:
            annotation_counts[label] += 1
        else:
            plain_tokens.append(tok)

    total_annotation = sum(annotation_counts.values())
    print(f"Tokens matching an annotation pattern: {total_annotation}/{total_unique} "
          f"({100*total_annotation/total_unique:.1f}%)")
    for label, count in annotation_counts.most_common():
        print(f"  {label}: {count}")

    print(f"\nTokens NOT matching any annotation pattern (likely plain glosses): "
          f"{len(plain_tokens)}/{total_unique} ({100*len(plain_tokens)/total_unique:.1f}%)")

    print(f"\nSample of 10 'plain' tokens (candidates for real overlap once lowercased):")
    import random
    for tok in random.sample(plain_tokens, min(10, len(plain_tokens))):
        print(f"  {tok!r}  (count={token_counter[tok]})")

    print()
    print("==========================")
    print("RECOMMENDATION")
    print("==========================")
    low_freq_pct = 100 * twice_or_fewer / total_unique
    annotation_pct = 100 * total_annotation / total_unique
    if low_freq_pct > 30 or annotation_pct > 30:
        print("Significant fraction of ASLLRP tokens are either low-frequency")
        print("(unlearnable via CTC) or annotation-convention tokens (not plain")
        print("glosses). Recommend NOT merging the raw token set as-is. Consider:")
        print("  - Lowercasing both vocabularies before merging (cheap, real fix)")
        print("  - Stripping/normalizing annotation markers (_N suffixes, fs- prefix,")
        print("    classifier predicate descriptions) before treating as vocab tokens")
        print("  - Setting a minimum-frequency threshold (e.g. >=3 occurrences) for")
        print("    inclusion as a real trainable token; route everything else to <UNK>")
        print("  - This is a real methodological decision for Week 1, not a quick fix")
    else:
        print("Distribution looks reasonable -- case fix alone may be sufficient.")


if __name__ == "__main__":
    main()