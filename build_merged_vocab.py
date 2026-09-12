"""
build_merged_vocab.py (REVISED)

Day 29 (Month 2, Week 1), revised after audit_asllrp_vocab.py's findings.

WHY REVISED: the first version merged every unique raw ASLLRP gloss
string as its own vocab token. The audit showed this was wrong in two
different directions at once:

  1. Zero overlap with WLASL was mostly a CASE bug (515 tokens overlap
     once lowercased -- e.g. 'friend', 'future', 'not').
  2. Blanket-flagging "annotation-looking" tokens as noise was ALSO
     wrong -- the highest-frequency ASLLRP tokens (IX-1p: 483,
     IX-3p:i: 363, IX-loc:i: 216, part:indef: 153, IX-2p: 118, ...)
     are indexing/pronoun/grammatical forms, i.e. real, highly
     learnable, CORE vocabulary -- not noise to exclude.

This version treats different token categories differently, instead of
one blanket rule:

  - Indexing/grammatical forms (IX-*, part:indef, etc.): kept AS-IS,
    exactly as they appear. These are real, frequent, learnable tokens.
  - Numbered variants of a base lexeme (e.g. COMPUTER_5, ALL-NIGHT_3):
    the trailing _N suffix is stripped and collapsed into the base
    token, increasing that token's effective frequency rather than
    fragmenting it into near-duplicate rare tokens.
  - Plain lexical tokens: lowercased, then merged case-insensitively
    against WLASL's existing vocab (reusing WLASL's existing token ID
    where there's a real match).
  - Fingerspelling (fs-*, ns-fs-*): routed to a single <FINGERSPELL>
    placeholder token, not treated as individual vocab entries -- these
    are spelled proper nouns, unbounded in principle, not a fixed
    vocabulary. This is a documented, acknowledged system limitation,
    not something this script tries to solve.
  - Classifier predicates (*CL:, DCL: ..., including embedded quoted
    descriptions): routed to a single <CLASSIFIER> placeholder for the
    same reason -- productive/unbounded, not enumerable as fixed vocab.
  - Any remaining lexical token appearing fewer than MIN_FREQUENCY
    times (after the above normalization) is excluded from the vocab
    and will resolve to <UNK> at training time -- cannot be learned
    from too few examples via CTC regardless.

OUTPUTS:
  - data/vocab_merged.json          token -> id (the actual training vocab)
  - data/asllrp_token_to_id.json    RAW ASLLRP gloss string -> final id
                                     (precomputed lookup for training code;
                                     avoids re-deriving normalization logic
                                     at train time)

Usage:
    uv run python build_merged_vocab.py
    uv run python build_merged_vocab.py --min-freq 5
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessing.config import ASLLRP_XML_DIR, ASLLRP_VIDEO_DIR
from src.preprocessing.metadata_readers.asllrp_reader import ASLLRPReader

EXISTING_VOCAB_PATH = Path("data") / "vocab.json"
MERGED_VOCAB_PATH = Path("data") / "vocab_merged.json"
TOKEN_MAP_PATH = Path("data") / "asllrp_token_to_id.json"

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
NEW_PLACEHOLDER_TOKENS = ["<FINGERSPELL>", "<CLASSIFIER>"]

FINGERSPELL_PATTERN = re.compile(r'^(ns-)?fs-', re.IGNORECASE)
CLASSIFIER_PATTERN = re.compile(r'^[A-Za-z]{1,3}CL:')
VARIANT_SUFFIX_PATTERN = re.compile(r'_\d+$')


def normalize_token(raw_tok: str):
    """Returns (category, key) for a single raw ASLLRP gloss token.

    category in {"fingerspell", "classifier", "lexical"}.
    key is what gets used for vocab lookup/frequency counting:
      - fingerspell/classifier -> the placeholder string itself
      - lexical -> lowercased, numbered-variant-suffix-stripped form
    """
    if FINGERSPELL_PATTERN.match(raw_tok):
        return "fingerspell", "<FINGERSPELL>"
    if CLASSIFIER_PATTERN.match(raw_tok):
        return "classifier", "<CLASSIFIER>"

    # Lexical: strip trailing numbered-variant suffix, then lowercase.
    base = VARIANT_SUFFIX_PATTERN.sub("", raw_tok)
    return "lexical", base.lower()


def collect_asllrp_raw_tokens():
    reader = ASLLRPReader(xml_dir=ASLLRP_XML_DIR, video_dir=ASLLRP_VIDEO_DIR)
    raw_token_counter = Counter()
    utterance_count = 0

    for sample in reader.samples():
        utterance_count += 1
        for tok in sample.gloss.split("|"):
            tok = tok.strip()
            if tok:
                raw_token_counter[tok] += 1

    print(f"[ASLLRPReader] {utterance_count} utterances scanned")
    return raw_token_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-freq", type=int, default=3,
                         help="Minimum occurrence count (after normalization) "
                              "for a lexical token to be kept as a real vocab "
                              "entry, rather than falling back to <UNK>.")
    args = parser.parse_args()
    min_freq = args.min_freq

    with open(EXISTING_VOCAB_PATH, "r", encoding="utf-8") as f:
        wlasl_vocab = json.load(f)  # token -> id, WLASL tokens already lowercase

    wlasl_lexical = {t: i for t, i in wlasl_vocab.items() if t not in SPECIAL_TOKENS}
    print(f"WLASL vocab (excluding special tokens): {len(wlasl_lexical)} tokens\n")

    raw_token_counter = collect_asllrp_raw_tokens()
    print(f"Total unique raw ASLLRP tokens: {len(raw_token_counter)}\n")

    # --- Categorize every raw token, aggregate lexical frequency post-normalization ---
    raw_to_category = {}
    raw_to_key = {}
    lexical_freq = Counter()

    category_raw_counts = Counter()  # for reporting: how many RAW tokens per category

    for raw_tok, count in raw_token_counter.items():
        category, key = normalize_token(raw_tok)
        raw_to_category[raw_tok] = category
        raw_to_key[raw_tok] = key
        category_raw_counts[category] += 1
        if category == "lexical":
            lexical_freq[key] += count  # aggregate occurrences across variant forms

    print("==========================")
    print("Category Breakdown (raw token counts)")
    print("==========================")
    for cat in ["lexical", "fingerspell", "classifier"]:
        print(f"  {cat:<12}: {category_raw_counts[cat]} raw tokens")
    print()

    print(f"After normalization, {len(lexical_freq)} unique LEXICAL keys "
          f"(post lowercase + variant-suffix collapse)\n")

    # --- Build the merged vocab ---
    merged_vocab = dict(wlasl_vocab)  # preserves WLASL IDs and special tokens
    next_id = max(wlasl_vocab.values()) + 1

    for placeholder in NEW_PLACEHOLDER_TOKENS:
        if placeholder not in merged_vocab:
            merged_vocab[placeholder] = next_id
            next_id += 1

    reused_from_wlasl = 0
    new_lexical_added = 0
    below_threshold_excluded = 0
    below_threshold_examples = []

    lexical_key_to_id = {}

    for key, freq in lexical_freq.items():
        if key in wlasl_lexical:
            lexical_key_to_id[key] = wlasl_lexical[key]
            reused_from_wlasl += 1
        elif freq >= min_freq:
            merged_vocab[key] = next_id
            lexical_key_to_id[key] = next_id
            next_id += 1
            new_lexical_added += 1
        else:
            below_threshold_excluded += 1
            if len(below_threshold_examples) < 15:
                below_threshold_examples.append((key, freq))
            # not added to vocab -- will resolve to <UNK> via the token map below

    print("==========================")
    print("Lexical Merge Result")
    print("==========================")
    print(f"Reused existing WLASL ID (case-insensitive match): {reused_from_wlasl}")
    print(f"New lexical tokens added (freq >= {min_freq})     : {new_lexical_added}")
    print(f"Excluded, below frequency threshold (-> <UNK>)     : {below_threshold_excluded}")
    print()
    print(f"Sample of excluded low-frequency tokens (key, freq):")
    for key, freq in below_threshold_examples:
        print(f"  {key!r}: {freq}")
    print()

    # --- Build the raw-token -> final-id lookup for training code ---
    unk_id = merged_vocab["<UNK>"]
    asllrp_token_to_id = {}

    for raw_tok, category in raw_to_category.items():
        if category in ("fingerspell", "classifier"):
            asllrp_token_to_id[raw_tok] = merged_vocab[raw_to_key[raw_tok]]
        else:  # lexical
            key = raw_to_key[raw_tok]
            asllrp_token_to_id[raw_tok] = lexical_key_to_id.get(key, unk_id)

    print("==========================")
    print("Final Merged Vocab Summary")
    print("==========================")
    print(f"Total vocab size: {len(merged_vocab)}")
    print(f"  WLASL originals (incl. special tokens): {len(wlasl_vocab)}")
    print(f"  New placeholders (<FINGERSPELL>, <CLASSIFIER>): {len(NEW_PLACEHOLDER_TOKENS)}")
    print(f"  New lexical tokens: {new_lexical_added}")
    print()

    # --- Sanity samples ---
    print("Sample of 10 IX-*/grammatical-form tokens (should be kept AS-IS, unmodified):")
    ix_like = [t for t in raw_token_counter if t.upper().startswith("IX-") or t == "part:indef"]
    for tok in sorted(ix_like, key=lambda t: -raw_token_counter[t])[:10]:
        final_id = asllrp_token_to_id[tok]
        print(f"  {tok!r} (raw freq={raw_token_counter[tok]}) -> vocab id {final_id} "
              f"(in vocab as {'<UNK>' if final_id == unk_id else 'a real token'})")
    print()

    print("Sample of 10 fingerspelling tokens (should route to <FINGERSPELL>):")
    fs_like = [t for t, c in raw_to_category.items() if c == "fingerspell"]
    for tok in random.sample(fs_like, min(10, len(fs_like))):
        print(f"  {tok!r} -> id {asllrp_token_to_id[tok]} (== <FINGERSPELL> id: "
              f"{asllrp_token_to_id[tok] == merged_vocab['<FINGERSPELL>']})")
    print()

    print("Sample of 10 classifier-predicate tokens (should route to <CLASSIFIER>):")
    cl_like = [t for t, c in raw_to_category.items() if c == "classifier"]
    for tok in random.sample(cl_like, min(10, len(cl_like))):
        print(f"  {tok!r} -> id {asllrp_token_to_id[tok]} (== <CLASSIFIER> id: "
              f"{asllrp_token_to_id[tok] == merged_vocab['<CLASSIFIER>']})")
    print()

    # --- Write outputs ---
    with open(MERGED_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_vocab, f, indent=2, ensure_ascii=False)
    with open(TOKEN_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(asllrp_token_to_id, f, indent=2, ensure_ascii=False)

    print(f"Merged vocab written to      {MERGED_VOCAB_PATH}")
    print(f"Raw-token lookup written to  {TOKEN_MAP_PATH}")
    print(f"Original vocab.json left untouched at {EXISTING_VOCAB_PATH}")


if __name__ == "__main__":
    main()