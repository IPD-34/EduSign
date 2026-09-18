"""
scripts/pseudo_gloss_groq_validation.py

Small-scale validation of the "LLM-generated pseudo-gloss" technique
(Guo, Li & Cohn, NeurIPS 2025, arXiv:2505.15438) for How2Sign: given a
few real English-sentence -> ASL-gloss-sequence example pairs
(in-context learning), ask an LLM (Groq-hosted Llama 3.3 70B Versatile)
to generate approximate pseudo-gloss sequences for a SMALL sample
(30-50) of real How2Sign train sentences. This is validation-only --
NOT run against the full 31,165-sentence corpus.

Loads GROQ_API_KEY from the project-root .env file via python-dotenv
(installed but never previously wired up anywhere in this project --
confirmed by grep before adding this).
"""
import sys
import os
import csv
import json
import random
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

# llama-3.3-70b-versatile (originally requested) no longer exists on
# Groq's platform -- confirmed via client.models.list() against this
# account, not assumed. Substituted the largest general-purpose
# instruction-following model actually available today: openai/gpt-oss-120b
# (120B, vs. the next options being 27B Qwen models or Groq's own
# tool-calling "compound" system, not a plain single model suited to a
# controlled few-shot completion task).
MODEL = "openai/gpt-oss-120b"
HOW2SIGN_TRAIN_CSV = (
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\how2sign_inspect\how2sign_holistic_features"
    r"\metadata\how2sign_train.csv"
)
SAMPLE_SEED = 42
SAMPLE_SIZE = 40
RATE_LIMIT_DELAY_SECONDS = 2.5  # 30 req/min free-tier limit -> stay comfortably under it

OUT_JSON = "logs/pseudo_gloss_groq_validation_results_v3_raw.json"
OUT_JSON_NORMALIZED = "logs/pseudo_gloss_groq_validation_results_v3_normalized.json"


def get_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not found in os.environ even after load_dotenv() -- check .env location/format.")
    return Groq(api_key=key)


def test_api_call(client):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly the word: OK"}],
        max_tokens=500,
        temperature=0,
        reasoning_effort="low",
    )
    return resp.choices[0].message.content.strip()


# --- Few-shot examples -----------------------------------------------
# SOURCE CHECK, CORRECTED (see report): an earlier draft of this script
# wrongly assumed ASLLRP had no usable English translation, based on
# `SampleMetadata.translation` being an Optional field without actually
# checking whether it's ever POPULATED. Direct check against real data
# found the opposite: 50/50 sampled ASLLRP train .npz files have a
# real, non-empty `translation` (e.g. "My cat only eats cat food.",
# "She is fine with broccoli, but she refuses to eat spinach."). ASLLRP
# DOES supply real English-sentence-to-gloss pairs directly, per the
# task's own preferred source -- these 6 are real utterance-level
# (translation, gloss) pairs pulled unmodified from
# data/processed/asllrp/train/*.npz (gloss field, "|"-joined ->
# space-joined here, otherwise byte-identical to what's stored).
#
# SELECTION CRITERION, STATED EXPLICITLY: out of 506 candidate pairs
# with a clean (no dialogue quotes, single-sentence, 15-90 chars)
# translation and a short (2-8 token) gloss, these 6 were hand-picked
# for being the LEAST encumbered by ASLLRP's own research-transcription
# notation (classifier predicates like DCL:B-L"...", repetition marks,
# handshape codes) -- not because that notation is wrong, but because
# it's specific to ASLLRP's own annotation convention and not something
# an LLM should be expected to imitate faithfully from 6 examples, nor
# something a downstream CTC vocabulary could easily use as discrete
# tokens. The two ASLLRP-specific conventions that DO carry over
# cleanly -- fs- for fingerspelling, ns- for name-signs -- are kept
# deliberately (examples 2 and 4 below), since both are genuine,
# useful ASL phenomena How2Sign sentences will also need.
# NORMALIZATION PASS, DOCUMENTED EXPLICITLY: the stricter system prompt
# below (from the user) tells the model "match the exact form already
# used in the examples" for possessives/pronouns/tense markers -- but
# checked directly, the RAW ASLLRP examples did NOT already satisfy
# that premise. The 1st-person possessive concept appeared as THREE
# different surface forms across just 6 examples (POSS-1p, POSS-1p+,
# (G/Q)POSS-1p+), and the 1st-person pronoun appeared as IX-1p
# (capitalized) even though the new prompt's own "Known high-risk
# categories" text specifies lowercase ix-1p as canonical. Real ASLLRP
# field transcription is not pre-normalized -- that's expected of
# genuine linguistic annotation, but it means "the examples" cannot be
# used as an already-consistent standard without a curation pass first.
# A prompt instructing the model to copy a form that varies within its
# own few-shot set cannot succeed; this fixes the SPECIFIC categories
# the new prompt calls out (possessive, indexed-pronoun casing) via
# minimal, listed edits -- not a general rewrite:
#   - "My uncle never vists me." gloss: POSS-1p+ -> POSS-1p (dropped
#     the "+" repetition marker to match example 1's bare form);
#     IX-1p -> ix-1p (lowercased to match the new prompt's own stated
#     convention).
#   - "My favorite dessert is pumpkin pie." gloss: (G/Q)POSS-1p+ ->
#     POSS-1p (dropped both the "(G/Q)" prefix and "+" suffix, same
#     unification).
# Every other token in every example is untouched -- this is a targeted
# fix for exactly the categories the new prompt names, not a general
# cleanup of ASLLRP's real notation.
FEW_SHOT_EXAMPLES = [
    ("While my father is quieter", "DURING/WHILE POSS-1p FATHER MORE QUIET"),
    ("There are a lot of studies on speech disorders", "STUDY ON SPEECH/ORAL fs-DISORDER A-LOT"),
    ("I easily find things funny.", "EASY FIND/FIND-OUT (2h)THING FUNNY+"),
    ("Boston is a great city.", "ns-BOSTON WOW+ WONDERFUL CITY/COMMUNITY_3++"),
    ("My uncle never vists me.", "POSS-1p UNCLE IX-loc:i NEVER VISIT-1p ix-1p NEVER"),
    ("My favorite dessert is pumpkin pie.", "DATE/DESSERT+ POSS-1p FAVORITE/PREFER+ MELON/PUMPKIN+ PIE"),
    # V3 ADDITION: a real ASLLRP future-tense example. V2's system prompt
    # only INSTRUCTED "use one consistent form for tense markers" but
    # never DEMONSTRATED future marking anywhere in the few-shot set --
    # V2's actual output then dropped FUTURE entirely (0/40, vs V1's 5
    # occurrences), the opposite of consistency. The possessive-pronoun
    # fix worked specifically because POSS-1p was DEMONSTRATED, not just
    # described; applying the same logic here. Source: real ASLLRP pair,
    # "FUTURE+|READ|BOOK|IX-1p" -- normalized the same way as the other
    # examples (dropped the "+" repetition marker, lowercased IX-1p).
    ("I will read a book.", "FUTURE READ BOOK ix-1p"),
]
FEW_SHOT_SOURCE = ("real ASLLRP (translation, gloss) pairs from data/processed/asllrp/train/*.npz, "
                    "with a documented possessive/pronoun/tense normalization pass -- see comments above")

SYSTEM_PROMPT_TEMPLATE = """You are a certified ASL linguist producing gloss annotations for a bidirectional
ASL translation system. You will be given an English sentence and must output its
ASL gloss transcription.

CANONICAL NOTATION -- CLOSED SET
You must use ONLY the exact notation forms shown in the few-shot examples below.
Never invent a variant, abbreviation, or alternate spelling of a form that already
appears in the examples, even if the variant is linguistically valid ASL notation
in general. Consistency across the corpus matters more than which specific
convention was chosen.

Specifically, before writing your final answer, check every token you produce
against this rule: "If this concept (tense marking, possession, pronoun, aspect,
etc.) appears anywhere in the few-shot examples, my token for it must be
character-for-character identical to how it appears there."

Known high-risk categories where drift has occurred before -- hold these to one
canonical form each:
- Tense/aspect markers (e.g. future marking): use exactly one form throughout,
  never alternate between a full word and an abbreviation for the same marker.
  IMPORTANT, positive requirement (not just a consistency rule): if the source
  English sentence expresses future tense in ANY form ("will", "going to",
  "shall", "would", etc.), you MUST include the sign FUTURE in your gloss
  output to mark this -- future tense information must never be silently
  dropped. Always write it as exactly FUTURE (never an abbreviation like FUT),
  exactly as demonstrated in the "I will read a book." example below.
- Possessive pronouns: use exactly one notation pattern for all persons
  (e.g. POSS-1p / POSS-2p / POSS-3p), never mix with numbered forms like
  POSS-1SG for the same concept.
- Indexed pronouns: follow the ix-1p / ix-2p / ix-3p pattern already used in
  the examples -- these are not missing vocabulary, they are the correct
  notation for I/you/he/she and must not be re-glossed as separate signs.
- Negation, WH-question, and topic markers: use exactly the marker token shown
  in the examples (e.g. NEG, WH-Q, TOP) -- do not paraphrase or expand them.

Also do NOT include English function words that have no separate sign: no "the",
"a", "an", "is", "are", "was", "were", "do", "does", "did" (as auxiliaries), "to"
(as infinitive marker). Fingerspelled words (names, no dedicated sign) are written
hyphenated letter by letter, e.g. J-O-H-N.

FEW-SHOT EXAMPLES
{examples}

OUTPUT FORMAT
Return only the gloss sequence, space-separated, uppercase for signs, using the
notation conventions above. No explanation, no alternate readings, no commentary.

SELF-CHECK (perform silently before answering)
1. Does every tense/aspect/possessive/pronoun token match an exact form seen in
   the examples, not a plausible-sounding variant?
2. Is any single concept represented by two different tokens anywhere in your
   own output? If so, pick the form used in the examples and fix it.
3. Only after passing both checks, output the final gloss line.

English sentence: {sentence}
Gloss:"""


def build_prompt(sentence):
    examples_block = "\n".join(f"English: {eng}\nGloss: {gloss}" for eng, gloss in FEW_SHOT_EXAMPLES)
    return SYSTEM_PROMPT_TEMPLATE.format(examples=examples_block, sentence=sentence)


def generate_pseudo_gloss(client, sentence):
    prompt = build_prompt(sentence)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=900,  # raised from 600: the v2 run (600) produced 1 fully
                          # empty output and 1 severely-truncated output (a
                          # 28-word sentence collapsing to a single token) that
                          # v1 (400->600 increase) did not have -- consistent
                          # with the longer v2/v3 system prompt (self-check
                          # instructions, now also the extra FUTURE-marking
                          # instruction and 7th example) leaving less of the
                          # same budget for the actual answer on already-long
                          # inputs. openai/gpt-oss-120b is a reasoning model --
                          # reasoning tokens draw from the SAME budget as the
                          # visible answer (confirmed earlier: max_tokens=5 or
                          # 100 produced finish_reason='length' with EMPTY
                          # content every time, reasoning alone consuming the
                          # whole budget). 900 leaves real headroom beyond the
                          # ~7-40 reasoning tokens observed in testing, with
                          # margin for the longest (32-word) sentences too.
        temperature=0.3,
        reasoning_effort="low",  # confirmed via direct A/B test: cuts
                                  # reasoning-token overhead ~6x (41->7 tokens
                                  # on a trivial prompt) with no loss of
                                  # correctness -- this task (pattern-matching
                                  # a few-shot gloss format) doesn't need
                                  # heavy reasoning, and full-corpus-scale
                                  # cost/latency depends directly on this.
    )
    finish_reason = resp.choices[0].finish_reason
    content = resp.choices[0].message.content.strip()
    if finish_reason != "stop" and not content:
        content = f"[EMPTY: finish_reason={finish_reason}]"
    return content


def load_how2sign_sample(csv_path, seed, size):
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    rng = random.Random(seed)
    # Mix of short and long sentences: sort by word count, sample from
    # short/medium/long tertiles separately so the validation sample
    # isn't accidentally all-short or all-long.
    rows_with_len = [(len(r["SENTENCE"].split()), r) for r in rows]
    rows_with_len.sort(key=lambda x: x[0])
    n = len(rows_with_len)
    thirds = [rows_with_len[:n//3], rows_with_len[n//3:2*n//3], rows_with_len[2*n//3:]]

    sample = []
    per_third = size // 3
    for third in thirds:
        sample.extend(rng.sample(third, min(per_third, len(third))))
    while len(sample) < size:
        sample.append(rng.choice(rows_with_len))

    rng.shuffle(sample)
    return [(wc, r["SENTENCE"], r["SENTENCE_ID"]) for wc, r in sample[:size]]


# --- Post-processing normalization (item 3) -------------------------
# DETERMINISTIC, applied AFTER generation, on every run's raw output --
# not more prompt engineering. Necessary because each of the 40 (or,
# at full scale, 31,165) sentences is generated by a FULLY INDEPENDENT
# API call with no shared state -- a prompt can raise the odds any one
# call is internally consistent, but nothing about a stateless call
# can guarantee two SEPARATE calls pick the same surface form for the
# same concept. Confirmed directly: even v3's stricter prompt still
# produced ix-1p/IX-1p case-drift in the v2 run despite an explicit
# self-check instruction telling the model to prevent exactly this.
#
# Built from REAL data, not assumed: every entry below was found by
# systematically grouping every token from the ALREADY-GENERATED v1+v2
# outputs by a normalized key (lowercased, hyphens/slashes/plus/parens
# stripped) and inspecting every group with more than one distinct raw
# surface form -- this is not a guess at "the two known issues," it is
# the actual list of every variant pair the real data contained,
# including two neither v1 nor v2's own analysis had caught:
#   - SHIRAZ vs S-H-I-R-A-Z (plain gloss vs fingerspelling for the same
#     wine name -- "Shiraz" has no dedicated ASL sign, so canonicalized
#     to fingerspelling, matching this project's own fs-/hyphenated-
#     letter convention for exactly this case)
#   - TUNE-UP vs TUNE/UP (hyphen vs slash for the same compound verb --
#     canonicalized to hyphen; the slash convention in these few-shot
#     examples denotes ALTERNATE gloss names for one sign, e.g.
#     SPEECH/ORAL, not a compound-word separator, so TUNE/UP was a
#     misapplication of that convention, not a legitimate alternate)
#   - NOT vs NEG (found independently while checking the FUTURE/FUT
#     pair: v1 used the plain lexical sign NOT, v2 used the abbreviated
#     grammatical marker NEG for negation -- same zero-overlap drift
#     signature as FUTURE/FUT. Canonicalized to NEG, matching the
#     system prompt's own stated convention for grammatical markers.)
NORMALIZATION_MAP = {
    "IX-1p": "ix-1p", "IX-2p": "ix-2p", "IX-3p": "ix-3p",
    "FUT": "FUTURE",
    "POSS-1sg": "POSS-1p",
    "NOT": "NEG",
    "SHIRAZ": "S-H-I-R-A-Z",
    "TUNE/UP": "TUNE-UP",
}


def normalize_gloss(gloss):
    if not gloss or gloss.startswith("["):
        return gloss
    tokens = gloss.split()
    normalized = [NORMALIZATION_MAP.get(t, t) for t in tokens]
    return " ".join(normalized)


def main():
    print("=== Item 1: API test call ===")
    client = get_client()
    result = test_api_call(client)
    print(f"Model: {MODEL}")
    print(f"Response: {result!r}")
    if "OK" not in result.upper():
        print("WARNING: response did not contain 'OK' as expected -- inspect manually.")
    print("API call SUCCEEDED.\n")

    print("=== Item 3: Generating pseudo-glosses for validation sample ===")
    sample = load_how2sign_sample(HOW2SIGN_TRAIN_CSV, SAMPLE_SEED, SAMPLE_SIZE)
    print(f"Sample size: {len(sample)} sentences (seed={SAMPLE_SEED}), word-count range: "
          f"{min(s[0] for s in sample)}-{max(s[0] for s in sample)}")

    results = []
    for i, (wc, sentence, sent_id) in enumerate(sample):
        try:
            gloss = generate_pseudo_gloss(client, sentence)
        except Exception as e:
            gloss = f"[ERROR: {type(e).__name__}: {e}]"
        results.append({"sentence_id": sent_id, "word_count": wc, "sentence": sentence, "pseudo_gloss": gloss})
        print(f"[{i+1}/{len(sample)}] ({wc}w) {sentence[:60]!r} -> {gloss[:80]!r}")
        time.sleep(RATE_LIMIT_DELAY_SECONDS)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "few_shot_source": FEW_SHOT_SOURCE,
                    "sample_seed": SAMPLE_SEED, "results": results}, f, indent=2)
    print(f"\nSaved {len(results)} RAW results to {OUT_JSON}")

    print("\n=== Item 3: Applying post-processing normalization ===")
    normalized_results = []
    n_changed = 0
    for r in results:
        norm_gloss = normalize_gloss(r["pseudo_gloss"])
        if norm_gloss != r["pseudo_gloss"]:
            n_changed += 1
        normalized_results.append({**r, "pseudo_gloss_raw": r["pseudo_gloss"], "pseudo_gloss": norm_gloss})
    print(f"Normalization changed {n_changed}/{len(results)} outputs")

    with open(OUT_JSON_NORMALIZED, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "few_shot_source": FEW_SHOT_SOURCE, "normalization_map": NORMALIZATION_MAP,
                    "sample_seed": SAMPLE_SEED, "results": normalized_results}, f, indent=2)
    print(f"Saved {len(normalized_results)} NORMALIZED results to {OUT_JSON_NORMALIZED}")


if __name__ == "__main__":
    main()
