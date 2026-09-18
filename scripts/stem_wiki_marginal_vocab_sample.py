"""
scripts/stem_wiki_marginal_vocab_sample.py

Cheap validation BEFORE committing to ASL STEM Wiki's full 187 GiB
extraction: sample real sentences from its videos.csv (64,266 rows,
sentence-level, no gloss labels -- same situation as How2Sign),
generate pseudo-glosses for the sample using the SAME validated V3
pipeline, and measure how many of the resulting gloss words are
actually NEW relative to the vocabulary we already have real, usable
data for (the 3,924-word ASLLRP+WLASL+ASL-Citizen gloss union). This
estimates STEM Wiki's real marginal contribution before spending
extraction cost on it, not after.

RESUME-SAFE (added after this script stalled twice sharing a Groq key
with one of How2Sign's 14 active workers -- every key is a dedicated
How2Sign worker, so there is no genuinely uncontended key to switch to;
the real fix is making repeated stalls cost nothing instead of losing
all progress). Writes one JSONL line per sentence as it completes
(same pattern as pseudo_gloss_groq_full_run.py) instead of buffering
all 60 results until the end -- a kill-and-restart now resumes from
wherever it left off rather than starting over.
"""
import sys
import csv
import json
import time
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pseudo_gloss_groq_validation import normalize_gloss
from pseudo_gloss_groq_full_run import get_client, generate_pseudo_gloss, clean_sentence

STEM_WIKI_CSV = (
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\asl_stem_wiki\ASL_STEM_Wiki\videos.csv"
)
SAMPLE_SIZE = 60
SAMPLE_SEED = 42
DELAY_SECONDS = 5.0  # conservative: shares a key with one of How2Sign's active workers
OUT_JSONL = Path("logs/stem_wiki_marginal_vocab_sample.jsonl")


def load_gloss_union():
    def load_set(path):
        with open(path, encoding="utf-8") as f:
            return set(json.load(f).keys())
    return (load_set("logs/_tmp_asllrp_canonical_gloss_counts.json")
            | load_set("logs/_tmp_wlasl_gloss_counts.json")
            | load_set("logs/_tmp_asl_citizen_full_gloss_counts.json"))


def load_done_indices():
    done = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if not r["pseudo_gloss"].startswith("["):
                        done.add(r["sample_idx"])
                except Exception:
                    continue
    return done


def main():
    client = get_client()
    existing_vocab = load_gloss_union()
    print(f"Existing real gloss-labeled vocabulary (ASLLRP+WLASL+Citizen): {len(existing_vocab)} words")

    with open(STEM_WIKI_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(SAMPLE_SEED)
    sample = rng.sample(rows, min(SAMPLE_SIZE, len(rows)))

    done_idx = load_done_indices()
    print(f"Sampled {len(sample)} STEM Wiki sentences from {len(rows)} total ({len(done_idx)} already done)")

    with open(OUT_JSONL, "a", encoding="utf-8") as out_f:
        for i, row in enumerate(sample):
            if i in done_idx:
                continue
            raw = row["sentence text"]
            cleaned = clean_sentence(raw)
            try:
                gloss = generate_pseudo_gloss(client, cleaned)
            except Exception as e:
                gloss = f"[ERROR: {type(e).__name__}: {e}]"
            norm = normalize_gloss(gloss)
            record = {"sample_idx": i, "article": row["article"], "sentence": raw, "pseudo_gloss": norm}
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            print(f"[{i+1}/{len(sample)}] ({row['article'][:30]}) {raw[:50]!r} -> {norm[:60]!r}")
            time.sleep(DELAY_SECONDS)

    # Recompute the marginal-vocab estimate from the full JSONL (covers
    # this run plus any earlier resumed runs) -- not just this run's rows.
    all_gloss_words = Counter()
    results = []
    with open(OUT_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            results.append(r)
            if not r["pseudo_gloss"].startswith("["):
                for w in r["pseudo_gloss"].split():
                    all_gloss_words[w.lower()] += 1

    distinct_gloss_words = set(all_gloss_words.keys())
    new_words = distinct_gloss_words - existing_vocab
    print(f"\n=== MARGINAL VOCAB ESTIMATE ===")
    print(f"Distinct gloss words produced from {len(sample)}-sentence sample: {len(distinct_gloss_words)}")
    print(f"Of those, NEW (not in existing 3,924-word gloss union): {len(new_words)}")
    print(f"New-word rate: {len(new_words)}/{len(distinct_gloss_words)} = {len(new_words)/max(len(distinct_gloss_words),1)*100:.1f}%")
    print(f"Sample new words: {sorted(new_words)[:30]}")

    with open("logs/stem_wiki_marginal_vocab_sample_result.json", "w") as f:
        json.dump({
            "sample_size": len(sample), "existing_vocab_size": len(existing_vocab),
            "distinct_gloss_words_in_sample": len(distinct_gloss_words),
            "new_words_count": len(new_words), "new_words": sorted(new_words),
            "results": results,
        }, f, indent=2)
    print("\nSaved to logs/stem_wiki_marginal_vocab_sample_result.json")


if __name__ == "__main__":
    main()
