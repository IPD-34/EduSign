"""
scripts/pseudo_gloss_groq_full_run.py

Full-scale run of the V3-validated pseudo-gloss pipeline (pseudo_gloss_
groq_validation.py) across How2Sign's complete 35,265-sentence corpus
(train+val+test, not just the 31,165-sentence train split the
validation script sampled from) -- so real per-word train/val/test
counts can be computed the same way as every other dataset in this
project, and the 10+/split floor can be applied honestly.

NEW: source-text pre-cleaning stage, absent from the validation script,
built from REAL corruption found by scanning the actual corpus (not
assumed): 8/31,165 train rows had double-encoded UTF-8 mojibake
(genuine apostrophes/quotes corrupted into 3-byte garbage, e.g. "you
\xe2\x80\x99re" instead of "you're" -- classic UTF-8-decoded-as-
Latin-1-then-resaved corruption), 5/31,165 had unescaped HTML entities
(&amp;). Both confirmed fixable via ftfy.fix_text() + html.unescape()
on the ACTUAL raw strings before writing this script, not a synthetic
test. Applied to every sentence before it reaches the LLM -- a raw
mojibake byte sequence is not something the model should be asked to
gloss literally.

RESUME-SAFE, same philosophy as every other long-running job in this
project: writes one result per sentence to a JSONL file, appending as
it goes (not buffering the whole 35,265-sentence corpus in memory
until the end) -- if interrupted, rerun and already-completed
sentence_ids are skipped.

MULTI-KEY ROTATION: loads GROQ_API_KEY plus GROQ_API_KEY_2..GROQ_API_KEY_N
from .env (however many are actually filled in -- not all 14 slots
need to be used). Each key gets its OWN independent 2.5s pacing
(KeyRotator tracks last-used-time per key, not globally) -- so N keys
round-robined gives ~N x the combined throughput without any single
key exceeding the same per-key rate the original 1-key version used.
With 1 key filled in, this behaves identically to the original script.

RATE LIMIT: same 2.5s per-key delay as the validation script (24 req/min
per key, under Groq's free-tier 30/min cap) -- carried over unchanged, not
re-derived, since that's the already-proven-safe value.
"""
import sys
import os
import re
import csv
import json
import time
import html
import threading
from queue import Queue, Empty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ftfy
import groq
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

from pseudo_gloss_groq_validation import (
    MODEL, FEW_SHOT_EXAMPLES, SYSTEM_PROMPT_TEMPLATE, NORMALIZATION_MAP,
    build_prompt, normalize_gloss, RATE_LIMIT_DELAY_SECONDS,
)

METADATA_DIR = Path(
    r"C:\Users\KRUPAS~1\AppData\Local\Temp\claude\D--College-Projects---Resume-bidirectional-sign-language"
    r"\1f4ac0b6-58be-4111-abc9-012b28f39d49\scratchpad\how2sign_inspect\how2sign_holistic_features\metadata"
)
SPLITS = {
    "train": METADATA_DIR / "how2sign_train.csv",
    "val": METADATA_DIR / "how2sign_val.csv",
    "test": METADATA_DIR / "how2sign_test.csv",
}
OUT_PATH = Path("logs/how2sign_pseudo_gloss_full.jsonl")


def clean_sentence(text):
    """Real corruption found by scanning the actual corpus, see module
    docstring -- not a hypothetical cleaning step. ftfy.fix_text handles
    the double-encoded-UTF-8 mojibake; html.unescape handles literal
    HTML entities. Order matters: ftfy first (operates on the raw
    corrupted text), then unescape (in case an entity survived inside
    already-valid text)."""
    return html.unescape(ftfy.fix_text(text))


def load_all_rows():
    rows = []
    for split, path in SPLITS.items():
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                rows.append({"split": split, "sentence_id": r["SENTENCE_ID"], "sentence": r["SENTENCE"]})
    return rows


def load_done_ids():
    """"Done" means a real gloss was produced -- an [ERROR]/[EMPTY]
    line does NOT count (see generate_pseudo_gloss's retry-fix docstring:
    those are exactly the sentences the old, retry-less version silently
    lost, and they must be retried, not skipped forever)."""
    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if not record["pseudo_gloss"].startswith("["):
                        done.add(record["sentence_id"])
                except Exception:
                    continue
    return done


def load_all_keys(max_slots=20):
    """GROQ_API_KEY (slot 1) plus GROQ_API_KEY_2..GROQ_API_KEY_{max_slots},
    whichever are actually non-empty -- scans the full range rather than
    stopping at the first gap, since slots may not be filled in order."""
    keys = []
    k1 = os.environ.get("GROQ_API_KEY")
    if k1:
        keys.append(k1)
    for i in range(2, max_slots + 1):
        k = os.environ.get(f"GROQ_API_KEY_{i}")
        if k:
            keys.append(k)
    if not keys:
        raise RuntimeError("No GROQ_API_KEY* found in os.environ even after load_dotenv().")
    return keys


CLIENT_TIMEOUT_SECONDS = 30.0  # SPEED/RELIABILITY FIX: no timeout was set before --
                                 # a stalled connection could hang forever with no
                                 # exception ever raised (this is what actually
                                 # stalled the STEM Wiki sample: 13+ minutes, zero
                                 # output, process alive but stuck -- no retry logic
                                 # can help if the call never returns in the first
                                 # place). 30s is generous for this model/prompt
                                 # size (real calls complete in low single-digit
                                 # seconds) while still failing fast enough to retry.


def get_client():
    return Groq(api_key=load_all_keys()[0], timeout=CLIENT_TIMEOUT_SECONDS)


MAX_RETRIES = 8

MAX_TOKENS = 650  # SPEED FIX, verified not guessed: a real call against the
                    # longest known sentence in this corpus (34 words, nested
                    # parens) used only 324 completion tokens with
                    # finish_reason=stop (no truncation); a second real call
                    # used 398. Almost all of that is invisible reasoning
                    # tokens (the visible gloss output itself is 10-15 words
                    # typically) -- reasoning_effort="low" already minimizes
                    # this, it can't be cut further. Groq's per-account TPM
                    # accounting bills against prompt_tokens + max_tokens as
                    # a request-admission ceiling (confirmed directly: two
                    # 429 errors on the SAME max_tokens=900 quoted different
                    # "Requested" totals, consistent with prompt+max_tokens
                    # pre-admission, not just actual usage after the fact) --
                    # so lowering max_tokens directly raises how many calls
                    # fit in the TPM budget per minute. 650 keeps ~2x margin
                    # above the largest real completion_tokens measured (325),
                    # well clear of the 600 value that caused ONE truncation
                    # in this project's history (that was before
                    # reasoning_effort=low existed).


def _parse_retry_after_seconds(error, default=6.0):
    """RATE-LIMIT FIX: the real binding constraint turned out to be a
    tokens-per-minute (TPM) cap (8,000 TPM on this account/model), not
    the 30 req/min this script was originally paced around -- discovered
    via a live 429 whose message states the account's real limit
    directly. Groq's 429 body includes a human-readable hint ("Please
    try again in 1.3275s") -- prefer the Retry-After header if httpx
    exposes one, else parse that hint, else fall back to a fixed wait.
    Getting this number right matters: too short and we just re-hit the
    same TPM ceiling immediately, burning retries."""
    response = getattr(error, "response", None)
    if response is not None:
        header_val = response.headers.get("retry-after")
        if header_val:
            try:
                return float(header_val)
            except ValueError:
                pass
    match = re.search(r"try again in ([\d.]+)s", str(error))
    if match:
        return float(match.group(1))
    return default


def generate_pseudo_gloss(client, sentence):
    """RETRIES on rate-limit (429) errors instead of treating them as
    permanent failures -- the ORIGINAL version recorded a RateLimitError
    as "[ERROR: ...]" and moved on, silently losing that sentence's
    gloss every time the TPM ceiling was hit (confirmed happening on
    live output before this fix: 8/90 lines in the running job were
    already lost this way). A rate limit is a transient, expected
    condition given the real per-account TPM cap -- not a reason to
    drop data."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": build_prompt(sentence)}],
                max_tokens=MAX_TOKENS,
                temperature=0.3,
                reasoning_effort="low",
            )
            finish_reason = resp.choices[0].finish_reason
            content = resp.choices[0].message.content.strip()
            if finish_reason != "stop" and not content:
                content = f"[EMPTY: finish_reason={finish_reason}]"
            return content
        except groq.RateLimitError as e:
            last_error = e
            wait_s = _parse_retry_after_seconds(e)
            time.sleep(wait_s + 0.25)  # small margin past the server's own estimate
        except (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError) as e:
            last_error = e
            time.sleep(min(2 ** attempt, 30))  # transient network/server issues: exponential backoff
    return f"[ERROR after {MAX_RETRIES} retries: {type(last_error).__name__}: {last_error}]"


def worker_loop(client, work_queue, out_f, write_lock, stats, stats_lock, min_interval):
    """One thread per API key. Genuine concurrency, not sequential
    round-robin -- a blocking HTTP call in one thread doesn't hold up
    the others (releases the GIL during I/O wait), so N keys really do
    give ~N x combined throughput, not just N x on paper. Each thread
    paces only ITS OWN successive calls (min_interval after its own
    previous call finishes) -- independent per-key rate, not a shared
    budget split N ways."""
    while True:
        try:
            row = work_queue.get_nowait()
        except Empty:
            return
        raw = row["sentence"]
        cleaned = clean_sentence(raw)
        t_call_start = time.time()
        try:
            gloss = generate_pseudo_gloss(client, cleaned)
            had_error = False
        except Exception as e:
            gloss = f"[ERROR: {type(e).__name__}: {e}]"
            had_error = True
        norm_gloss = normalize_gloss(gloss)
        record = {
            "sentence_id": row["sentence_id"], "split": row["split"],
            "sentence_raw": raw, "sentence_cleaned": cleaned,
            "pseudo_gloss_raw": gloss, "pseudo_gloss": norm_gloss,
        }
        with write_lock:
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
        with stats_lock:
            stats["done"] += 1
            if cleaned != raw:
                stats["cleaned"] += 1
            if had_error:
                stats["errors"] += 1
            if stats["done"] % 100 == 0 or stats["done"] == stats["total"]:
                elapsed = time.time() - stats["t0"]
                rate = stats["done"] / elapsed * 60 if elapsed > 0 else 0
                remaining_n = stats["total"] - stats["done"]
                eta_h = (remaining_n / rate / 60) if rate > 0 else float("inf")
                print(f"[{stats['done']}/{stats['total']}] cleaned={stats['cleaned']} "
                      f"errors={stats['errors']} rate={rate:.1f}/min ETA={eta_h:.1f}h")
        work_queue.task_done()

        elapsed_call = time.time() - t_call_start
        if elapsed_call < min_interval:
            time.sleep(min_interval - elapsed_call)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    keys = load_all_keys()
    print(f"Using {len(keys)} API key(s) -- expected combined rate ~{len(keys) * 60 / RATE_LIMIT_DELAY_SECONDS:.0f}/min")
    clients = [Groq(api_key=k, timeout=CLIENT_TIMEOUT_SECONDS) for k in keys]

    all_rows = load_all_rows()
    done_ids = load_done_ids()
    remaining = [r for r in all_rows if r["sentence_id"] not in done_ids]

    print(f"Total sentences (train+val+test): {len(all_rows)}")
    print(f"Already done: {len(done_ids)}")
    print(f"Remaining: {len(remaining)}")

    if args.limit is not None:
        remaining = remaining[:args.limit]
        print(f"(--limit applied: processing {len(remaining)})")

    work_queue = Queue()
    for row in remaining:
        work_queue.put(row)

    stats = {"done": 0, "cleaned": 0, "errors": 0, "total": len(remaining), "t0": time.time()}
    stats_lock = threading.Lock()
    write_lock = threading.Lock()

    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        threads = [
            threading.Thread(target=worker_loop, args=(
                client, work_queue, out_f, write_lock, stats, stats_lock, RATE_LIMIT_DELAY_SECONDS
            ))
            for client in clients
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(f"\nDone. {stats['done']} processed this run, {stats['cleaned']} needed cleaning, {stats['errors']} errors.")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
