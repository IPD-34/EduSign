"""Fetch a sample of ASL STEM Wiki clips from Microsoft's remote zip, one member at a time (HTTP range requests).
The ~187 GB archive is never downloaded: only the zip index and the chosen members are read.

Usage (repo root):
    python data/fetch_clips.py --manifest manifests/eval300.csv --n 21
Options:
    --manifest  CSV with a clip_id column (e.g. manifests/eval300.csv)
    --n         number of clips to sample (seed 42, without replacement); 0 = all
    --url       zip URL (default: Microsoft Download Center link below)
    --out       output folder (default local/clips/; git-ignored). Each clip is saved as <clip_id>.mp4.
Existing files are skipped. Clips whose member is not in the zip are reported.
Needs: pip install remotezip. Licence: the dataset is non-commercial research only and must not be redistributed
(see ASL_STEM_Wiki/use.txt inside the zip); clips stay under local/.
"""
import argparse
import csv
import random
import re
import sys
import time
from pathlib import Path

from remotezip import RemoteZip

DEFAULT_URL = "https://download.microsoft.com/download/4/c/f/4cfec788-7478-4e47-9a15-ace9b6a96198/ASL_STEM_Wiki.zip"
PREFIX = "ASL_STEM_Wiki/videos/"
SAFE_ID = re.compile(r"^[0-9A-Za-z&._-]+$")
SEED = 42


def read_ids(path):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if "clip_id" not in (rd.fieldnames or []):
            sys.exit(f"{path}: no clip_id column (found {rd.fieldnames})")
        ids = [r["clip_id"].strip() for r in rd if r["clip_id"].strip()]
    bad = [i for i in ids if not SAFE_ID.match(i)]
    if bad:
        sys.exit(f"{path}: refusing unexpected clip_ids, e.g. {bad[:3]}")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default="local/clips")
    a = ap.parse_args()

    ids = sorted(set(read_ids(a.manifest)))
    if a.n and a.n < len(ids):
        ids = sorted(random.Random(SEED).sample(ids, a.n))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(ids)} clips selected from {a.manifest} (seed {SEED}) -> {out}")

    ok = skipped = 0
    missing, failed = [], []
    nbytes, t0 = 0, time.time()
    with RemoteZip(a.url, timeout=60) as rz:
        names = set(rz.namelist())
        for i, cid in enumerate(ids, 1):
            dest = out / f"{cid}.mp4"
            if dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                continue
            member = f"{PREFIX}{cid}.mp4"
            if member not in names:
                missing.append(cid)
                continue
            for attempt in range(3):
                try:
                    data = rz.read(member)
                    dest.write_bytes(data)
                    nbytes += len(data)
                    ok += 1
                    break
                except Exception as e:
                    if attempt == 2:
                        failed.append((cid, f"{type(e).__name__}: {e}"))
                    else:
                        time.sleep(2)
            if i % 10 == 0 or i == len(ids):
                print(f"  {i}/{len(ids)}", flush=True)
    dt = time.time() - t0
    print(f"downloaded {ok} ({nbytes/1e6:.1f} MB in {dt:.0f}s) | skipped (already present) {skipped} | "
          f"missing from zip {len(missing)} | failed {len(failed)}")
    for cid in missing:
        print("  MISSING in zip:", cid)
    for cid, why in failed:
        print("  FAILED:", cid, why)
    if missing or failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
