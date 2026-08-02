# scripts/retry_failed_downloads.py
import argparse
import csv
import logging
import subprocess
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MANIFEST = Path("data/raw/wlasl/manifests/download_manifest.csv")
RETRY_OUT = Path("data/raw/wlasl/manifests/download_manifest_retry.csv")
VIDEOS_DIR = Path("data/raw/wlasl/videos")


def try_direct(url: str, out_path: Path) -> tuple[bool, str]:
    try:
        r = requests.get(url, timeout=30, stream=True)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True, ""
    except Exception as e:
        return False, f"direct: {e}"


def try_ytdlp(url: str, out_path: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["yt-dlp", "-o", str(out_path), url],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode == 0 and out_path.exists():
        return True, ""
    return False, f"yt-dlp: {result.stderr.strip()[:300]}"


def retry_one(video_id: str, url: str, source: str) -> tuple[str, str]:
    out_path = VIDEOS_DIR / f"{video_id}.mp4"
    if out_path.exists():
        return "already_exists", ""

    ok, err1 = try_direct(url, out_path)
    if ok:
        return "success", ""

    ok, err2 = try_ytdlp(url, out_path)
    if ok:
        return "success", ""

    return "failed", f"{err1} | {err2}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="only retry this source domain")
    parser.add_argument("--limit", type=int, default=None, help="cap rows processed this run")
    args = parser.parse_args()

    rows = list(csv.DictReader(MANIFEST.open()))
    failed_rows = [r for r in rows if r["status"] == "failed"]
    if args.source:
        candidates = [r for r in failed_rows if r["source"] == args.source]
    else:
        candidates = failed_rows

    already_done = len(rows) - len(failed_rows)
    log.info(f"Already done: {already_done} | retry candidates: {len(candidates)}")

    if args.limit:
        candidates = candidates[: args.limit]
        log.info(f"Limiting to {len(candidates)} rows for this run")

    results = []
    success = already_exists = failed = 0
    total = len(candidates)

    for i, row in enumerate(candidates, 1):
        status, error = retry_one(row["video_id"], row["url"], row["source"])
        row["status"] = status
        row["error"] = error
        results.append(row)

        snippet = error[:70] if error else ""
        log.info(f"[{i}/{total}] {row['video_id']} ({row['source']}) -> {status} ({snippet})")

        if status == "success":
            success += 1
        elif status == "already_exists":
            already_exists += 1
        else:
            failed += 1

    log.info(f"Done. success={success} already_exists={already_exists} failed={failed}")

    with RETRY_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(results)

    log.info(f"Results written to {RETRY_OUT} — merge into {MANIFEST} once reviewed.")


if __name__ == "__main__":
    main()