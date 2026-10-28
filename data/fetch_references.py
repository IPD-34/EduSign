"""Fetch the reference sentences for a manifest's clips from the ASL STEM Wiki zip.

Only the single member ASL_STEM_Wiki/videos.csv is read from the remote zip (HTTP range requests, ~5 MB); the ~190 GB
archive is never downloaded. Rows whose clip_id (the 'video filename' without .mp4) is in the manifest are kept and
written as clip_id,reference to --out. The sentence text is written ONLY to --out and is never printed. --out must be
git-ignored (default local/references.csv); any other path is refused before anything is fetched.

    python data/fetch_references.py --manifest manifests/eval300.csv
Options: --out <csv> (default local/references.csv), --url <zip url> (default: same as fetch_clips.py)
Licence: the dataset is non-commercial research only and must not be redistributed (see use.txt in the zip).
"""
import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

from remotezip import RemoteZip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_clips import DEFAULT_URL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MEMBER = "ASL_STEM_Wiki/videos.csv"
FILENAME_COL = "video filename"
TEXT_COL = "sentence text"
FIELDS = ["clip_id", "reference"]


def is_gitignored(path):
    """True only if git confirms `path` is ignored. Anything else (not ignored, tracked, outside the repo, no git) is False."""
    try:
        r = subprocess.run(["git", "check-ignore", "-q", "--", str(Path(path).resolve())],
                           cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def read_manifest_ids(path):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if "clip_id" not in (rd.fieldnames or []):
            sys.exit(f"{path}: no clip_id column (found {rd.fieldnames})")
        ids = [r["clip_id"].strip() for r in rd if r["clip_id"].strip()]
    return list(dict.fromkeys(ids))   # unique, manifest order


def open_videos_csv(url):
    """Context manager yielding a binary stream of videos.csv read straight from the remote zip."""
    class _Ctx:
        def __enter__(self):
            # Microsoft's download server answers suffix ranges (bytes=-N) with HTTP 501, so use explicit ranges
            self.rz = RemoteZip(url, timeout=60, support_suffix_range=False)
            self.f = self.rz.open(MEMBER)
            return self.f

        def __exit__(self, *exc):
            self.f.close()
            self.rz.close()
    return _Ctx()


def select_references(binary_stream, wanted):
    """Return ({clip_id: reference}, duplicate_count) for the wanted clip_ids; keeps the first row of a duplicated clip."""
    rd = csv.DictReader(io.TextIOWrapper(binary_stream, encoding="utf-8-sig", newline=""))
    missing_cols = [c for c in (FILENAME_COL, TEXT_COL) if c not in (rd.fieldnames or [])]
    if missing_cols:
        sys.exit(f"{MEMBER}: missing column(s) {missing_cols}; columns found: {rd.fieldnames}")
    wanted, found, dup = set(wanted), {}, 0
    for row in rd:
        name = (row[FILENAME_COL] or "").strip()
        if not name.endswith(".mp4"):
            continue
        cid = name[:-4]
        if cid not in wanted:
            continue
        if cid in found:
            dup += 1
            continue
        found[cid] = row[TEXT_COL]
    return found, dup


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path, help="manifest CSV with a clip_id column")
    ap.add_argument("--out", type=Path, default=ROOT / "local" / "references.csv",
                    help="output CSV (clip_id,reference); must be git-ignored (default local/references.csv)")
    ap.add_argument("--url", default=DEFAULT_URL, help="ASL_STEM_Wiki.zip URL (default: as in fetch_clips.py)")
    args = ap.parse_args(argv)

    if not is_gitignored(args.out):
        sys.exit(f"refusing to write to {args.out}: git does not report it as ignored, and the references contain "
                 f"sentence text that must never be committed. Use a path under local/ (see .gitignore).")
    ids = read_manifest_ids(args.manifest)
    print(f"{len(ids)} clips in {args.manifest.name}; reading {MEMBER} from the remote zip")
    with open_videos_csv(args.url) as f:
        found, dup = select_references(f, ids)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows((cid, found[cid]) for cid in ids if cid in found)
    tmp.replace(args.out)

    missing = [c for c in ids if c not in found]
    print(f"wrote {len(found)} rows to {args.out}")
    if dup:
        print(f"WARNING: {dup} duplicate row(s) for the same clip in videos.csv; kept the first of each")
    if missing:
        print(f"WARNING: {len(missing)} manifest clip(s) have no reference in videos.csv, e.g. {missing[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
