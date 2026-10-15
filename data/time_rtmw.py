"""Time RTMW extraction (data/extract_rtmw.py) on N clips and write seconds per clip to docs/rtmw_speed.md.

Per clip the time covers video decoding + detection + pose for every frame (the same work as a real extraction run);
model loading and a warm-up clip are excluded. Nothing is written to disk except the report.

    python data/time_rtmw.py --manifest manifests/eval300.csv --clips <clip_folder> [--n 20] [--device cuda]
"""
import argparse
import random
import statistics
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_rtmw as ex  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else "n/a"
    except Exception:
        return "n/a"


class GpuSampler:
    """Polls nvidia-smi once per second in a background thread while the timed loop runs."""

    def __init__(self, gpu=None):
        self.gpu = gpu      # nvidia-smi id/UUID of the GPU in use (nvidia-smi ignores CUDA_VISIBLE_DEVICES)
        self.samples = []   # (utilization %, memory MiB)
        self.raw = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        q = ["nvidia-smi"] + (["-i", self.gpu] if self.gpu else []) + [
            "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
        while not self._stop.is_set():
            try:
                line = subprocess.run(q, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
                util, mem = (float(x) for x in line.split(","))
                self.samples.append((util, mem))
                self.raw.append(line)
            except Exception:
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--clips", required=True, type=Path, help="folder holding <clip_id>.mp4")
    ap.add_argument("--n", type=int, default=20, help="number of clips to time (default 20)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--det-model", default=None)
    ap.add_argument("--pose-model", default=None)
    ap.add_argument("--report", type=Path, default=ROOT / "docs" / "rtmw_speed.md")
    ap.add_argument("--shard", type=ex.parse_shard, default=(0, 1), metavar="i/N",
                    help="time only sorted clip_ids with index %% N == i (default 0/1 = all)")
    args = ap.parse_args(argv)

    ids = ex.shard_ids(ex.read_clip_ids(args.manifest), *args.shard)
    present = [c for c in ids if (args.clips / f"{c}.mp4").exists()]
    if len(present) < args.n + 1:
        sys.exit(f"need {args.n + 1} clips present in {args.clips} (n timed + 1 warm-up), found {len(present)}")
    picked = random.Random(SEED).sample(present, args.n + 1)
    warmup, clips = picked[0], picked[1:]

    wb = ex.build_extractor(args.device, args.det_model, args.pose_model)
    ex.extract_clip(wb, args.clips / f"{warmup}.mp4")   # warm-up, not timed

    smi_id = ex.physical_gpu(args.device)
    idle = GpuSampler(smi_id)   # idle baseline before the timed run
    with idle:
        time.sleep(3)
    providers = ex.providers_in_use(wb)
    print(f"providers before timing: {providers}")
    rows = []
    with GpuSampler(smi_id) as gpu:
        for cid in clips:
            path = args.clips / f"{cid}.mp4"
            t0 = time.perf_counter()
            data = ex.extract_clip(wb, path)
            dt = time.perf_counter() - t0
            ex.assert_on_cuda(wb)   # fail loudly on a mid-run CPU fallback
            rows.append((cid, len(data["keypoints"]), dt))
            print(f"{cid}: {len(data['keypoints'])} frames, {dt:.2f}s")
    providers_after = ex.providers_in_use(wb)
    print(f"providers after timing:  {providers_after}")
    utils = [s[0] for s in gpu.samples]
    idle_utils = [s[0] for s in idle.samples]
    print(f"nvidia-smi during timed run ({len(utils)} samples, 1/s; columns: util %, memory MiB):")
    for line in gpu.raw[:5] + (["..."] if len(gpu.raw) > 10 else []) + gpu.raw[-5:]:
        print("  ", line)
    if not utils:
        sys.exit("no nvidia-smi samples collected - cannot confirm GPU utilisation")
    print(f"GPU utilisation: mean {statistics.mean(utils):.0f}%, max {max(utils):.0f}% "
          f"(idle before run: {statistics.mean(idle_utils) if idle_utils else float('nan'):.0f}%)")

    secs = [r[2] for r in rows]
    frames = sum(r[1] for r in rows)
    lines = [
        "# RTMW extraction speed",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Model: rtmlib Wholebody `{ex.MODE}` (YOLOX-tiny + RTMW-dw-l-m 256x192), onnxruntime",
        f"- Device: `{args.device}` ({gpu_name()})",
        f"- Clips timed: {len(rows)} (seed {SEED}, sampled from `{args.manifest.name}`, "
        f"shard {args.shard[0]}/{args.shard[1]}), 1 extra warm-up clip excluded",
        "- Timed work per clip: video decode + detection + pose for every frame; model load excluded",
        f"- onnxruntime providers in use (before / after the timed run): `{providers}` / `{providers_after}`",
        f"- GPU utilisation from nvidia-smi during the timed run ({len(utils)} samples at 1/s): "
        f"mean {statistics.mean(utils):.0f}%, max {max(utils):.0f}%, "
        f"memory {min(s[1] for s in gpu.samples):.0f}-{max(s[1] for s in gpu.samples):.0f} MiB "
        f"(idle before run: {statistics.mean(idle_utils) if idle_utils else float('nan'):.0f}%)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean seconds per clip | {statistics.mean(secs):.2f} |",
        f"| Median seconds per clip | {statistics.median(secs):.2f} |",
        f"| Min / max seconds per clip | {min(secs):.2f} / {max(secs):.2f} |",
        f"| Mean seconds per frame | {sum(secs) / frames:.4f} |",
        f"| Total frames | {frames} |",
        f"| Total seconds | {sum(secs):.1f} |",
        "",
        "## Per clip",
        "",
        "| clip_id | frames | seconds |",
        "|---|---|---|",
    ] + [f"| {cid} | {n} | {dt:.2f} |" for cid, n, dt in rows]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"mean {statistics.mean(secs):.2f}s/clip -> {args.report}")


if __name__ == "__main__":
    main()
