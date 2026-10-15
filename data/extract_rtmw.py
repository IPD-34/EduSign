"""Extract RTMW whole-body keypoints (rtmlib 'lightweight': YOLOX-tiny + RTMW-dw-l-m 256x192) for every clip in a manifest.

For each manifest row, reads <clips>/<clip_id>.mp4 and writes <out>/<clip_id>.pkl:
    {"keypoints": [ndarray (1, 133, 2) float32, ...],   # one per frame, x/frame_width, y/frame_height
     "scores":    [ndarray (1, 133)    float32, ...],   # one per frame
     "n_frames":  int}                                   # frame count of the original clip
Frames are counted with a grab() pass; clips with n <= 256 keep every frame, longer clips keep the 256 frames with
index (i*(n-1))//255, i = 0..255 (first and last frame included). Pose runs only on kept frames. Each clip's
"<clip_id>,<n_frames>" is appended to <out>/frame_counts.csv.
Several people detected -> the one with the largest bounding box is kept. None detected -> zeros for both.
Existing .pkl files are skipped, so an interrupted run (e.g. a Kaggle session) can simply be restarted.

All paths come from the command line, so the same script runs locally and on Kaggle:
    python data/extract_rtmw.py --manifest manifests/eval300.csv --clips <clip_folder> --out <pose_folder>
Optional: --device cuda|cuda:N|cpu (default cuda; N indexes the GPUs visible after CUDA_VISIBLE_DEVICES; fails loudly
          if the CUDA provider cannot be created), --shard i/N (e.g. 0/2 and 1/2 on two GPUs/sessions),
          --det-model / --pose-model <local .onnx> for machines without internet (default: rtmlib downloads and caches them).
"""
import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

NUM_KPTS = 133
MODE = "lightweight"
MAX_FRAMES = 256   # clips longer than this are subsampled evenly (see keep_indices)
COUNTS_FILE = "frame_counts.csv"


def _expose_pip_cuda_dlls():
    """Windows: make the pip-installed nvidia-* CUDA/cuDNN DLLs loadable. onnxruntime otherwise loads cuDNN lazily,
    fails on a sub-library and silently falls back to CPU. No-op where those folders do not exist (Linux/Kaggle)."""
    if not hasattr(os, "add_dll_directory"):
        return
    bins = [str(b) for p in sys.path if p for b in Path(p).glob("nvidia/*/bin") if b.is_dir()]
    for b in bins:
        os.add_dll_directory(b)
    if bins:
        os.environ["PATH"] = os.pathsep.join(bins + [os.environ.get("PATH", "")])


def build_extractor(device="cuda", det_model=None, pose_model=None):
    """Return an rtmlib Wholebody (lightweight) on onnxruntime. Raises if `device` is cuda but ORT fell back to CPU."""
    if device.startswith("cuda"):
        _expose_pip_cuda_dlls()
    import onnxruntime as ort
    if device.startswith("cuda") and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()   # no-op if unavailable
    from rtmlib import Wholebody

    modes = Wholebody.MODE[MODE]
    wb = Wholebody(det=det_model or modes["det"], det_input_size=modes["det_input_size"],
                   pose=pose_model or modes["pose"], pose_input_size=modes["pose_input_size"],
                   to_openpose=False, backend="onnxruntime", device=device)
    if device.startswith("cuda"):
        # cuDNN is loaded lazily, so run both models once; ORT drops to CPU on the first failing run, not at creation
        warm = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
        wb.det_model(warm)
        wb.pose_model(warm, bboxes=[[0, 0, 320, 240]])
    print(f"providers in use ({device}): {providers_in_use(wb)}", flush=True)
    assert_on_cuda(wb)
    return wb


def cuda_device_index(device):
    """'cuda' -> 0, 'cuda:N' -> N, anything else -> None. N indexes the GPUs *visible* to the process, i.e. after
    CUDA_VISIBLE_DEVICES has been applied (so with CUDA_VISIBLE_DEVICES=1 the only GPU is 'cuda' / 'cuda:0')."""
    if device == "cuda":
        return 0
    if device.startswith("cuda:") and device[5:].isdigit():
        return int(device[5:])
    return None


def visible_gpus(env=None):
    """Entries of CUDA_VISIBLE_DEVICES (ids or UUIDs), or None when it is unset."""
    raw = (os.environ if env is None else env).get("CUDA_VISIBLE_DEVICES")
    return None if raw is None else [t.strip() for t in raw.split(",") if t.strip()]


def physical_gpu(device, env=None):
    """The nvidia-smi id/UUID behind `device`, honouring CUDA_VISIBLE_DEVICES (for utilisation sampling)."""
    idx = cuda_device_index(device) or 0
    vis = visible_gpus(env)
    return str(idx) if vis is None else (vis[idx] if idx < len(vis) else None)


def providers_in_use(wb):
    """Providers the live onnxruntime sessions are actually using (first entry is what runs the graph)."""
    return {"detector": wb.det_model.session.get_providers(), "pose": wb.pose_model.session.get_providers()}


def assert_on_cuda(wb):
    """Raise unless both sessions run on CUDAExecutionProvider, on the requested device (no-op for a cpu extractor).
    Works when CUDA_VISIBLE_DEVICES selects one GPU: the device id is the index among visible GPUs, not the physical id."""
    expected = cuda_device_index(wb.pose_model.device)
    if expected is None:
        return
    vis = visible_gpus()
    if vis is not None and expected >= len(vis):
        raise RuntimeError(f"device {wb.pose_model.device!r} but CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r} "
                           f"exposes only {len(vis)} GPU(s); use 'cuda' or 'cuda:<index among visible GPUs>'.")
    for name, tool in (("detector", wb.det_model), ("pose", wb.pose_model)):
        active = tool.session.get_providers()
        if not active or active[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{name} is not running on CUDA (providers: {active}); onnxruntime fell back to CPU. "
                               f"Check the CUDA 12 / cuDNN 9 DLLs or pass --device cpu explicitly.")
        got = int(tool.session.get_provider_options().get("CUDAExecutionProvider", {}).get("device_id", 0))
        if got != expected:
            raise RuntimeError(f"{name} is on CUDA device_id {got}, expected {expected} "
                               f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}).")


def parse_shard(text):
    """'i/N' -> (i, N) with 0 <= i < N."""
    try:
        i, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--shard must look like i/N, got {text!r}")
    if n < 1 or not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"--shard i/N needs N >= 1 and 0 <= i < N, got {text!r}")
    return i, n


def shard_ids(clip_ids, i=0, n=1):
    """Sort clip_ids, keep those whose index % n == i."""
    return [c for k, c in enumerate(sorted(clip_ids)) if k % n == i]


def _largest(bboxes):
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
    return bboxes[int(np.argmax(areas))]


def extract_frame(wb, frame):
    """Return (keypoints (1,133,2), scores (1,133)) in pixel coordinates; zeros if no person is detected."""
    bboxes = wb.det_model(frame)
    if bboxes is None or len(bboxes) == 0:
        return np.zeros((1, NUM_KPTS, 2), np.float32), np.zeros((1, NUM_KPTS), np.float32)
    kpts, scores = wb.pose_model(frame, bboxes=[_largest(bboxes)])   # one bbox -> exactly one person
    return kpts.astype(np.float32), scores.astype(np.float32)


def count_frames(video_path):
    """Number of frames by decoding through the whole file with grab() (CAP_PROP_FRAME_COUNT is not trusted)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    n = 0
    try:
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return n


def keep_indices(n, max_frames=MAX_FRAMES):
    """All frames if n <= max_frames, else max_frames indices (i*(n-1))//(max_frames-1), integer arithmetic only."""
    if n <= max_frames:
        return list(range(n))
    return [(i * (n - 1)) // (max_frames - 1) for i in range(max_frames)]


def extract_clip(wb, video_path, max_frames=MAX_FRAMES):
    n = count_frames(video_path)
    if n == 0:
        raise IOError(f"no frames decoded from {video_path}")
    keep = keep_indices(n, max_frames)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    keypoints, scores = [], []
    try:
        i = j = 0
        while j < len(keep):
            if not cap.grab():
                break
            if i == keep[j]:   # keep is strictly increasing; detection and pose run only on kept frames
                ok, frame = cap.retrieve()
                if not ok:
                    raise IOError(f"cannot decode frame {i} of {video_path}")
                h, w = frame.shape[:2]
                k, s = extract_frame(wb, frame)
                k = k.copy()
                k[..., 0] /= w
                k[..., 1] /= h
                keypoints.append(k)
                scores.append(s)
                j += 1
            i += 1
    finally:
        cap.release()
    if len(keypoints) != len(keep):
        raise IOError(f"{video_path}: decoded {i} frames on the second pass, counted {n} on the first")
    return {"keypoints": keypoints, "scores": scores, "n_frames": n}


def read_clip_ids(manifest):
    with open(manifest, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if "clip_id" not in (rd.fieldnames or []):
            sys.exit(f"{manifest}: no clip_id column (columns: {rd.fieldnames})")
        return [r["clip_id"] for r in rd]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path, help="manifest CSV with a clip_id column")
    ap.add_argument("--clips", required=True, type=Path, help="folder holding <clip_id>.mp4")
    ap.add_argument("--out", required=True, type=Path, help="folder to write <clip_id>.pkl into")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu (default: cuda)")
    ap.add_argument("--det-model", default=None, help="local YOLOX .onnx (default: rtmlib downloads lightweight)")
    ap.add_argument("--pose-model", default=None, help="local RTMW .onnx (default: rtmlib downloads lightweight)")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="i/N",
                    help="process only sorted clip_ids with index %% N == i (default 0/1 = all)")
    args = ap.parse_args(argv)

    clip_ids = shard_ids(read_clip_ids(args.manifest), *args.shard)
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in clip_ids if not (args.out / f"{c}.pkl").exists()]
    print(f"{len(clip_ids)} clips in manifest, {len(clip_ids) - len(todo)} already done, {len(todo)} to extract")
    if not todo:
        return 0

    counts_file = args.out / COUNTS_FILE
    try:   # exclusive create, so shards writing to the same folder cannot both write the header
        with open(counts_file, "x", newline="", encoding="utf-8") as f:
            f.write("clip_id,n_frames\n")
    except FileExistsError:
        pass

    wb = build_extractor(args.device, args.det_model, args.pose_model)
    failed = []
    try:
        from tqdm import tqdm
        it = tqdm(todo, unit="clip")
    except ImportError:
        it = todo
    for cid in it:
        src = args.clips / f"{cid}.mp4"
        try:
            if not src.exists():
                raise FileNotFoundError(src)
            data = extract_clip(wb, src)
            assert_on_cuda(wb)   # a mid-run CPU fallback must stop the job, not silently slow it down
            tmp = args.out / f"{cid}.pkl.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(args.out / f"{cid}.pkl")   # atomic: a half-written .pkl never counts as done
            with open(counts_file, "a", newline="", encoding="utf-8") as f:   # after the pkl; n_frames is also in the pkl
                f.write(f"{cid},{data['n_frames']}\n")
        except RuntimeError as e:
            if "not running on CUDA" in str(e):
                raise
            failed.append((cid, f"{type(e).__name__}: {e}"))
        except Exception as e:   # keep going; report everything at the end
            failed.append((cid, f"{type(e).__name__}: {e}"))
    print(f"done: {len(todo) - len(failed)} written, {len(failed)} failed")
    for cid, why in failed[:20]:
        print(f"  FAILED {cid}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
