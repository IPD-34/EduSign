"""Extract RTMW whole-body keypoints (rtmlib 'lightweight': YOLOX-tiny + RTMW-dw-l-m 256x192) for every clip in a manifest.

For each manifest row, reads <clips>/<clip_id>.mp4 and writes <out>/<clip_id>.pkl:
    {"keypoints": [ndarray (1, 133, 2) float32, ...],   # one per frame, x/frame_width, y/frame_height
     "scores":    [ndarray (1, 133)    float32, ...]}   # one per frame
Several people detected -> the one with the largest bounding box is kept. None detected -> zeros for both.
Existing .pkl files are skipped, so an interrupted run (e.g. a Kaggle session) can simply be restarted.

All paths come from the command line, so the same script runs locally and on Kaggle:
    python data/extract_rtmw.py --manifest manifests/eval300.csv --clips <clip_folder> --out <pose_folder>
Optional: --device cuda|cuda:1|cpu (default cuda; fails loudly if the CUDA provider cannot be created),
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


def providers_in_use(wb):
    """Providers the live onnxruntime sessions are actually using (first entry is what runs the graph)."""
    return {"detector": wb.det_model.session.get_providers(), "pose": wb.pose_model.session.get_providers()}


def assert_on_cuda(wb):
    """Raise unless both sessions run on CUDAExecutionProvider (no-op for an explicit cpu extractor)."""
    if not wb.pose_model.device.startswith("cuda"):
        return
    for name, active in providers_in_use(wb).items():
        if not active or active[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{name} is not running on CUDA (providers: {active}); onnxruntime fell back to CPU. "
                               f"Check the CUDA 12 / cuDNN 9 DLLs or pass --device cpu explicitly.")


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


def extract_clip(wb, video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    keypoints, scores = [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            k, s = extract_frame(wb, frame)
            k = k.copy()
            k[..., 0] /= w
            k[..., 1] /= h
            keypoints.append(k)
            scores.append(s)
    finally:
        cap.release()
    if not keypoints:
        raise IOError(f"no frames decoded from {video_path}")
    return {"keypoints": keypoints, "scores": scores}


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
    args = ap.parse_args(argv)

    clip_ids = read_clip_ids(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in clip_ids if not (args.out / f"{c}.pkl").exists()]
    print(f"{len(clip_ids)} clips in manifest, {len(clip_ids) - len(todo)} already done, {len(todo)} to extract")
    if not todo:
        return 0

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
            tmp =args.out / f"{cid}.pkl.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(args.out / f"{cid}.pkl")   # atomic: a half-written .pkl never counts as done
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
