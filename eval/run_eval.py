r"""Run one evaluation: one control, one dtype, one seed, over a manifest of clips.

Usage:  python eval/run_eval.py CONFIG.yaml      (run with the Uni-Sign python env)

Config (YAML, every field required, no defaults):
    unisign_dir  path to the Uni-Sign checkout (the model is built with it as cwd)
    checkpoint   path to the .pth checkpoint (loaded with strict=True)
    pose_dir     folder of <clip_id>.pkl pose files ({'keypoints', 'scores'})
    manifest     CSV with a clip_id column (ids may end in .mp4; the stem is used)
    references   CSV with columns (clip_id, reference) or ('video filename',
                 'sentence text'), as eval_translation.py read_refs
    control      real | zeroed | shuffled
    dtype        bf16 | fp32
    seed         int (shuffle control and the model-build seed)
    max_frames   int, frame cap applied with eval/subsample.py
    out_dir      folder for pred_<...>.csv and metrics_<...>.json

Per clip, in manifest order: load pose -> control (eval/controls.py) ->
subsample_pose (eval/subsample.py) -> translate.

Generation is fixed in code: num_beams=4, max_new_tokens=100, as in
Uni-Sign/run_batch_unisign.py line 59 (translate, lines 52-60). The prompt language
is English because the Uni_Sign args use --dataset OpenASL (models.py lines 104-107
choose 'Chinese' for any dataset name containing CSL). The model path (construction,
strict checkpoint load, model.eval(), cast) mirrors run_batch_unisign.py lines 72-85,
with the cast taken from the config instead of the hard-coded bfloat16, and the
inputs cast to the same dtype (the original translate() hard-codes bfloat16 on
line 58, which would not match an fp32 model).
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval import bootstrap, controls, metrics, subsample  # noqa: E402

REQUIRED = ["unisign_dir", "checkpoint", "pose_dir", "manifest", "references",
            "control", "dtype", "seed", "max_frames", "out_dir"]
CONTROLS = ("real", "zeroed", "shuffled")
DTYPES = ("bf16", "fp32")
NUM_BEAMS = 4          # run_batch_unisign.py:59
MAX_NEW_TOKENS = 100   # run_batch_unisign.py:59
N_RESAMPLES = 1000


# ---------------------------------------------------------------- config
def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    missing = [k for k in REQUIRED if k not in cfg or cfg[k] is None]
    if missing:
        raise ValueError(f"config is missing required fields: {missing}")
    if cfg["control"] not in CONTROLS:
        raise ValueError(f"control must be one of {CONTROLS}, got {cfg['control']!r}")
    if cfg["dtype"] not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {cfg['dtype']!r}")
    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int):
        raise ValueError("seed must be an int")
    if isinstance(cfg["max_frames"], bool) or not isinstance(cfg["max_frames"], int) or cfg["max_frames"] < 1:
        raise ValueError("max_frames must be an int >= 1")
    return cfg


def load_config(path):
    import yaml  # PyYAML, listed in requirements.txt

    with open(path, encoding="utf-8") as fh:
        return validate_config(yaml.safe_load(fh))


# ---------------------------------------------------------------- data
def clip_stem(clip_id):
    return os.path.splitext(os.path.basename(clip_id.strip()))[0]


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        if "clip_id" not in (rd.fieldnames or []):
            raise ValueError(f"{path}: needs a clip_id column")
        return [clip_stem(r["clip_id"]) for r in rd]


def read_references(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames or []
        kid = "clip_id" if "clip_id" in cols else "video filename"
        kref = "reference" if "reference" in cols else "sentence text"
        if kid not in cols or kref not in cols:
            raise ValueError(f"{path}: need clip_id,reference or 'video filename','sentence text'")
        return {clip_stem(r[kid]): r[kref].strip() for r in rd}


def make_pose_loader(pose_dir):
    import pickle

    def load(clip_id):
        with open(os.path.join(pose_dir, clip_id + ".pkl"), "rb") as fh:
            return pickle.load(fh)

    return load


# ---------------------------------------------------------------- core loop
def run_clips(clip_ids, load_pose, control, max_frames, seed, translate_fn):
    """Return the hypotheses for clip_ids, in order.

    translate_fn(pose_dict) -> str. torch-free, so it can be tested with a fake.
    """
    if control not in CONTROLS:
        raise ValueError(f"unknown control {control!r}")
    mapping = controls.shuffle_mapping(clip_ids, seed) if control == "shuffled" else None
    hyps = []
    for cid in clip_ids:
        if control == "real":
            pose = load_pose(cid)
        elif control == "zeroed":
            pose = controls.zeroed_pose(load_pose(cid))
        else:
            pose = load_pose(mapping[cid])
        pose = subsample.subsample_pose(pose, max_frames)
        hyps.append(translate_fn(pose))
    return hyps


# ---------------------------------------------------------------- outputs
def git_info():
    def run(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)

    h = run("rev-parse", "HEAD")
    st = run("status", "--porcelain")
    return {
        "git_hash": h.stdout.strip() if h.returncode == 0 else "unavailable (no commits)",
        "git_dirty": bool(st.stdout.strip()) if st.returncode == 0 else None,
    }


def compute_metrics(hyps, refs, cfg, device, versions):
    out = {"n_clips": len(hyps), "control": cfg["control"], "dtype": cfg["dtype"],
           "seed": cfg["seed"], "max_frames": cfg["max_frames"],
           "num_beams": NUM_BEAMS, "max_new_tokens": MAX_NEW_TOKENS,
           "n_resamples": N_RESAMPLES}
    for name, fn in (("bleu", metrics.corpus_bleu), ("chrf", metrics.chrf), ("rouge_l", metrics.rouge_l)):
        out[name] = bootstrap.bootstrap_ci(hyps, refs, fn, N_RESAMPLES, seed=cfg["seed"])
    out.update(git_info())
    out["config"] = dict(cfg)
    out["device"] = device
    out.update(versions)
    return out


def _tag(cfg):
    return f"{cfg['control']}_{cfg['dtype']}_seed{cfg['seed']}"


def write_predictions(out_dir, clip_ids, hyps, cfg):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred = out / f"pred_{_tag(cfg)}.csv"
    with open(pred, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "condition", "hypothesis"])
        w.writeheader()
        for cid, h in zip(clip_ids, hyps):
            w.writerow({"id": cid, "condition": cfg["control"], "hypothesis": h})
    return pred


def write_metrics(out_dir, cfg, result):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mj = out / f"metrics_{_tag(cfg)}.json"
    with open(mj, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return mj


def write_outputs(out_dir, clip_ids, hyps, cfg, result):
    return write_predictions(out_dir, clip_ids, hyps, cfg), write_metrics(out_dir, cfg, result)


# ---------------------------------------------------------------- model (torch)
def build_translate(cfg):
    """Build the Uni-Sign model and return (translate_fn, device, versions)."""
    import argparse
    import importlib.metadata as md

    unisign_dir = os.path.abspath(cfg["unisign_dir"])
    checkpoint = os.path.abspath(cfg["checkpoint"])
    os.chdir(unisign_dir)  # config.py uses ./pretrained_weight/mt5-base
    sys.path.insert(0, unisign_dir)
    import torch
    from torch.utils.data import DataLoader
    import run_batch_unisign as rb  # also installs the deepspeed/decord stubs

    dtype = torch.bfloat16 if cfg["dtype"] == "bf16" else torch.float32
    parser = argparse.ArgumentParser(parents=[rb.utils.get_args_parser()])
    args = parser.parse_args(["--finetune", checkpoint, "--dataset", "OpenASL",
                              "--seed", str(cfg["seed"]), "--max_length", str(cfg["max_frames"])])
    rb.utils.set_seed(args.seed)
    ds = rb.S2T_Dataset_online(args=args)
    ds.rgb_data = None
    model = rb.Uni_Sign(args=args).cuda()
    model.train()
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    sd = torch.load(args.finetune, map_location="cpu")["model"]
    ret = model.load_state_dict(sd, strict=True)
    if ret.missing_keys or ret.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {ret.missing_keys} {ret.unexpected_keys}")
    model.eval()
    model.to(dtype)

    @torch.no_grad()
    def translate(pose):
        ds.pose_data = pose
        dl = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)
        for src_input, tgt_input in dl:
            for k in src_input.keys():
                if isinstance(src_input[k], torch.Tensor):
                    src_input[k] = src_input[k].to(dtype).cuda()
            out = model.generate(model(src_input, tgt_input),
                                 max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS)
            return model.mt5_tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    import transformers
    versions = {"torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "sacrebleu_version": md.version("sacrebleu")}
    return translate, str(next(model.parameters()).device), versions


# ---------------------------------------------------------------- resources
def cuda_available():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def start_tracking(track_gpu):
    """Start the wall clock and, if track_gpu, reset the CUDA peak-memory counter."""
    if track_gpu:
        import torch

        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def resource_fields(t_start, track_gpu):
    """wall_time_s since t_start and peak_gpu_mem_mb (torch.cuda.max_memory_allocated).

    peak_gpu_mem_mb is None when track_gpu is False (no CUDA, or a fake translate
    function was used).
    """
    peak = None
    if track_gpu:
        import torch

        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return {"wall_time_s": time.perf_counter() - t_start, "peak_gpu_mem_mb": peak}


# ---------------------------------------------------------------- run
def evaluate_run(clip_ids, refs, cfg, load_pose, translate_fn, device, versions,
                 track_gpu=False, t_start=None):
    """Loop, write predictions, compute metrics, add resource fields, write metrics.

    Predictions are written before the metrics so a metrics failure cannot lose a
    finished run. Returns (result, pred_path, metrics_path). t_start comes from
    start_tracking(); if None, the clock and the GPU peak counter start here.
    """
    if t_start is None:
        t_start = start_tracking(track_gpu)
    hyps = run_clips(clip_ids, load_pose, cfg["control"], cfg["max_frames"], cfg["seed"], translate_fn)
    pred = write_predictions(cfg["out_dir"], clip_ids, hyps, cfg)
    result = compute_metrics(hyps, refs, cfg, device, versions)
    result.update(resource_fields(t_start, track_gpu))
    mj = write_metrics(cfg["out_dir"], cfg, result)
    return result, pred, mj


# ---------------------------------------------------------------- main
def main(config_path):
    cfg = load_config(config_path)
    # resolve inputs before the model build changes the working directory
    for k in ("unisign_dir", "checkpoint", "pose_dir", "manifest", "references", "out_dir"):
        cfg[k] = os.path.abspath(cfg[k])
    track_gpu = cuda_available()
    t_start = start_tracking(track_gpu)  # clock and peak counter start before the model build
    clip_ids = read_manifest(cfg["manifest"])
    refs_all = read_references(cfg["references"])
    missing = [c for c in clip_ids if c not in refs_all]
    if missing:
        raise ValueError(f"{len(missing)} manifest clips have no reference, e.g. {missing[:3]}")
    refs = [refs_all[c] for c in clip_ids]
    load_pose = make_pose_loader(cfg["pose_dir"])
    translate, device, versions = build_translate(cfg)
    result, pred, mj = evaluate_run(clip_ids, refs, cfg, load_pose, translate, device, versions,
                                    track_gpu=track_gpu, t_start=t_start)
    print("predictions:", pred)
    print("metrics:", mj)
    for m in ("bleu", "chrf", "rouge_l"):
        r = result[m]
        print(f"{m}: {r['estimate']:.2f}  95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
    print(f"wall_time_s: {result['wall_time_s']:.1f}  peak_gpu_mem_mb: {result['peak_gpu_mem_mb']}")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python eval/run_eval.py CONFIG.yaml")
    main(sys.argv[1])
