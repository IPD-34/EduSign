"""run_eval checks without torch or the model (made-up poses and sentences)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval import run_eval  # noqa: E402

CFG = {
    "unisign_dir": "u", "checkpoint": "c.pth", "pose_dir": "p", "manifest": "m.csv",
    "references": "r.csv", "control": "real", "dtype": "fp32", "seed": 42,
    "max_frames": 8, "out_dir": "o",
}
IDS = [f"clip_{i}" for i in range(6)]
REFS = ["the cat sat", "a dog ran far", "we measured heat", "water boils fast",
        "plants need light", "the model learns"]


def _pose(n, value):
    return {"keypoints": [np.full((1, 133, 2), value, dtype=np.float32) for _ in range(n)],
            "scores": [np.full((1, 133), value, dtype=np.float32) for _ in range(n)]}


def _loader(cid):
    i = IDS.index(cid)
    return _pose(5 + 4 * i, float(i + 1))  # lengths 5..25, so some exceed max_frames=8


def _fake_translate(pose):
    # echo the frame count and the pose value so the control and the cap are visible
    return f"n={len(pose['scores'])} v={float(pose['scores'][0].max()):.0f}"


@pytest.mark.parametrize("drop", ["dtype", "seed"])
def test_missing_dtype_or_seed_raises(drop):
    cfg = {k: v for k, v in CFG.items() if k != drop}
    with pytest.raises(ValueError, match=drop):
        run_eval.validate_config(cfg)


def test_valid_config_and_bad_values():
    assert run_eval.validate_config(dict(CFG)) == CFG
    with pytest.raises(ValueError):
        run_eval.validate_config({**CFG, "dtype": "fp16"})
    with pytest.raises(ValueError):
        run_eval.validate_config({**CFG, "control": "noise"})


def test_loop_real_applies_cap_and_order():
    out = run_eval.run_clips(IDS, _loader, "real", 8, 42, _fake_translate)
    assert out[0] == "n=5 v=1"
    assert all(int(o.split()[0][2:]) <= 8 for o in out)
    assert out[5] == "n=8 v=6"


def test_loop_zeroed_and_shuffled():
    z = run_eval.run_clips(IDS, _loader, "zeroed", 8, 42, _fake_translate)
    assert all(o.endswith("v=0") for o in z)
    s = run_eval.run_clips(IDS, _loader, "shuffled", 8, 42, _fake_translate)
    real = run_eval.run_clips(IDS, _loader, "real", 8, 42, _fake_translate)
    assert all(a != b for a, b in zip(s, real))  # nobody keeps their own pose
    assert s == run_eval.run_clips(IDS, _loader, "shuffled", 8, 42, _fake_translate)


def test_metrics_json_has_all_fields(tmp_path):
    hyps = run_eval.run_clips(IDS, _loader, "real", 8, 42, lambda p: "the cat sat")
    res = run_eval.compute_metrics(hyps, REFS, CFG, "cpu",
                                   {"torch_version": "x", "transformers_version": "y"})
    for k in ("bleu", "chrf", "rouge_l"):
        assert {"estimate", "ci_low", "ci_high"} <= set(res[k])
    for k in ("n_clips", "control", "dtype", "seed", "git_hash", "config", "device",
              "torch_version", "transformers_version"):
        assert k in res
    assert res["n_clips"] == 6 and res["config"] == CFG and res["device"] == "cpu"
    pred, mj = run_eval.write_outputs(str(tmp_path), IDS, hyps, CFG, res)
    assert os.path.basename(pred).startswith("pred_")
    with open(mj, encoding="utf-8") as fh:
        assert json.load(fh)["seed"] == 42


def _write_yaml(path, cfg):
    import yaml

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)


def test_load_config_reads_yaml_and_requires_dtype_and_seed(tmp_path):
    pytest.importorskip("yaml")
    good = tmp_path / "good.yaml"
    _write_yaml(good, CFG)
    cfg = run_eval.load_config(str(good))
    for k in run_eval.REQUIRED:
        assert cfg[k] == CFG[k]
    for drop in ("dtype", "seed"):
        bad = tmp_path / f"no_{drop}.yaml"
        _write_yaml(bad, {k: v for k, v in CFG.items() if k != drop})
        with pytest.raises(ValueError, match=drop):
            run_eval.load_config(str(bad))


def test_resource_fields_in_json_with_fake_translate(tmp_path):
    cfg = {**CFG, "out_dir": str(tmp_path)}
    result, pred, mj = run_eval.evaluate_run(
        IDS, REFS, cfg, _loader, lambda p: "the cat sat", "cpu",
        {"torch_version": None, "transformers_version": None})
    with open(mj, encoding="utf-8") as fh:
        data = json.load(fh)
    assert "wall_time_s" in data and "peak_gpu_mem_mb" in data
    assert isinstance(data["wall_time_s"], float) and data["wall_time_s"] >= 0
    assert data["peak_gpu_mem_mb"] is None  # fake translate: no GPU tracking
    assert os.path.exists(pred)
