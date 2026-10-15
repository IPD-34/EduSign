"""--shard i/N behaviour of data/extract_rtmw.py and data/time_rtmw.py, and the CUDA assertion under CUDA_VISIBLE_DEVICES.
Run: pytest tests/data/test_shard.py
"""
import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
import extract_rtmw as ex  # noqa: E402

IDS = [f"{h}-{a}_{s}_{k}" for h in (3, 11, 7) for a in (5, 40) for s in range(3) for k in range(4)]   # unsorted, 72 ids


def test_parse_shard():
    assert ex.parse_shard("0/1") == (0, 1)
    assert ex.parse_shard("1/2") == (1, 2)
    for bad in ("2/2", "-1/2", "0/0", "a/b", "1", "1/2/3"):
        with pytest.raises(argparse.ArgumentTypeError):
            ex.parse_shard(bad)


def test_shards_disjoint_and_cover_all():
    s0, s1 = ex.shard_ids(IDS, 0, 2), ex.shard_ids(IDS, 1, 2)
    assert not set(s0) & set(s1), "shards overlap"
    assert sorted(s0 + s1) == sorted(IDS), "shards do not cover every clip"
    assert abs(len(s0) - len(s1)) <= 1
    # index % N == i over the sorted list
    ordered = sorted(IDS)
    assert s0 == ordered[0::2] and s1 == ordered[1::2]


def test_default_shard_is_everything():
    assert ex.shard_ids(IDS) == sorted(IDS)
    assert ex.shard_ids(IDS, 0, 1) == sorted(IDS)


def test_three_way_shards_partition():
    parts = [ex.shard_ids(IDS, i, 3) for i in range(3)]
    assert sorted(sum(parts, [])) == sorted(IDS)
    assert len({c for p in parts for c in p}) == len(IDS)


def test_shards_of_real_manifest_are_disjoint_and_cover(tmp_path):
    manifest = ROOT / "manifests" / "eval300.csv"
    if not manifest.exists():
        pytest.skip("manifests/eval300.csv not generated")
    ids = ex.read_clip_ids(manifest)
    s0, s1 = ex.shard_ids(ids, 0, 2), ex.shard_ids(ids, 1, 2)
    assert not set(s0) & set(s1)
    assert set(s0) | set(s1) == set(ids) and len(s0) + len(s1) == len(ids)


def test_main_processes_only_its_shard(tmp_path, monkeypatch):
    """Extraction itself is untouched: with a stub extractor, shard 0/2 and 1/2 write disjoint, complete .pkl sets."""
    manifest = tmp_path / "m.csv"
    ids = [f"c{k:02d}" for k in range(7)]
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "article_id", "signer_id", "split"])
        for c in ids:
            w.writerow([c, "1", "P0", "test"])
    clips = tmp_path / "clips"
    clips.mkdir()
    for c in ids:
        (clips / f"{c}.mp4").write_bytes(b"x")
    monkeypatch.setattr(ex, "build_extractor", lambda *a, **k: object())
    monkeypatch.setattr(ex, "extract_clip", lambda wb, p: {"keypoints": [], "scores": []})
    monkeypatch.setattr(ex, "assert_on_cuda", lambda wb: None)
    out = tmp_path / "out"
    written = []
    for i in (0, 1):
        before = {p.name for p in out.glob("*.pkl")} if out.exists() else set()
        assert ex.main(["--manifest", str(manifest), "--clips", str(clips), "--out", str(out),
                        "--shard", f"{i}/2"]) == 0
        written.append({p.name for p in out.glob("*.pkl")} - before)
    assert not written[0] & written[1]
    assert written[0] | written[1] == {f"{c}.pkl" for c in ids}


def _wb(device, providers, device_id):
    sess = SimpleNamespace(get_providers=lambda: providers,
                           get_provider_options=lambda: {"CUDAExecutionProvider": {"device_id": str(device_id)}})
    tool = SimpleNamespace(session=sess, device=device)
    return SimpleNamespace(det_model=tool, pose_model=tool)


CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_assert_cuda_single_visible_gpu(monkeypatch):
    # physical GPU 1 is the only visible one: it is device 0 inside the process
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    ex.assert_on_cuda(_wb("cuda", CUDA, 0))
    ex.assert_on_cuda(_wb("cuda:0", CUDA, 0))
    with pytest.raises(RuntimeError, match="exposes only 1"):
        ex.assert_on_cuda(_wb("cuda:1", CUDA, 1))
    with pytest.raises(RuntimeError, match="not running on CUDA"):
        ex.assert_on_cuda(_wb("cuda", ["CPUExecutionProvider"], 0))


def test_assert_cuda_without_env_and_cpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    ex.assert_on_cuda(_wb("cuda:1", CUDA, 1))
    with pytest.raises(RuntimeError, match="device_id 0, expected 1"):
        ex.assert_on_cuda(_wb("cuda:1", CUDA, 0))
    ex.assert_on_cuda(_wb("cpu", ["CPUExecutionProvider"], 0))   # cpu extractor: nothing to assert


def test_physical_gpu_mapping():
    assert ex.physical_gpu("cuda", {}) == "0"
    assert ex.physical_gpu("cuda:1", {}) == "1"
    assert ex.physical_gpu("cuda", {"CUDA_VISIBLE_DEVICES": "1"}) == "1"
    assert ex.physical_gpu("cuda:1", {"CUDA_VISIBLE_DEVICES": "3,2"}) == "2"
    assert ex.physical_gpu("cuda:5", {"CUDA_VISIBLE_DEVICES": "3,2"}) is None
