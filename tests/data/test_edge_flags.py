"""data/make_edge_flags.py and manifests/edge_text_flags.csv. Run: pytest tests/data/test_edge_flags.py
The synthetic tests use made-up sentences; the checks on the real output need only the committed manifests
(the byte-for-byte regeneration check also needs local/videos.csv and is skipped without it).
"""
import csv
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
import make_edge_flags as mf  # noqa: E402

MAN = ROOT / "manifests"
FLAG_COLS = ["has_digit", "has_acronym", "long_sentence", "short_sentence"]
EXPECTED_COLUMNS = ["clip_id", "article_id", "signer_id", "split"] + FLAG_COLS
TEXT_LIKE = ("text", "caption", "transcript", "translation", "reference", "words")   # *_sentence flags are required names


def _no_text_columns(cols):
    assert cols == EXPECTED_COLUMNS, f"unexpected columns: {cols}"
    assert not [c for c in cols if any(w in c.lower() for w in TEXT_LIKE)], f"text-like column in {cols}"
VIDEO_HEADER = ["participant", "article", "section index", "sentence index", "video filename", "video length",
                "sentence text"]


# ---------------------------------------------------------------- synthetic fixtures
def _words(n, first="word"):
    return " ".join([first] + ["filler"] * (n - 1))


def _fake_dataset(tmp_path, val_test_len=60, shuffle_seed=None):
    """train: 20 clips with 1..20 words (clip t01..t20); val/test: a few crafted sentences."""
    train = {f"t{n:02d}": _words(n) for n in range(1, 21)}
    train["t05"] = "The 2 samples were measured"            # digit, 5 words
    train["t07"] = "We used DNA and mRNA in the iPhone"     # acronym (DNA) in 7 words
    train["t08"] = "The CO2 level rose over time"           # CO2 counts: CO touches only a digit
    train["t09"] = "Only mRNA and iPhone appear here now"   # no acronym
    val = {"v1": _words(val_test_len), "v2": "NASA's rocket flew", "v3": _words(10)}
    test = {"x1": _words(1), "x2": _words(val_test_len), "x3": "An ordinary short line here"}
    d = tmp_path / "man"
    d.mkdir(parents=True, exist_ok=True)
    for split, clips in (("train", train), ("val", val), ("test", test)):
        with open(d / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "article_id", "signer_id", "split"])
            for i, cid in enumerate(sorted(clips)):
                w.writerow([cid, str(100 + i), f"P{i % 3}", split])
    texts = {**train, **val, **test, "unused-clip": "Not in any manifest 123"}
    items = sorted(texts.items())
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(items)
    videos = tmp_path / "videos.csv"
    with open(videos, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(VIDEO_HEADER)
        for cid, text in items:
            w.writerow(["P9", "1", "0", "0", f"{cid}.mp4", "5.0", text])
    return d, videos, texts


def _run(tmp_path, d, videos, out_name="flags.csv"):
    out = tmp_path / out_name
    assert mf.main(["--videos", str(videos), "--manifests", str(d), "--out", str(out)]) == 0
    with open(out, newline="") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd), out


def _by_id(rows):
    return {r["clip_id"]: r for r in rows}


def test_row_count_columns_and_boolean_flags(tmp_path):
    d, videos, texts = _fake_dataset(tmp_path)
    cols, rows, _ = _run(tmp_path, d, videos)
    assert cols == EXPECTED_COLUMNS
    assert len(rows) == 20 + 3 + 3, "one row per manifest clip; the extra clip in videos.csv is ignored"
    assert all(r[c] in ("True", "False") for r in rows for c in FLAG_COLS)
    assert [r["clip_id"] for r in rows] == sorted(r["clip_id"] for r in rows)


def test_split_and_ids_agree_with_manifests(tmp_path):
    d, videos, _ = _fake_dataset(tmp_path)
    _, rows, _ = _run(tmp_path, d, videos)
    want = {}
    for split in ("train", "val", "test"):
        with open(d / f"{split}.csv", newline="") as f:
            for r in csv.DictReader(f):
                want[r["clip_id"]] = (r["article_id"], r["signer_id"], r["split"])
    got = {r["clip_id"]: (r["article_id"], r["signer_id"], r["split"]) for r in rows}
    assert got == want


def test_digit_and_acronym_flags(tmp_path):
    d, videos, _ = _fake_dataset(tmp_path)
    r = _by_id(_run(tmp_path, d, videos)[1])
    assert r["t05"]["has_digit"] == "True" and r["t05"]["has_acronym"] == "False"
    assert r["t07"]["has_acronym"] == "True" and r["t07"]["has_digit"] == "False"    # DNA yes, mRNA/iPhone no
    assert r["t08"]["has_digit"] == "True" and r["t08"]["has_acronym"] == "True"     # CO2
    assert r["t09"]["has_acronym"] == "False"
    assert r["v2"]["has_acronym"] == "True"                                           # NASA's
    assert r["x3"]["has_digit"] == "False" and r["x3"]["has_acronym"] == "False"


def test_length_flags_use_train_percentiles_only(tmp_path):
    d, videos, _ = _fake_dataset(tmp_path)
    r = _by_id(_run(tmp_path, d, videos)[1])
    # train lengths 1..20: p90 = 18.1, p10 = 2.9 (linear interpolation)
    assert [n for n in range(1, 21) if r[f"t{n:02d}"]["long_sentence"] == "True"] == [19, 20]
    assert [n for n in range(1, 21) if r[f"t{n:02d}"]["short_sentence"] == "True"] == [1, 2]
    assert r["v1"]["long_sentence"] == "True" and r["x2"]["long_sentence"] == "True"   # 60 words > train p90
    assert r["v3"]["long_sentence"] == "False" and r["v3"]["short_sentence"] == "False"
    assert r["x1"]["short_sentence"] == "True"
    # val/test lengths must not move the thresholds: change them wildly, train rows stay identical
    d2, videos2, _ = _fake_dataset(tmp_path / "other", val_test_len=2000)
    r2 = _by_id(_run(tmp_path / "other", d2, videos2)[1])
    assert {k: v for k, v in r.items() if k.startswith("t")} == {k: v for k, v in r2.items() if k.startswith("t")}


def test_deterministic_and_independent_of_videos_row_order(tmp_path):
    d, videos, _ = _fake_dataset(tmp_path)
    _, _, out1 = _run(tmp_path, d, videos, "a.csv")
    _, _, out2 = _run(tmp_path, d, videos, "b.csv")
    assert out1.read_bytes() == out2.read_bytes()
    d3, videos3, _ = _fake_dataset(tmp_path / "shuffled", shuffle_seed=7)
    _, _, out3 = _run(tmp_path / "shuffled", d3, videos3)
    assert out3.read_bytes() == out1.read_bytes()


def test_no_text_in_output_or_stdout(tmp_path, capsys):
    d, videos, texts = _fake_dataset(tmp_path)
    cols, rows, out = _run(tmp_path, d, videos)
    _no_text_columns(cols)
    blob = out.read_text() + capsys.readouterr().out + capsys.readouterr().err
    for text in ("samples were measured", "rocket flew", "ordinary short line", "level rose over time"):
        assert text not in blob
    assert all(len(r["clip_id"]) < 80 and " " not in r["clip_id"] for r in rows)


def test_missing_or_duplicate_clip_in_videos_stops_without_output(tmp_path):
    d, videos, texts = _fake_dataset(tmp_path)
    rows = [r for r in csv.reader(open(videos, newline="", encoding="utf-8-sig"))]
    cut = [r for r in rows if r[4] != "v1.mp4"]
    with open(videos, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(cut)
    with pytest.raises(SystemExit, match="no row"):
        mf.main(["--videos", str(videos), "--manifests", str(d), "--out", str(tmp_path / "o.csv")])
    with open(videos, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows + [rows[1]])
    with pytest.raises(SystemExit, match="more than once"):
        mf.main(["--videos", str(videos), "--manifests", str(d), "--out", str(tmp_path / "o.csv")])
    assert not (tmp_path / "o.csv").exists()


# ---------------------------------------------------------------- the real output
def _manifest_rows():
    out = {}
    for split in ("train", "val", "test"):
        with open(MAN / f"{split}.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[r["clip_id"]] = (r["article_id"], r["signer_id"], r["split"])
    return out


def _real_flags():
    p = MAN / "edge_text_flags.csv"
    assert p.exists(), f"{p} missing - run python data/make_edge_flags.py"
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def test_real_output_row_count_equals_total_clip_count():
    cols, rows = _real_flags()
    assert cols == EXPECTED_COLUMNS
    assert len(rows) == len(_manifest_rows())
    assert len({r["clip_id"] for r in rows}) == len(rows)


def test_real_output_flags_boolean_and_no_text_columns():
    cols, rows = _real_flags()
    assert all(r[c] in ("True", "False") for r in rows for c in FLAG_COLS)
    _no_text_columns(cols)
    assert all(len(r["clip_id"]) < 80 and " " not in r["clip_id"] and " " not in r["signer_id"] for r in rows)


def test_real_output_split_agrees_with_manifests():
    _, rows = _real_flags()
    want = _manifest_rows()
    assert {r["clip_id"]: (r["article_id"], r["signer_id"], r["split"]) for r in rows} == want


def test_real_output_regenerates_identically(tmp_path):
    videos = ROOT / "local" / "videos.csv"
    if not videos.exists():
        pytest.skip("local/videos.csv not present (git-ignored dataset index)")
    out = tmp_path / "again.csv"
    assert mf.main(["--out", str(out)]) == 0
    assert out.read_bytes() == (MAN / "edge_text_flags.csv").read_bytes(), "output is not deterministic / is stale"


# ---------------------------------------------------------------- manifests/edge_candidates_test.csv
CAND_COLUMNS = ["clip_id", "article_id", "signer_id", "split", "flag"]
TEXT_FLAG_NAMES = ["has_digit", "has_acronym", "long_sentence", "short_sentence"]


def _big_dataset(tmp_path):
    """train: 1..20 words (sets the percentiles). val and test both hold flagged clips, so a split leak would show.
    test: 70 with digits, 20 with acronyms, 60 long, 7 short (and some clips carry several flags)."""
    train = {f"t{n:02d}": _words(n) for n in range(1, 21)}
    val = {f"v{i:02d}": f"Val clip {i} with DNA and a very long tail " + "w " * 40 for i in range(15)}
    test = {}
    for i in range(70):
        test[f"x{i:03d}"] = f"Measured {i} samples today"            # digit
    for i in range(20):
        test[f"y{i:03d}"] = "The NASA team met"                       # acronym only
    for i in range(60):
        test[f"z{i:03d}"] = _words(60)                                # long only
    for i in range(7):
        test[f"s{i:03d}"] = "One"                                     # short only
    for i in range(10):
        test[f"m{i:03d}"] = "NASA 2 " + "w " * 50                     # digit + acronym + long
    d = tmp_path / "man"
    d.mkdir(parents=True, exist_ok=True)
    for split, clips in (("train", train), ("val", val), ("test", test)):
        with open(d / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "article_id", "signer_id", "split"])
            for i, cid in enumerate(sorted(clips)):
                w.writerow([cid, str(100 + i), f"P{i % 5}", split])
    videos = tmp_path / "videos.csv"
    with open(videos, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(VIDEO_HEADER)
        for cid, text in sorted({**train, **val, **test}.items()):
            w.writerow(["P9", "1", "0", "0", f"{cid}.mp4", "5.0", text])
    return d, videos, set(train), set(val), set(test)


def _run_big(tmp_path, name="flags.csv"):
    d, videos, train, val, test = _big_dataset(tmp_path)
    out = tmp_path / name
    assert mf.main(["--videos", str(videos), "--manifests", str(d), "--out", str(out)]) == 0
    cand = out.with_name("edge_candidates_test.csv")
    with open(cand, newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        return rd.fieldnames, rows, cand, out, (train, val, test)


def test_candidates_only_test_clips_and_at_most_50_per_flag(tmp_path):
    cols, rows, _, _, (train, val, test) = _run_big(tmp_path)
    assert cols == CAND_COLUMNS
    assert rows and all(r["split"] == "test" for r in rows)
    ids = {r["clip_id"] for r in rows}
    assert ids <= test and not ids & train and not ids & val, "candidates must come from the test split only"
    for flag in TEXT_FLAG_NAMES:
        n = sum(r["flag"] == flag for r in rows)
        assert n <= 50, f"{flag}: {n} candidates"
    assert {r["flag"] for r in rows} == set(TEXT_FLAG_NAMES)


def test_candidates_counts_when_pool_is_larger_or_smaller_than_50(tmp_path):
    _, rows, _, flags_out, _ = _run_big(tmp_path)
    by_flag = {f: [r for r in rows if r["flag"] == f] for f in TEXT_FLAG_NAMES}
    assert len(by_flag["has_digit"]) == 50        # 80 test clips have a digit
    assert len(by_flag["long_sentence"]) == 50    # 70 test clips are long
    assert len(by_flag["has_acronym"]) == 30      # only 30 exist: all of them are taken
    assert len(by_flag["short_sentence"]) == 7    # only 7 exist
    assert len({r["clip_id"] for r in by_flag["has_acronym"]}) == 30
    for f in TEXT_FLAG_NAMES:
        assert len({r["clip_id"] for r in by_flag[f]}) == len(by_flag[f]), f"{f}: a clip is listed twice"


def test_candidates_really_have_their_flag(tmp_path):
    _, rows, _, flags_out, _ = _run_big(tmp_path)
    col = dict(zip(TEXT_FLAG_NAMES, ["has_digit", "has_acronym", "long_sentence", "short_sentence"]))
    with open(flags_out, newline="") as f:
        flagged = {r["clip_id"]: r for r in csv.DictReader(f)}
    for r in rows:
        assert flagged[r["clip_id"]][col[r["flag"]]] == "True"
        assert flagged[r["clip_id"]]["split"] == "test"
        assert (flagged[r["clip_id"]]["article_id"], flagged[r["clip_id"]]["signer_id"]) == (r["article_id"], r["signer_id"])


def test_candidates_are_seeded_deterministic_and_not_just_the_first_50(tmp_path):
    _, rows1, cand1, _, _ = _run_big(tmp_path, "a.csv")
    _, rows2, cand2, _, _ = _run_big(tmp_path / "again", "b.csv")
    assert cand1.read_bytes() == cand2.read_bytes()
    digits = [r["clip_id"] for r in rows1 if r["flag"] == "has_digit"]
    assert digits == sorted(digits)
    pool = sorted(f"x{i:03d}" for i in range(70)) + sorted(f"m{i:03d}" for i in range(10))
    assert digits != sorted(pool)[:50], "sample must be random, not the first 50 by clip_id"


def test_candidates_contain_no_text(tmp_path, capsys):
    cols, rows, cand, _, _ = _run_big(tmp_path)
    assert cols == CAND_COLUMNS and not [c for c in cols if any(w in c.lower() for w in TEXT_LIKE)]
    blob = cand.read_text() + capsys.readouterr().out
    for fragment in ("samples today", "NASA team met", "Val clip"):
        assert fragment not in blob


def test_real_candidates_are_test_only_at_most_50_per_flag_and_agree_with_flags():
    p = MAN / "edge_candidates_test.csv"
    assert p.exists(), f"{p} missing - run python data/make_edge_flags.py"
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        assert rd.fieldnames == CAND_COLUMNS
        rows = list(rd)
    split_of = {}
    for split in ("train", "val", "test"):
        with open(MAN / f"{split}.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                split_of[r["clip_id"]] = split
    assert all(split_of[r["clip_id"]] == "test" and r["split"] == "test" for r in rows)
    assert not [r for r in rows if split_of[r["clip_id"]] in ("train", "val")]
    for flag in TEXT_FLAG_NAMES:
        assert 0 < sum(r["flag"] == flag for r in rows) <= 50
    _, flagged = _real_flags()
    flagged = {r["clip_id"]: r for r in flagged}
    assert all(flagged[r["clip_id"]][r["flag"]] == "True" for r in rows)
