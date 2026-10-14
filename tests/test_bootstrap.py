"""Bootstrap checks on made-up sentences (no dataset content)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval import metrics  # noqa: E402
from eval.bootstrap import bootstrap_ci, group_table, paired_bootstrap  # noqa: E402

REFS = [
    "the cat is sitting on the mat",
    "the quick brown fox jumped over the lazy dog",
    "we measured the total energy of the system",
    "water boils at 100 degrees celsius",
    "the experiment was repeated three times",
    "heat flows from hot objects to cold objects",
    "the graph shows a linear relationship",
    "plants convert light into chemical energy",
    "the sample was stored at low temperature",
    "each student completed the lab report",
    "the model predicts the next word",
    "gravity pulls objects toward the earth",
]
# light edits: mostly right, a few words wrong
HYPS_MID = [
    "the cat sat on the mat",
    "a quick brown fox jumps over the dog",
    "we measured the energy of the system",
    "water boils at one hundred degrees",
    "the experiment was repeated twice",
    "heat flows from hot to cold objects",
    "the graph shows a straight line",
    "plants turn light into energy",
    "the sample was kept cold",
    "each student finished the report",
    "the model guesses the next word",
    "gravity pulls things to the ground",
]
# deliberately degraded: unrelated text
HYPS_BAD = ["lorem ipsum dolor sit amet"] * len(REFS)


def test_same_seed_identical_results():
    a = bootstrap_ci(HYPS_MID, REFS, metrics.chrf, 200, seed=7)
    b = bootstrap_ci(HYPS_MID, REFS, metrics.chrf, 200, seed=7)
    assert a == b
    p = paired_bootstrap(HYPS_BAD, HYPS_MID, REFS, metrics.corpus_bleu, 200, seed=7)
    q = paired_bootstrap(HYPS_BAD, HYPS_MID, REFS, metrics.corpus_bleu, 200, seed=7)
    assert p == q


def test_identical_systems_delta_zero_p_one():
    r = paired_bootstrap(HYPS_MID, HYPS_MID, REFS, metrics.chrf, 200, seed=1)
    assert r["delta"] == 0.0
    assert r["p_value"] == 1.0
    assert r["delta_ci_low"] == 0.0 and r["delta_ci_high"] == 0.0


def test_perfect_vs_degraded_is_significant():
    r = paired_bootstrap(HYPS_BAD, REFS, REFS, metrics.chrf, 500, seed=3)
    assert r["delta"] > 0
    assert r["p_value"] < 0.05
    assert r["delta"] == pytest.approx(r["metric_b"] - r["metric_a"])


def test_ci_contains_point_estimate():
    for fn in (metrics.chrf, metrics.corpus_bleu):
        r = bootstrap_ci(HYPS_MID, REFS, fn, 500, seed=11)
        assert r["ci_low"] <= r["estimate"] <= r["ci_high"]


def test_group_table_counts_sum_to_total():
    groups = ["s1", "s2", "s3"] * 4
    df = pd.DataFrame({"hyp": HYPS_MID, "ref": REFS, "g": groups})
    t = group_table(df, "hyp", "ref", "g", metrics.chrf)
    assert t["n_clips"].sum() == len(df)
    assert list(t["g"]) == ["s1", "s2", "s3"]
    for _, row in t.iterrows():
        sub = df[df.g == row["g"]]
        assert row["score"] == metrics.chrf(sub.hyp.tolist(), sub.ref.tolist())
