"""Metric checks on made-up sentences (no dataset content)."""
import os
import sys

import pytest
import sacrebleu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval import metrics  # noqa: E402

HYPS = [
    "the cat sat on the mat",
    "a quick brown fox jumps over the dog",
    "we measured the energy of the system",
    "water boils at one hundred degrees",
]
REFS = [
    "the cat is sitting on the mat",
    "the quick brown fox jumped over the lazy dog",
    "we measured the total energy of the system",
    "water boils at 100 degrees celsius",
]


def test_bleu_matches_sacrebleu():
    expected = sacrebleu.corpus_bleu(HYPS, [REFS]).score
    assert metrics.corpus_bleu(HYPS, REFS) == expected


def test_chrf_matches_sacrebleu():
    expected = sacrebleu.corpus_chrf(HYPS, [REFS]).score
    assert metrics.chrf(HYPS, REFS) == expected


def test_perfect_match_scores_100():
    assert metrics.corpus_bleu(REFS, REFS) == pytest.approx(100.0)
    assert metrics.chrf(REFS, REFS) == pytest.approx(100.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.corpus_bleu(HYPS, REFS[:-1])


def test_rouge_l_identical_is_100():
    pytest.importorskip("rouge")
    assert metrics.rouge_l(REFS, REFS) == pytest.approx(100.0)
