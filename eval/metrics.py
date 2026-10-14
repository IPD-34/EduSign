"""Corpus-level translation metrics for EduSign.

BLEU and chrF replicate the calls in eval_translation.py (cmd_score), which the
baseline numbers were produced with:

    sacrebleu.corpus_bleu(hyps, [refs]).score
    sacrebleu.corpus_chrf(hyps, [refs]).score

Settings: sacrebleu defaults only. BLEU uses the "13a" tokenizer, exponential
smoothing, case-sensitive, no lowercasing. chrF uses character n-gram order 6,
word order 0, beta 2, case-sensitive. No preprocessing is applied to hyps or
refs: the caller passes the strings as they are (eval_translation.py only
strips whitespace when reading its input files, which is the caller's job).
Scores are on a 0-100 scale. One reference per hypothesis.

ROUGE-L: eval_translation.py does not compute ROUGE-L, so this module uses the
`rouge` package listed in requirements.txt (Rouge().get_scores(..., avg=True),
"rouge-l" f-measure, multiplied by 100 to match the BLEU/chrF scale). Tokenisation
and casing are whatever that package does; they are not the same code as
Uni-Sign's vendored ROUGE in SLRT_metrics.py, so do not compare ROUGE numbers
across the two.
"""
import sacrebleu


def _check(hyps, refs):
    if len(hyps) != len(refs):
        raise ValueError(f"hyps and refs differ in length: {len(hyps)} vs {len(refs)}")
    if len(hyps) == 0:
        raise ValueError("empty input")


def corpus_bleu(hyps, refs):
    """Corpus BLEU (0-100), identical to sacrebleu.corpus_bleu(hyps, [refs]).score."""
    _check(hyps, refs)
    return sacrebleu.corpus_bleu(list(hyps), [list(refs)]).score


def chrf(hyps, refs):
    """chrF (0-100), identical to sacrebleu.corpus_chrf(hyps, [refs]).score."""
    _check(hyps, refs)
    return sacrebleu.corpus_chrf(list(hyps), [list(refs)]).score


def rouge_l(hyps, refs):
    """Mean ROUGE-L f-measure (0-100) from the `rouge` package."""
    _check(hyps, refs)
    from rouge import Rouge  # imported lazily so BLEU/chrF work without it

    scores = Rouge().get_scores(list(hyps), list(refs), avg=True)
    return scores["rouge-l"]["f"] * 100
