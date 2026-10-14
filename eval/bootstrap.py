"""Bootstrap confidence intervals and paired comparison for corpus-level metrics.

A metric function has the signature metric_fn(hyps, refs) -> float (for example
eval.metrics.corpus_bleu or eval.metrics.chrf). Every resample recomputes the
CORPUS metric on the resampled clips; per-clip scores are never averaged.
The seed is always an explicit argument and fully determines the output.
"""
import numpy as np
import pandas as pd


def _validate(n_resamples, *seqs):
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    n = len(seqs[0])
    if n == 0:
        raise ValueError("empty input")
    if any(len(s) != n for s in seqs):
        raise ValueError("inputs differ in length")
    return n


def _take(seq, idx):
    return [seq[i] for i in idx]


def bootstrap_ci(hyps, refs, metric_fn, n_resamples=1000, *, seed):
    """Percentile bootstrap CI of a corpus metric.

    Resamples clip indices with replacement, recomputes metric_fn on each
    resampled corpus, and returns the point estimate on the full set with the
    95% percentile interval (2.5th and 97.5th percentiles of the resamples).

    Returns a dict: estimate, ci_low, ci_high, n_clips, n_resamples, seed.
    """
    n = _validate(n_resamples, hyps, refs)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores[b] = metric_fn(_take(hyps, idx), _take(refs, idx))
    low, high = np.percentile(scores, [2.5, 97.5])
    return {
        "estimate": float(metric_fn(list(hyps), list(refs))),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_clips": n,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def paired_bootstrap(hyps_a, hyps_b, refs, metric_fn, n_resamples=1000, *, seed):
    """Paired percentile bootstrap of system B against system A.

    The SAME resampled clip indices are used for both systems in every resample.
    delta = metric(B) - metric(A), so a positive delta means B is better.

    p_value is one-sided: the fraction of resamples where delta <= 0, i.e. how
    often B is not better than A. A small p_value supports "B better than A".
    It is not a two-sided test.

    Returns a dict: metric_a, metric_b, delta (all on the full set), delta_ci_low,
    delta_ci_high (95% percentile CI of delta), p_value, n_clips, n_resamples, seed.
    """
    n = _validate(n_resamples, hyps_a, hyps_b, refs)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        r = _take(refs, idx)
        deltas[b] = metric_fn(_take(hyps_b, idx), r) - metric_fn(_take(hyps_a, idx), r)
    low, high = np.percentile(deltas, [2.5, 97.5])
    metric_a = float(metric_fn(list(hyps_a), list(refs)))
    metric_b = float(metric_fn(list(hyps_b), list(refs)))
    return {
        "metric_a": metric_a,
        "metric_b": metric_b,
        "delta": metric_b - metric_a,
        "delta_ci_low": float(low),
        "delta_ci_high": float(high),
        "p_value": float(np.mean(deltas <= 0)),
        "n_clips": n,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def group_table(df, hyps_col, refs_col, group_col, metric_fn):
    """Corpus metric per value of group_col (for per-signer / per-article tables).

    Returns a DataFrame with columns [group_col, "n_clips", "score"], one row per
    group, sorted by group. No confidence intervals.
    """
    rows = []
    for key, g in df.groupby(group_col, sort=True):
        rows.append({
            group_col: key,
            "n_clips": len(g),
            "score": float(metric_fn(g[hyps_col].tolist(), g[refs_col].tolist())),
        })
    return pd.DataFrame(rows, columns=[group_col, "n_clips", "score"])
