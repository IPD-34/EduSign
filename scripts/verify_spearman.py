"""
verify_spearman.py

Independent verification of robustness/mc_dropout.py's hand-rolled
spearman_correlation() (no scipy dependency, used to correlate MC-dropout
variance against error rate). That function has so far only been checked
against its own unit tests (e.g. the synthetic "0.878" case in
test_mc_dropout.py) -- this script checks it against machinery that does
NOT share code with it, so a bug in the shared logic (rank assignment,
tie handling) can't hide by agreeing with itself.

Two independent references, computed for every test case:
  1. A FRESH from-scratch implementation written in THIS file --
     average_rank() and pearson() below do not call, import, or copy
     robustness.mc_dropout's _rankdata/_pearson_correlation. Written in
     a different style (itertools.groupby over sorted (value, index)
     pairs) specifically so it isn't a disguised copy of the module
     under test.
  2. scipy.stats.spearmanr -- a widely-used, independently-implemented,
     already-validated library function. (scipy is only a verification
     tool here, in a standalone script; it is deliberately NOT a
     dependency of mc_dropout.py itself, which has its own reasons for
     avoiding it -- see that module's docstring.)

Test cases are hand-constructed, not randomly generated, specifically to
include tied blocks (including a block of 3+ equal values, and a case
tied on both sides), since average-rank tie correction is where a
hand-rolled rank-correlation implementation most commonly breaks.

Usage:
    uv run python -m scripts.verify_spearman
"""

from __future__ import annotations

from itertools import groupby

from robustness.mc_dropout import spearman_correlation as mc_dropout_spearman

try:
    from scipy.stats import spearmanr as scipy_spearmanr
except ImportError:
    scipy_spearmanr = None


# ---------------------------------------------------------------------
# Fresh, from-scratch reference implementation (independent of
# robustness/mc_dropout.py's _rankdata / _pearson_correlation)
# ---------------------------------------------------------------------


def average_rank(values: list[float]) -> list[float]:
    """
    1-based average rank for each value, ties sharing the mean of the
    ranks they'd occupy. Implemented via itertools.groupby over sorted
    (value, original_index) pairs -- deliberately a different coding
    approach from mc_dropout._rankdata's manual index-scanning loop, so
    this isn't just a renamed copy of the function under test.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])

    ranks = [0.0] * n
    pos = 0
    for _value, group_iter in groupby(order, key=lambda i: values[i]):
        group = list(group_iter)
        size = len(group)
        # Ranks pos+1 .. pos+size (1-based), averaged.
        avg = sum(range(pos + 1, pos + size + 1)) / size
        for idx in group:
            ranks[idx] = avg
        pos += size

    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    """Textbook Pearson correlation coefficient, written fresh here."""
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    return sxy / (sxx**0.5 * syy**0.5)


def spearman_reference(x: list[float], y: list[float]) -> float:
    """Tie-corrected Spearman rho = Pearson correlation of average ranks
    -- the standard definition (this is also what scipy.stats.spearmanr
    computes, as opposed to the simplified 1 - 6*sum(d^2)/(n^3-n) formula,
    which is only valid when there are NO ties)."""
    return pearson(average_rank(x), average_rank(y))


# ---------------------------------------------------------------------
# Hand-constructed test cases, chosen specifically to exercise tie
# handling (including a 3+ tied block, and ties on both sides).
# ---------------------------------------------------------------------

CASES = [
    {
        "name": "tied block of 3 in x, no ties in y",
        "x": [10, 20, 20, 20, 30],
        "y": [1, 2, 3, 4, 5],
    },
    {
        "name": "tied block of 4 in x (over half the data), distinct y",
        "x": [5, 5, 5, 5, 10, 15, 20],
        "y": [3, 1, 2, 4, 5, 6, 7],
    },
    {
        "name": "ties on BOTH sides simultaneously",
        "x": [1, 2, 2, 3, 3, 3, 4],
        "y": [10, 20, 20, 10, 30, 30, 40],
    },
    {
        "name": "mc_dropout.py's own synthetic ECE-style case "
                "(3 low-variance/correct, 3 high-variance/incorrect -- "
                "each triplet a tied block on the error side)",
        # This mirrors test_mc_dropout.py's
        # test_correlate_variance_with_error_known_perfect_correlation,
        # decomposed to the raw (variance, error) pairs it feeds into
        # spearman_correlation, reported there as ~0.878.
        "x": [0.01, 0.02, 0.03, 0.5, 0.6, 0.7],
        "y": [0, 0, 0, 1, 1, 1],
    },
    {
        "name": "no ties at all (sanity control -- all methods must agree trivially)",
        "x": [3.1, 1.2, 4.5, 2.2, 5.9],
        "y": [10.0, 20.0, 5.0, 30.0, 1.0],
    },
    {
        "name": "everything tied on one side (degenerate: rho undefined)",
        "x": [5, 5, 5, 5],
        "y": [1, 2, 3, 4],
    },
]


def main() -> None:
    print("=== Spearman correlation verification (independent cross-check) ===\n")
    if scipy_spearmanr is None:
        print("NOTE: scipy not available in this environment -- scipy column will show n/a.\n")

    all_match = True

    for case in CASES:
        name = case["name"]
        x, y = case["x"], case["y"]

        print(f"--- {name} ---")
        print(f"  x = {x}")
        print(f"  y = {y}")

        try:
            ref = spearman_reference(x, y)
        except ZeroDivisionError:
            ref = None  # degenerate: constant x or y, rank variance is 0

        try:
            under_test = mc_dropout_spearman(x, y)
        except Exception as e:
            under_test = None
            print(f"  robustness.mc_dropout.spearman_correlation raised: {type(e).__name__}: {e}")

        scipy_val = None
        if scipy_spearmanr is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = scipy_spearmanr(x, y)
            scipy_val = result.correlation
            if scipy_val != scipy_val:  # NaN check without importing math/numpy just for this
                scipy_val = None  # scipy reports NaN, not an exception, for constant input

        def fmt(v):
            return "n/a" if v is None else f"{v:.10f}"

        print(f"  fresh from-scratch reference : {fmt(ref)}")
        print(f"  scipy.stats.spearmanr        : {fmt(scipy_val)}")
        print(f"  mc_dropout.spearman_correlation: {fmt(under_test)}")

        # Degenerate (constant-input) cases: all three methods should agree
        # that no real correlation is defined (None / NaN), not just that
        # mc_dropout happens to return None.
        if under_test is None and ref is None and scipy_val is None:
            print("  -> MATCH: all three agree this case is degenerate (no defined rho)")
            print()
            continue

        mismatches = []
        if under_test is None or ref is None:
            mismatches.append(
                f"degenerate-handling disagreement (mc_dropout={fmt(under_test)}, "
                f"reference={fmt(ref)})"
            )
        else:
            if abs(under_test - ref) > 1e-9:
                mismatches.append(f"reference (diff={abs(under_test - ref):.2e})")
            if scipy_val is not None and abs(under_test - scipy_val) > 1e-9:
                mismatches.append(f"scipy (diff={abs(under_test - scipy_val):.2e})")

        if mismatches:
            all_match = False
            print(f"  ** MISMATCH ** mc_dropout.spearman_correlation disagrees with: {', '.join(mismatches)}")
            print(f"  reference average ranks: x_ranks={average_rank(x)}, y_ranks={average_rank(y)}")
        else:
            print("  -> MATCH (agrees with both independent references to 1e-9)")
        print()

    print("=== Summary ===")
    if all_match:
        print("robustness.mc_dropout.spearman_correlation agrees with both the fresh "
              "from-scratch reference and scipy.stats.spearmanr, to the decimal place "
              "(1e-9 tolerance), across all non-degenerate cases -- including the "
              "tied-block-of-3+ and both-sides-tied cases. VERIFIED CORRECT.")
    else:
        print("MISMATCH FOUND -- see the case(s) marked '** MISMATCH **' above for the "
              "exact input and the disagreement.")


if __name__ == "__main__":
    main()
