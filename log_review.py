"""
log_review.py

Day 37 (Month 2, Week 6): repeatable training-log review tool.

Built per the Week 6 doc's Day 37 goal -- "stand up a repeatable way to
review training health across long, multi-session Kaggle runs" -- and
intended to become standing infrastructure, re-run daily for the rest
of Month 2, not a one-off script.

Parses train_month2_asllrp.py's log format (via train_utils.log_line):
  - Step lines:      "Step N: ctc_loss=X boundary_loss=Y grad_norm=Z steps/min=W"
  - Session markers: "=== Training session start: begin_step=N, resume=B, device=D ==="
                      "=== Session end: step=N, session_steps=S, elapsed=E min, steps/min=R ==="
  - Checkpoint saves: "[CHECKPOINT] Saved (reason): .../step_XXXXXXX.pt"
  - Val evals:        "[VAL] Step N: mean_ctc_loss_per_gloss_token=V (best so far: ...)"
  - Early stop:       "[EARLY STOP] ..."

IMPORTANT -- multiple runs in one log file:
This project's log file is append-only across the whole project history,
not per-run. As of Day 36/37 it contains the pre-patch overfitting run
(steps 0-428,000) immediately followed by the patched, early-stopping
run's false-start test resumes and then the one real continuous run
(steps 300-14,000). A script that silently analyzed "the whole file" or
"the last N lines" would blend two unrelated runs together without
anyone noticing -- exactly the kind of silent-drift failure this
project's discipline exists to catch.

So: --start-step and --end-step are REQUIRED, not defaulted. The caller
states explicitly which run they mean, every time.

Usage:
    python log_review.py path/to/month2_asllrp_training.log \
        --start-step 300 --end-step 14000 \
        --regions 7 \
        [--reference-ratio 9.41]  # optional override; see --reference-ratio's own
                                   # help text below -- this is an APPROXIMATE
                                   # historical reference (~8-9x, plateau-region
                                   # order of magnitude), not an exact gate

Exit behavior: prints a plain-text report to stdout. Nothing here writes
files or mutates checkpoints -- this is a read-only review tool.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field


STEP_RE = re.compile(
    r"^Step (\d+): ctc_loss=([\-\d.]+) boundary_loss=([\-\d.]+) "
    r"grad_norm=([\-\d.]+) steps/min=([\-\d.]+)"
)
SESSION_START_RE = re.compile(
    r"^=== Training session start: begin_step=(\d+), resume=(\w+), device=(\w+) ==="
)
SESSION_END_RE = re.compile(
    r"^=== Session end: step=(\d+), session_steps=(\d+), elapsed=([\-\d.]+) min, "
    r"steps/min=([\-\d.]+) ==="
)
CHECKPOINT_RE = re.compile(
    r"^\[CHECKPOINT\] Saved \(([\w\-]+)\): .*step_(\d+)\.pt"
)
VAL_RE = re.compile(
    r"^\[VAL\] Step (\d+): mean_ctc_loss_per_gloss_token=([\-\d.]+) "
    r"\(best so far: ([\-\d.]+|inf)"
)
EARLY_STOP_RE = re.compile(r"^\[EARLY STOP\]")


@dataclass
class StepRow:
    step: int
    ctc_loss: float
    boundary_loss: float
    grad_norm: float
    steps_per_min: float
    session_idx: int


@dataclass
class SessionInfo:
    idx: int
    begin_step: int
    resume: bool
    end_step: int = None
    session_steps: int = None
    elapsed_min: float = None
    steps_per_min: float = None


def parse_log(path):
    steps = []
    sessions = []
    checkpoints = []  # (step, reason)
    val_evals = []     # (step, val_loss, best_so_far)
    early_stops = []   # step at which EARLY STOP fired (from following step context)

    session_idx = -1
    cur_session = None

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")

            m = SESSION_START_RE.match(line)
            if m:
                session_idx += 1
                cur_session = SessionInfo(
                    idx=session_idx,
                    begin_step=int(m.group(1)),
                    resume=(m.group(2) == "True"),
                )
                sessions.append(cur_session)
                continue

            m = SESSION_END_RE.match(line)
            if m and cur_session is not None:
                cur_session.end_step = int(m.group(1))
                cur_session.session_steps = int(m.group(2))
                cur_session.elapsed_min = float(m.group(3))
                cur_session.steps_per_min = float(m.group(4))
                continue

            m = STEP_RE.match(line)
            if m:
                steps.append(StepRow(
                    step=int(m.group(1)),
                    ctc_loss=float(m.group(2)),
                    boundary_loss=float(m.group(3)),
                    grad_norm=float(m.group(4)),
                    steps_per_min=float(m.group(5)),
                    session_idx=session_idx,
                ))
                continue

            m = CHECKPOINT_RE.match(line)
            if m:
                checkpoints.append((int(m.group(2)), m.group(1)))
                continue

            m = VAL_RE.match(line)
            if m:
                best = m.group(3)
                best_val = float("inf") if best == "inf" else float(best)
                val_evals.append((int(m.group(1)), float(m.group(2)), best_val))
                continue

            if EARLY_STOP_RE.match(line):
                early_stops.append(line)
                continue

    return steps, sessions, checkpoints, val_evals, early_stops


def filter_range(steps, start_step, end_step):
    return [s for s in steps if start_step <= s.step <= end_step]


def region_stats(rows, n_regions):
    """Day 18/19-style region means/peaks: split the selected step range
    into n_regions contiguous chunks (by row count, not by step number,
    so uneven logging cadence doesn't skew region size) and report
    mean/peak grad_norm, ctc_loss, boundary_loss per region."""
    if not rows:
        return []
    n = len(rows)
    chunk = max(1, n // n_regions)
    regions = []
    for i in range(0, n, chunk):
        chunk_rows = rows[i:i + chunk]
        if not chunk_rows:
            continue
        step_lo = chunk_rows[0].step
        step_hi = chunk_rows[-1].step
        grad_norms = [r.grad_norm for r in chunk_rows]
        ctc = [r.ctc_loss for r in chunk_rows]
        bnd = [r.boundary_loss for r in chunk_rows]
        regions.append({
            "step_range": (step_lo, step_hi),
            "n": len(chunk_rows),
            "grad_norm_mean": sum(grad_norms) / len(grad_norms),
            "grad_norm_peak": max(grad_norms),
            "ctc_loss_mean": sum(ctc) / len(ctc),
            "boundary_loss_mean": sum(bnd) / len(bnd),
            "ratio_mean": (sum(ctc) / len(ctc)) / (sum(bnd) / len(bnd)),
        })
    return regions


def checkpoint_cadence(checkpoints, start_step, end_step, expected_every):
    in_range = sorted(
        {step for step, reason in checkpoints if start_step <= step <= end_step}
    )
    gaps = []
    for prev, cur in zip(in_range, in_range[1:]):
        delta = cur - prev
        gaps.append((prev, cur, delta, delta != expected_every))
    return in_range, gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path")
    parser.add_argument("--start-step", type=int, required=True,
                         help="First step (inclusive) of the run to analyze. "
                              "Required -- see module docstring on why there's "
                              "no default.")
    parser.add_argument("--end-step", type=int, required=True,
                         help="Last step (inclusive) of the run to analyze.")
    parser.add_argument("--regions", type=int, default=7,
                         help="Number of contiguous regions to bin the range "
                              "into for grad-norm / loss-ratio trend review "
                              "(Day 18/19 style). Default 7.")
    parser.add_argument("--expected-checkpoint-cadence", type=int, default=2000,
                         help="Expected steps between checkpoints "
                              "(CHECKPOINT_EVERY_N_STEPS in the training script). "
                              "Default 2000.")
    # Default 9.41 = mean per-step ctc/boundary ratio over steps >=50 (post-
    # warmup) from a FRESH, fully-reproducible Day-33-style 150-step rerun,
    # with real per-step data logged to
    # logs/day33_extended_smoke_per_step_rerun.csv. This replaces the
    # original "8.1x" figure (Day 33's local run), which was investigated
    # and confirmed NOT to reproduce under any of 7 tested aggregation
    # formulas -- mean/median/geometric-mean of all 150 steps, ratio-of-
    # sums over all steps, steady-state mean, ratio-of-sums post-warmup,
    # and mean-of-last-20 -- checked against both the original Day 33
    # printed output and a fresh rerun. Root cause: Day 33's script only
    # ever printed to console, so its true per-step data was never
    # persisted and is not recoverable; 8.1x most likely reflects a value
    # eyeballed off that console output rather than a defined statistic.
    # Treat 9.41 (and the 8.1x it replaces) as an APPROXIMATE historical
    # reference for the ~8-9x plateau-region order of magnitude this
    # training setup settles into after the cold-start collapse -- NOT an
    # exact gate or pass/fail threshold.
    DEFAULT_REFERENCE_RATIO = 9.41

    parser.add_argument("--reference-ratio", type=float, default=DEFAULT_REFERENCE_RATIO,
                         help="Approximate historical reference ctc/boundary ratio "
                              "(~8-9x, plateau-region order of magnitude) to "
                              "compare the final region against -- NOT an exact "
                              f"gate. Default {DEFAULT_REFERENCE_RATIO} (a fresh, "
                              "fully-reproducible steady-state figure with real "
                              "per-step CSV data behind it; see the "
                              "DEFAULT_REFERENCE_RATIO comment above for why this "
                              "replaced the original, non-reproducible 8.1x).")
    parser.add_argument("--session", type=int, default=None,
                         help="Restrict analysis to a single session index. "
                              "REQUIRED in practice whenever multiple sessions "
                              "resume from the same begin_step within the "
                              "chosen range (e.g. short test/false-start "
                              "resumes) -- otherwise step numbers repeat "
                              "across sessions and region/trend stats blend "
                              "unrelated runs together. Use --session with no "
                              "other args first to list session indices.")
    args = parser.parse_args()

    steps, sessions, checkpoints, val_evals, early_stops = parse_log(args.log_path)

    if args.session is not None:
        steps = [s for s in steps if s.session_idx == args.session]

    rows = filter_range(steps, args.start_step, args.end_step)

    # Guard: if step numbers are non-monotonic within the selected range
    # (multiple sessions re-logging overlapping step numbers), refuse to
    # silently produce misleading region/trend stats.
    if args.session is None:
        step_nums = [r.step for r in rows]
        if step_nums != sorted(step_nums):
            distinct_sessions = sorted({r.session_idx for r in rows})
            print(f"ERROR: step numbers in range [{args.start_step}, {args.end_step}] "
                  f"are NOT monotonic -- {len(distinct_sessions)} sessions "
                  f"({distinct_sessions}) overlap this range and re-log some of "
                  f"the same step numbers (e.g. short test/false-start resumes "
                  f"before the real run). Re-run with --session <idx> to pick "
                  f"the one real session. Session summary:")
            for s in sessions:
                if s.idx in distinct_sessions:
                    resume_str = "resume" if s.resume else "first-launch"
                    end_str = s.end_step if s.end_step is not None else "?"
                    steps_str = s.session_steps if s.session_steps is not None else "?"
                    print(f"  session {s.idx}: begin_step={s.begin_step} ({resume_str}) "
                          f"-> end_step={end_str}, {steps_str} steps logged")
            sys.exit(1)
    if not rows:
        print(f"No step lines found in range [{args.start_step}, {args.end_step}]. "
              f"Check the range against the log -- nothing to report.")
        sys.exit(1)

    print("=" * 78)
    print(f"LOG REVIEW: {args.log_path}")
    print(f"Range analyzed: step {args.start_step} -> {args.end_step} "
          f"({len(rows)} logged step-lines)")
    print("=" * 78)

    # --- Sessions touching this range ---
    touching = [s for s in sessions if s.end_step is not None
                and not (s.end_step < args.start_step or s.begin_step > args.end_step)]
    print(f"\n[SESSIONS] {len(touching)} session(s) overlap this range:")
    for s in touching:
        resume_str = "resume" if s.resume else "first-launch"
        print(f"  session {s.idx}: begin_step={s.begin_step} ({resume_str}) -> "
              f"end_step={s.end_step}, {s.session_steps} steps in "
              f"{s.elapsed_min:.1f} min ({s.steps_per_min:.1f} steps/min)")
    if not touching:
        print("  (no completed session-end marker found in range -- "
              "run may still be in progress, or range spans a crash)")

    # Flag very short sessions (likely test/false-start resumes, not real
    # training) so they don't get silently averaged in with the real run.
    short_sessions = [s for s in touching if s.session_steps is not None and s.session_steps < 500]
    if short_sessions:
        print(f"\n  NOTE: {len(short_sessions)} session(s) in range have <500 steps "
              f"-- likely short test/false-start resumes rather than real training "
              f"(e.g. --max-steps-this-session or --eval-every-n-steps overrides "
              f"used to validate the patch). Throughput and early step-lines from "
              f"these are included in the region stats below since they were "
              f"logged in-range, but treat their steps/min figures (often low, "
              f"since the first step or two after a resume pays dataloader/model "
              f"warm-up cost) as noise, not signal.")

    # --- Loss curves: overall trend ---
    first, last = rows[0], rows[-1]
    print(f"\n[LOSS CURVES] step {first.step} -> step {last.step}")
    print(f"  ctc_loss:      {first.ctc_loss:.4f} -> {last.ctc_loss:.4f}")
    print(f"  boundary_loss: {first.boundary_loss:.4f} -> {last.boundary_loss:.4f}")
    ratio_first = first.ctc_loss / first.boundary_loss if first.boundary_loss else float("inf")
    ratio_last = last.ctc_loss / last.boundary_loss if last.boundary_loss else float("inf")
    print(f"  ctc/boundary ratio: {ratio_first:.2f}x -> {ratio_last:.2f}x")

    # --- Region breakdown (Day 18/19 style) ---
    regions = region_stats(rows, args.regions)
    print(f"\n[REGIONS] grad-norm / loss-ratio trend, {len(regions)} region(s):")
    print(f"  {'step range':>18} | {'n':>5} | {'grad mean':>9} | {'grad peak':>9} | "
          f"{'ctc mean':>9} | {'bnd mean':>9} | {'ratio':>7}")
    for r in regions:
        lo, hi = r["step_range"]
        print(f"  {f'{lo}-{hi}':>18} | {r['n']:>5} | {r['grad_norm_mean']:>9.3f} | "
              f"{r['grad_norm_peak']:>9.3f} | {r['ctc_loss_mean']:>9.4f} | "
              f"{r['boundary_loss_mean']:>9.4f} | {r['ratio_mean']:>6.2f}x")

    if len(regions) >= 2:
        first_ratio, last_ratio = regions[0]["ratio_mean"], regions[-1]["ratio_mean"]
        direction = "narrowing" if last_ratio < first_ratio else (
            "widening" if last_ratio > first_ratio else "flat")
        print(f"\n  ctc/boundary ratio across regions: {first_ratio:.2f}x -> "
              f"{last_ratio:.2f}x ({direction})")
        boundary_near_ln2 = abs(regions[-1]["boundary_loss_mean"] - 0.693) < 0.15
        print(f"  boundary_loss in final region: {regions[-1]['boundary_loss_mean']:.4f} "
              f"({'near ln(2)=0.693 baseline' if boundary_near_ln2 else 'has moved off the ln(2)=0.693 baseline'})")

    if args.reference_ratio is not None and regions:
        final_ratio = regions[-1]["ratio_mean"]
        cmp = "above" if final_ratio > args.reference_ratio else (
            "below" if final_ratio < args.reference_ratio else "equal to")
        print(f"\n  vs. approximate reference ratio ~{args.reference_ratio:.2f}x "
              f"(historical, ~8-9x order of magnitude -- NOT an exact gate): "
              f"final region is {final_ratio:.2f}x ({cmp} reference)")

    # --- Throughput ---
    print(f"\n[THROUGHPUT]")
    real_sessions = [s for s in touching if s.session_steps and s.session_steps >= 500]
    if real_sessions:
        for s in real_sessions:
            print(f"  session {s.idx}: {s.steps_per_min:.1f} steps/min "
                  f"({s.session_steps} steps / {s.elapsed_min:.1f} min)")
    else:
        print("  No session in range has >=500 steps logged as a completed "
              "session -- can't report a stable steps/min figure yet.")

    # --- Checkpoint cadence ---
    ckpts, gaps = checkpoint_cadence(
        checkpoints, args.start_step, args.end_step, args.expected_checkpoint_cadence
    )
    print(f"\n[CHECKPOINT CADENCE] expected every {args.expected_checkpoint_cadence} steps")
    print(f"  {len(ckpts)} checkpoint(s) in range: {ckpts}")
    anomalies = [g for g in gaps if g[3]]
    if anomalies:
        print(f"  {len(anomalies)} gap(s) deviate from the expected cadence:")
        for prev, cur, delta, _ in anomalies:
            print(f"    {prev} -> {cur}  (delta={delta}, expected {args.expected_checkpoint_cadence})")
    elif gaps:
        print(f"  All {len(gaps)} gap(s) match the expected cadence exactly. No drift.")
    else:
        print("  Fewer than 2 checkpoints in range -- nothing to compare cadence against.")

    # --- Val evals + early stop, for context ---
    val_in_range = [v for v in val_evals if args.start_step <= v[0] <= args.end_step]
    if val_in_range:
        print(f"\n[VAL EVALS] {len(val_in_range)} held-out eval(s) in range:")
        for step, val_loss, best in val_in_range:
            best_str = "inf" if best == float("inf") else f"{best:.4f}"
            print(f"  step {step}: val_loss={val_loss:.4f} (best so far: {best_str})")
    if early_stops:
        print(f"\n[EARLY STOP EVENTS] {len(early_stops)} fired in the full file "
              f"(not range-filtered):")
        for line in early_stops:
            print(f"  {line}")

    print("\n" + "=" * 78)
    print("End of report.")
    print("=" * 78)


if __name__ == "__main__":
    main()