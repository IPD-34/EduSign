"""
rollup_month2_throughput.py — parses logs/month2_asllrp_training.log
into a per-session table: elapsed time, steps/min, GPU-hours (local
sessions counted as GPU-hours too, since it's the same RTX 3050 the
whole time; Kaggle sessions are separately identifiable by their own
step ranges/timestamps if you want to split them out).
"""
import re
from pathlib import Path

LOG_PATH = Path("logs/month2_asllrp_training.log")

session_re = re.compile(
    r"=== Session end: step=(\d+), session_steps=(\d+), "
    r"elapsed=([\d.]+) min, steps/min=([\d.]+) ==="
)

sessions = []
for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
    m = session_re.search(line)
    if m:
        step, session_steps, elapsed, steps_per_min = m.groups()
        sessions.append({
            "end_step": int(step),
            "session_steps": int(session_steps),
            "elapsed_min": float(elapsed),
            "steps_per_min": float(steps_per_min),
        })

total_elapsed_min = sum(s["elapsed_min"] for s in sessions)
total_steps = sum(s["session_steps"] for s in sessions)

print(f"{'end_step':>10} {'session_steps':>14} {'elapsed_min':>12} {'steps/min':>10}")
for s in sessions:
    print(f"{s['end_step']:>10} {s['session_steps']:>14} {s['elapsed_min']:>12.1f} {s['steps_per_min']:>10.1f}")

print(f"\nTotal sessions: {len(sessions)}")
print(f"Total elapsed: {total_elapsed_min:.1f} min = {total_elapsed_min/60:.2f} GPU-hours")
print(f"Total steps this rollup covers: {total_steps}")
print(f"Avg steps/min across sessions: {sum(s['steps_per_min'] for s in sessions)/len(sessions):.1f}")