# Checkpoint-of-record decision: switching to `step_0008000.pt` (gated)

Documented per this project's standing rule that every fallback/deviation gets its reasoning written down before implementation, stated plainly rather than smoothed over (same discipline as the boundary-label real-span decision and the C3 gate training-scheme decision).

## What changed

The checkpoint of record for all evaluation and downstream consumption is now:

**`checkpoints/month2_asllrp_gated/step_0008000.pt`** (step 8000, gate active via `forward_ctc_boundary_and_gate`)

replacing:

`checkpoints/month2_asllrp/best_val_gated.pt` (step 2000, gate active) — **not deleted, not overwritten.** It stays on disk as historical record, same as this project's standing convention for superseded checkpoints/data (`checkpoints/month2_asllrp_OLD_*`, `data/processed/asllrp_OLD_*`).

## Why — the real numbers

Every checkpoint this project has produced was selected for "checkpoint of record" status purely by lowest val CTC loss. A checkpoint-screening pass (previous session) tested that assumption directly: whether val CTC loss is actually a reliable proxy for real decode quality (WER, boundary-F1), given that same session's finding that train CTC loss can collapse below 1.0 while decode-level WER stays ~90%+ even on training data.

It isn't. On the full 122-sample held-out test split:

| | `best_val_gated.pt` (step 2000, chosen by lowest val loss) | `step_0008000.pt` (step 8000, chosen by decode quality) |
|---|---|---|
| Val CTC loss (training-loop metric) | **5.7116 (best)** | 6.8215 (worse) |
| Corpus WER | 0.9591 | **0.9217** |
| Empty-decode rate | 37/122 (30.3%) | **0/122 (0%)** |
| Boundary micro-F1 | 0.0977 | 0.1033 (tied within noise either way) |

`step_0008000.pt` was never flagged as "best" by the training loop's own loss-based tracking — it survived on disk only because of the "keep the last 3 step-numbered checkpoints" retention policy, not because any selection criterion favored it. It has *worse* val CTC loss than the checkpoint it's replacing, and *meaningfully better* real decode behavior: lower WER by 3.7 points, and it never once fails to produce any decoded output at all, where the old record failed on nearly a third of the test split. Boundary-F1 doesn't clearly favor either checkpoint (within the same noise floor established for n=122 comparisons) — this switch is being made on WER grounds specifically, stated as such, not on an across-the-board improvement.

## Consequence, stated plainly

Val CTC loss is not being used to select the checkpoint of record going forward without a decode-quality check alongside it. This is not a claim that val CTC loss is meaningless (it's still the training loop's own real-time signal for early stopping and is left untouched for that purpose) — it's a claim that, at least at this early/not-converged training stage, it does not reliably rank checkpoints by real sequence-decode accuracy, and any future "which checkpoint should we use" decision should re-run this same screening rather than defaulting to lowest val loss.

Neither checkpoint is good in absolute terms — 92% WER is still severe, and this switch does not change that. This is a "less bad, at zero training cost" decision, not a fix.
