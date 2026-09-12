**To:** Person D, Person A
**Subject:** Confidence signal spec — finalized, needs your explicit ack before you build against it

Hi both,

The confidence-signal spec is finalized: [`docs/confidence_signal_spec.md`](confidence_signal_spec.md), backed by [`src/models/confidence_signal.py`](../src/models/confidence_signal.py) (`compute_confidence_signal(model, landmarks, encoder_attention_mask)`).

**Note if you saw an earlier version of this message:** the checkpoint it's verified against changed since — see point 3. If you already started building against the old checkpoint reference, nothing about the function or its shapes/dtypes changed, only which weights are loaded; re-read point 2 either way, since the caveat numbers moved too.

Three things before you start building against it:

1. **Shape/dtype/semantics, verified against the real checkpoint, not just described:** `confidence` (B,T, float32), `predicted_token_id` (B,T, int64), `is_blank` (B,T, bool). Per-timestep, not a pooled per-utterance score — Section 2 of the spec explains why. Verification run + real numbers are in Section 4.

2. **The one thing that matters most before you use this:** at the current checkpoint, high `confidence` very often means "certain it's `<blank>`," not "certain about sign content." Measured on 20 real held-out samples: 96.7% of timesteps predict blank, mean confidence given blank ≈0.97, mean confidence given *not* blank ≈0.42 (94 timesteps out of 2,828 total). **This is not an improvement over the previous checkpoint on the dimension that matters** — blank-confidence went up slightly, content-confidence went down slightly; what improved is only that far fewer individual samples are 100%-blank end to end (1/20 now vs. 15/20 before). **Always check `is_blank` before treating `confidence` as a content signal** — see Section 3 of the spec for the full old-vs-new numbers side by side.

3. **Checkpoint status — changed since the spec was first written:** this is now verified against `checkpoints/month2_asllrp_gated/step_0008000.pt` (step 8000, gate active), **not** `best_val_gated.pt` (step 2000) as originally written. A screening pass found val CTC loss doesn't reliably track real decode quality at this training stage — `step_0008000.pt` decodes meaningfully better (WER 0.92 vs 0.96, and it never fails to produce output at all, vs. failing on 30% of test utterances before). See [`docs/checkpoint_of_record_decision.md`](checkpoint_of_record_decision.md) for the real numbers behind that switch. This is still the Month 2 **best-available** checkpoint per the Master Plan fallback — **it is not converged.** Held-out WER/boundary-F1 are still poor (0.92 / 0.10). Please don't build anything that assumes this checkpoint's outputs are close to final — treat it, and this signal's current behavior, as early-stage.

**Please reply with an explicit acknowledgment that you've read all three points above** (not just "got it, thanks") — specifically that you understand the blank-dominance caveat (numbers changed, severity didn't) and the not-converged/best-available-checkpoint status (which checkpoint changed), since both directly affect how this signal should (and shouldn't) be used right now. Happy to walk through it live if useful.

Thanks,
[your name]
