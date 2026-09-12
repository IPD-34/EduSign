**To:** Person D, Person A
**Subject:** Confidence signal spec — finalized, needs your explicit ack before you build against it

Hi both,

The confidence-signal spec is finalized: [`docs/confidence_signal_spec.md`](confidence_signal_spec.md), backed by [`src/models/confidence_signal.py`](../src/models/confidence_signal.py) (`compute_confidence_signal(model, landmarks, encoder_attention_mask)`).

Three things before you start building against it:

1. **Shape/dtype/semantics, verified against the real checkpoint, not just described:** `confidence` (B,T, float32), `predicted_token_id` (B,T, int64), `is_blank` (B,T, bool). Per-timestep, not a pooled per-utterance score — Section 2 of the spec explains why. Verification run + real numbers are in Section 4.

2. **The one thing that matters most before you use this:** at the current checkpoint, high `confidence` very often means "certain it's `<blank>`," not "certain about sign content." Measured on 20 real held-out samples: 99.8% of timesteps predict blank, mean confidence given blank ≈0.95, mean confidence given *not* blank ≈0.52 (only 7 timesteps out of 2,828 total). **Always check `is_blank` before treating `confidence` as a content signal** — see Section 3 of the spec for the full numbers and what this means practically (the content-restricted signal will be sparse right now; that's expected, not a bug).

3. **Checkpoint status:** this is verified against `checkpoints/month2_asllrp/best_val_gated.pt` (step 2000, gate active). This is the Month 2 **best-available** checkpoint per the Master Plan fallback — **it is not converged.** Val loss diverges past step 2000-4000 in both training runs, and held-out WER/boundary-F1 are both poor (0.96 / 0.10). Please don't build anything that assumes this checkpoint's outputs are close to final — treat it, and this signal's current behavior, as early-stage.

**Please reply with an explicit acknowledgment that you've read all three points above** (not just "got it, thanks") — specifically that you understand the blank-dominance caveat and the not-converged/best-available-checkpoint status, since both directly affect how this signal should (and shouldn't) be used right now. Happy to walk through it live if useful.

Thanks,
[your name]
