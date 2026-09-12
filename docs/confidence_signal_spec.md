# Confidence Signal — Spec (Day 53, finalized; updated for the checkpoint-of-record switch)

**For:** Person D, Person A
**Implementation:** [`src/models/confidence_signal.py`](../src/models/confidence_signal.py) — `compute_confidence_signal(model, landmarks, encoder_attention_mask)`
**Checkpoint this spec is verified against:** `checkpoints/month2_asllrp_gated/step_0008000.pt` (step 8000, gate active) — see [`docs/checkpoint_of_record_decision.md`](checkpoint_of_record_decision.md). This **supersedes** `best_val_gated.pt` (step 2000), which this spec was originally verified against on Day 53; that checkpoint is not deleted and stays on disk as historical record. Sections 3 and 4 below show both checkpoints' real numbers side by side — nothing here is silently overwritten.

Same documentation discipline as this project's Month 1 landmark-format spec: exact tensor shape, exact dtype, a precise definition of the semantics, verified by actually running it against the real checkpoint — not described from memory of how the code was designed. Section 4 below is the verification run's real output.

---

## 1. What this is

A per-timestep confidence signal derived from the CTC head's own output distribution. Calling `compute_confidence_signal(model, landmarks, encoder_attention_mask)` returns:

| Field | Shape | Dtype | Definition |
|---|---|---|---|
| `confidence` | `(B, T)` | `float32` | `softmax(ctc_log_probs)[b, t, :].max()` — the model's own top post-softmax probability at timestep `t`, over the **full** `(ctc_vocab_size + 1)`-way distribution (every real gloss id **plus** the CTC blank token). |
| `predicted_token_id` | `(B, T)` | `int64` | `argmax` of that same distribution — the id `confidence` is a probability *for*. May equal `blank_id` (`== ctc_vocab_size`, currently `2222`). |
| `is_blank` | `(B, T)` | `bool` | `predicted_token_id == blank_id`. **Must be checked before interpreting `confidence` as a signal about sign content** — see Section 3. |

`B` = batch size, `T` = number of encoder timesteps (real frames, not gloss tokens — this is a per-encoder-timestep signal, aligned 1:1 with `landmarks`/`encoder_attention_mask`, the same convention this project uses for `boundary_target`).

## 2. Why per-timestep, not an aggregated per-utterance score

Two options were on the table: (a) this per-timestep signal, or (b) a single pooled scalar per utterance (e.g. mean confidence over `T`). **Per-timestep was chosen.** Reason: an aggregated scalar would average together timesteps that are "confidently blank" and timesteps that are "confidently a real gloss" into one number that hides exactly the distinction Section 3 says is critical — at the current checkpoint, that aggregate would mostly just measure "how blank-dominated is this utterance," not anything about content confidence, and there would be no way to recover the distinction afterward. Keeping it per-timestep, with `predicted_token_id`/`is_blank` alongside it, lets you filter or group by blank-vs-content *before* aggregating anything yourselves, which an aggregated score would foreclose.

## 3. THE CAVEAT — read this before using the signal for anything

**At the current checkpoint, `confidence` will very often reflect near-certainty in the `<blank>` token, not real gloss content. Do not treat high confidence as "the model is confident about content here" without first checking `is_blank`.**

This isn't a hypothetical edge case — it's the dominant behavior right now, **re-measured directly against the new checkpoint of record** (`step_0008000.pt`) on the identical protocol used on Day 53 (same 20 real held-out samples, 2,828 total timesteps), reported side by side with the old checkpoint's numbers so nothing here is asserted from memory of what it "should" look like now that empty-decode rate dropped to 0%:

| | `best_val_gated.pt` (step 2000, OLD) | `step_0008000.pt` (step 8000, **current**) |
|---|---|---|
| Timesteps where `predicted_token_id == blank_id` | 2,821 / 2,828 (99.8%) | **2,734 / 2,828 (96.7%)** |
| Samples where **every single timestep** predicted blank | 15 / 20 | **1 / 20** |
| Mean `confidence` **given** `is_blank == True` | 0.9498 | **0.9713** |
| Mean `confidence` **given** `is_blank == False` | 0.5244 (n=7) | **0.4206 (n=94)** |

**Plain read of this, not smoothed over: the caveat is NOT resolved by the checkpoint switch — on the core severity dimension it's essentially unchanged or slightly worse.** Blank-confidence went *up* slightly (0.95→0.97) and content-confidence went *down* slightly (0.52→0.42); when the model is confident, it is now if anything even more likely to be confidently wrong-about-content-being-absent than before. What *did* change: far fewer individual samples are 100%-blank end-to-end (15/20 → 1/20), and the volume of non-blank timesteps to even look at went up 13x (n=7 → n=94) — consistent with yesterday's finding that this checkpoint's empty-decode rate is 0% vs. the old checkpoint's 30%. So: **broader occurrence of some non-blank signal, but no improvement in how much to trust that signal when it appears.** Don't read "0% empty decodes" as "the blank-confidence problem is fixed" — it isn't, by this measurement.

**Why this is happening, and why it isn't a new problem:** this is the project's already-diagnosed and root-caused CTC blank-dominance (Month 2 Week 6), reconfirmed directly on the old checkpoint in Day 51/52's held-out evaluation (corpus WER 0.9591, boundary micro-F1 0.0977) and now on the new one too (corpus WER 0.9217, boundary micro-F1 0.1033 — still severe, just less severe on WER specifically). **`checkpoints/month2_asllrp_gated/step_0008000.pt` is the Month 2 best-available checkpoint, not a converged model — per the Master Plan fallback, it must be treated and labeled as early-stage everywhere it's used, including here.**

**Practical guidance (unchanged by the switch):**
- Always check `is_blank` alongside `confidence`. If you need a "how sure is the model about actual sign content" signal, that is `confidence` **restricted to timesteps where `is_blank == False`** — not raw `confidence` alone.
- The content-restricted signal is less sparse now (94 vs 7 non-blank timesteps in the same 20-sample check) but each individual instance is, if anything, *less* trustworthy (mean confidence 0.42 vs 0.52) — more data points, not better ones.
- Don't build a downstream feature whose behavior depends on this signal being dense or reliably content-bearing yet. Re-run this same measurement against whatever checkpoint supersedes `step_0008000.pt` before trusting it differently.

## 4. Verification (run this session, real output — not asserted)

Re-ran `compute_confidence_signal` against the new checkpoint of record, `step_0008000.pt`, batch of 3 real padded test-split samples (`B=3, T=205` — same setup as the original Day 53 run):

```
confidence:         shape=(3, 205)  dtype=torch.float32   [PASS - matches spec]
predicted_token_id: shape=(3, 205)  dtype=torch.int64     [PASS - matches spec]
is_blank:           shape=(3, 205)  dtype=torch.bool      [PASS - matches spec]

Independent recomputation cross-check (softmax+max done separately, outside the function):
  confidence         == recomputed softmax-max         : True  [PASS]
  predicted_token_id == recomputed argmax              : True  [PASS]
  is_blank            == recomputed (argmax == blank_id): True  [PASS]
```

**Result: PASS.** Shape, dtype, and semantics all match this document exactly, confirmed against the new checkpoint's actual model output — not the old decoder path (see Section 5) and not a description from memory. The original Day 53 verification run (same checks, same PASS result) was against `best_val_gated.pt` — shape/dtype/cross-check behavior is identical across both checkpoints, as expected (the function's correctness doesn't depend on which checkpoint's weights are loaded into `model`); only the caveat's real numbers in Section 3 differ between them.

## 5. Source confirmation — re-verified, not assumed

`compute_confidence_signal` reads **only** `SignTransformer.forward_ctc_boundary_and_gate()`'s CTC output ([`src/models/sign_transformer.py`](../src/models/sign_transformer.py)) — `self.ctc_head`, a plain `nn.Linear(hidden_dim, ctc_vocab_size + 1)` fed by the shared encoder pass (same encoder output the boundary head and gate also read).

It does **not** touch `SignTransformer.forward()` / `self.decoder` — the older single-token autoregressive decoder, kept for WLASL-style isolated-sign decoding. Confirmed directly against the current code this session:
- `self.ctc_head` and `self.decoder` are separate `nn.Module` attributes, constructed independently ([sign_transformer.py:48](../src/models/sign_transformer.py), [:63-65](../src/models/sign_transformer.py)).
- `forward()` requires `target_tokens` as an input (teacher-forced autoregressive decoding) — a fundamentally different call signature from the CTC path, which only needs `landmarks` + `encoder_attention_mask`.
- Every ASLLRP checkpoint this project has produced, including `step_0008000.pt` and `best_val_gated.pt`, loads only `encoder./ctc_head./boundary_head./bimanual_gate_head.` prefixes (`load_checkpoint_partial`) — **`decoder.*` is always left at fresh random initialization.** The decoder path isn't just architecturally separate, its weights are untrained noise for this task. Using it for a confidence signal would be actively wrong, not merely a different valid choice.

**One gap, stated plainly rather than smoothed over:** this repo has no committed record of the original Month 1 Day 6 discussion to diff this against line-by-line. This section re-derives the source from the current, live code and confirms *that* is correct and self-consistent, rather than assuming continuity with a prior decision this session cannot see. If Month 1's original intent differs from what's described here, that's a conversation to have explicitly — not something this spec can silently confirm or deny.
