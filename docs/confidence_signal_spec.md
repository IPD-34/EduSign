# Confidence Signal — Spec (Day 53, finalized)

**For:** Person D, Person A
**Implementation:** [`src/models/confidence_signal.py`](../src/models/confidence_signal.py) — `compute_confidence_signal(model, landmarks, encoder_attention_mask)`
**Checkpoint this spec is verified against:** `checkpoints/month2_asllrp/best_val_gated.pt` (step 2000, gate active)

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

This isn't a hypothetical edge case — it's the dominant behavior right now. Measured directly against `best_val_gated.pt` on 20 real held-out samples (2,828 total timesteps):

| | value |
|---|---|
| Timesteps where `predicted_token_id == blank_id` | **2,821 / 2,828 (99.8%)** |
| Samples where **every single timestep** predicted blank | **15 / 20** |
| Mean `confidence` **given** `is_blank == True` | **0.9498** |
| Mean `confidence` **given** `is_blank == False` | **0.5244** (n=7 timesteps, across all 20 samples) |

Read that last row carefully: when the model is confident, it is almost always confident *in blank*, at ~0.95. The rare timesteps where it predicts real content, its own confidence in that content is barely above half. **A naive "high confidence = trust this" filter would, right now, mean 'trust the model's certainty that nothing is happening' essentially everywhere** — the opposite of a useful content-confidence filter.

**Why this is happening, and why it isn't a new problem:** this is the project's already-diagnosed and root-caused CTC blank-dominance (Month 2 Week 6), reconfirmed directly on this exact checkpoint in Day 51/52's held-out evaluation (corpus WER 0.9591, boundary micro-F1 0.0977) and in the convergence analysis (val loss diverges past step 2000/4000 in both training runs; `gate_entropy` frozen near maximum uncertainty across ~12,000 steps). **`checkpoints/month2_asllrp/best_val_gated.pt` (step 2000, gate active) is the Month 2 best-available checkpoint, not a converged model — per the Master Plan fallback, it must be treated and labeled as early-stage everywhere it's used, including here.**

**Practical guidance:**
- Always check `is_blank` alongside `confidence`. If you need a "how sure is the model about actual sign content" signal, that is `confidence` **restricted to timesteps where `is_blank == False`** — not raw `confidence` alone.
- Given only 7 non-blank timesteps showed up across 2,828 checked here, expect this content-restricted signal to be **sparse** at this checkpoint — that sparsity is itself informative (it's consistent with, not contradicted by, everything Day 51/52 already found), not a bug in this function.
- Don't build a downstream feature whose behavior depends on this signal being dense or reliably content-bearing yet. Re-run the verification in Section 4 against whatever checkpoint supersedes `best_val_gated.pt` before trusting it differently.

## 4. Verification (run this session, real output — not asserted)

Ran `compute_confidence_signal` against `best_val_gated.pt`, batch of 3 real padded test-split samples (`B=3, T=205`):

```
confidence:         shape=(3, 205)  dtype=torch.float32   [PASS - matches spec]
predicted_token_id: shape=(3, 205)  dtype=torch.int64     [PASS - matches spec]
is_blank:           shape=(3, 205)  dtype=torch.bool      [PASS - matches spec]
confidence value range: min=0.3127, max=0.9962   (valid probabilities, [0,1])
blank_id used: 2222 (= len(data/vocab_merged.json) = ctc_vocab_size)  [PASS]

Independent recomputation cross-check (softmax+max done separately, outside the function):
  confidence         == recomputed softmax-max         : True  [PASS]
  predicted_token_id == recomputed argmax              : True  [PASS]
  is_blank            == recomputed (argmax == blank_id): True  [PASS]
```

**Result: PASS.** Shape, dtype, and semantics all match this document exactly, confirmed against the actual current model output — not the old decoder path (see Section 5) and not a description from memory.

## 5. Source confirmation — re-verified, not assumed

`compute_confidence_signal` reads **only** `SignTransformer.forward_ctc_boundary_and_gate()`'s CTC output ([`src/models/sign_transformer.py`](../src/models/sign_transformer.py)) — `self.ctc_head`, a plain `nn.Linear(hidden_dim, ctc_vocab_size + 1)` fed by the shared encoder pass (same encoder output the boundary head and gate also read).

It does **not** touch `SignTransformer.forward()` / `self.decoder` — the older single-token autoregressive decoder, kept for WLASL-style isolated-sign decoding. Confirmed directly against the current code this session:
- `self.ctc_head` and `self.decoder` are separate `nn.Module` attributes, constructed independently ([sign_transformer.py:48](../src/models/sign_transformer.py), [:63-65](../src/models/sign_transformer.py)).
- `forward()` requires `target_tokens` as an input (teacher-forced autoregressive decoding) — a fundamentally different call signature from the CTC path, which only needs `landmarks` + `encoder_attention_mask`.
- Every ASLLRP checkpoint this project has produced, including `best_val_gated.pt`, loads only `encoder./ctc_head./boundary_head./bimanual_gate_head.` prefixes (`load_checkpoint_partial`) — **`decoder.*` is always left at fresh random initialization.** The decoder path isn't just architecturally separate, its weights are untrained noise for this task. Using it for a confidence signal would be actively wrong, not merely a different valid choice.

**One gap, stated plainly rather than smoothed over:** this repo has no committed record of the original Month 1 Day 6 discussion to diff this against line-by-line. This section re-derives the source from the current, live code and confirms *that* is correct and self-consistent, rather than assuming continuity with a prior decision this session cannot see. If Month 1's original intent differs from what's described here, that's a conversation to have explicitly — not something this spec can silently confirm or deny.
