# ECE threshold policy for the calibration MLP decision

Status: decision finalized and wired into code. `robustness/confidence_baseline.py`'s `evaluate_calibration_decision()` applies this policy automatically and the CLI report prints the resulting decision alongside the raw ECE number — this document records the reasoning behind that function, not a separate manual step.

## 1. The threshold

**ECE ≥ 0.10 (10 percentage points) → build the calibration MLP.**
**ECE < 0.10 (with sufficient evidence, see §3) → use raw entropy / top-token confidence as-is, no MLP.**

Measured with `robustness/confidence_baseline.py`'s existing `compute_ece()` (10 equal-width bins, utterance-level exact-match correctness), on whatever the current held-out val set is at decision time.

## 2. Research grounding

No CTC-decoder-specific numeric threshold exists as an established convention, so this is grounded in the closest available reference points, with real numbers rather than a guess:

**General deep classifiers (Guo et al., 2017, "On Calibration of Modern Neural Networks," ICML — [arxiv.org/abs/1706.04599](https://arxiv.org/abs/1706.04599)).** Table 1 reports, before any calibration: ResNet-110 on CIFAR-10, ECE 4.6%; CIFAR-100, ECE **16.53%**; Wide ResNet on CIFAR-100, 15.0%; DenseNet-40 on CIFAR-100, 10.37%. After temperature scaling, these all drop to roughly **0.3%–2.3%**. So "uncalibrated-but-normally-trained" modern classifiers commonly sit anywhere from ~3% (easy task) to ~16.5% (harder task), and calibration reliably pulls that down under ~2.5%.

**Sequence/structured-prediction models — the closer analog to a CTC recognizer** (Kumar & Sarawagi, 2019, "Calibration of Encoder-Decoder Models for Neural Machine Translation" — [arxiv.org/abs/1903.00802](https://arxiv.org/abs/1903.00802)). Their Table 2 reports baseline (token-level) ECE across six NMT systems ranging from **2.9% to 9.8%** (En-Vi: 9.8%, En-De GNMT: 4.8–5.0%, De-En: 3.3–3.5%, Transformer En-De: 2.9%), dropping to roughly **1.7%–3.8%** after their recalibration method or temperature scaling. The paper explicitly calls every one of these baselines, including the 2.9% low end, "miscalibrated to various degrees" — a stricter standard than image classification typically applies, since even a seemingly modest ECE was judged worth correcting in a sequence-generation setting.

**CTC-specific work exists but didn't yield a citable numeric threshold.** Confirmed via search that CTC confidence calibration is an active, distinct research area — e.g. "Word-level Confidence Estimation for CTC Models" (Interspeech 2023, [isca-archive.org](https://www.isca-archive.org/interspeech_2023/naowarat23b_interspeech.html)) and "Identifying and Calibrating Overconfidence in Noisy Speech Recognition" ([arxiv.org/abs/2509.07195](https://arxiv.org/pdf/2509.07195)), both confirming CTC/ASR models are commonly overconfident and that calibration meaningfully helps — but their specific numeric ECE tables weren't extractable from the available sources. Stated plainly rather than fabricating a number: this project's threshold is grounded in the NMT and classification numbers above, not a CTC-specific published cutoff, because none was found.

**Why 0.10:** it sits above the low end of both literatures' "already miscalibrated, worth correcting" baselines (Guo's CIFAR-10 ~4.6%, Kumar & Sarawagi's low end ~2.9%) and comfortably above where calibrated models land (~0.3%–3.8% across both). Setting the line at 10% rather than closer to those lower baseline numbers is a deliberate, stated leniency: those papers' authors were motivated to demonstrate a calibration *technique* and treated any nonzero miscalibration as worth fixing; this project has a real cost to building and maintaining a calibration MLP (a new component, more surface area, more to keep correct) against a **Month 2 timeline** and a **currently tiny eval set** (21 utterances). A model performing roughly in line with a typical *uncalibrated* modern sequence model (up to ~10% ECE) does not yet justify that cost; a model clearly worse than that range does.

## 3. Minimum-evidence guard — required before the threshold means anything

**ECE is only evaluated against the threshold when `n_samples ≥ 30` AND `n_correct ≥ 5` AND `n_incorrect ≥ 5`. Otherwise: `INSUFFICIENT_EVIDENCE`, not a threshold verdict.**

This guard exists because of a failure mode already found this session: the step-4000 checkpoint's ECE (0.9814, 21 utterances, **0/21 correct**) is not a real calibration measurement — with zero correct predictions, every confidence value falls in one bin against 0% accuracy, and ECE mathematically collapses to just the mean confidence. It would be wrong to read that as "NEEDS_CALIBRATION" (the number looks bad, but it's not evidence about calibration quality — it's evidence about the model not decoding anything yet) and equally wrong to read it as "ACCEPTABLE" (nothing was actually verified). Both would be mistaking a degenerate input for a real signal.

- **`min_n = 30`**: the standard introductory-statistics rule-of-thumb minimum sample size for a proportion or mean to be reasonably estimable under a normal approximation. Not an ECE-specific derivation — none is a standard convention in the literature reviewed above — but a defensible, commonly-cited floor rather than an arbitrary number, and stated as a heuristic, not a rigorous bound.
- **`min_correct = 5` and `min_incorrect = 5`**: mirrors the "expected cell count ≥ 5" convention from categorical statistics (the same heuristic behind chi-square test validity), applied here to require that ECE's confidence-vs-accuracy comparison actually has *both* outcomes represented with a minimally stable count — directly targets the exact degenerate case already observed (all-correct or all-incorrect), not just low n alone.

Applying this guard to the current real case: n_samples=21 (< 30) **and** n_correct=0 (< 5) — two independent reasons this specific reading is `INSUFFICIENT_EVIDENCE`, not a threshold pass/fail.

## 4. Ambiguous case near the threshold: decision rule, not a delay

Define the ambiguous band as **ECE ∈ [0.08, 0.12)** — a ±0.02 margin sized to roughly track sampling noise from a small held-out set.

**Rule: within the ambiguous band (and once the §3 evidence guard is satisfied), build the calibration MLP.** Default to building, not to waiting for a bigger eval set first:
- A calibration layer (temperature scaling / small MLP on raw logits) is cheap to build relative to the cost of shipping a possibly-miscalibrated confidence signal into anything downstream that trusts it.
- These methods are monotonic transforms — they can't make an already-well-calibrated baseline meaningfully worse, so building "just in case" carries low downside.
- The goal of this policy is that a checkpoint's ECE reading triggers an immediate action, not a second discussion. A band that resolves to "wait and re-measure" would recreate the fork-in-the-road delay this document exists to remove.

Net effect: **ECE ≥ 0.08 (with sufficient evidence) → build the MLP. ECE < 0.08 → don't.**

## 5. What would make this threshold get revised

This is a judgment call, not a measured number, and is stated as revisable:

- **If the first non-degenerate checkpoint's ECE lands close to 0.10 either side** (roughly within the [0.08, 0.12) ambiguous band, or even just outside it but close), treat that as a signal to re-examine whether 0.10 was well-placed for this specific CTC/ASLLRP setting — not as confirmation the line was right just because a decision was reached.
- **If a converged checkpoint's ECE lands clearly on one side** (e.g. well under 5% or well over 20%), that's reasonably strong evidence the threshold's rough positioning is fine and doesn't need revisiting yet.
- **If the eval set grows well past 30 samples** (Month 2's plan calls for a larger held-out set eventually), revisit whether `min_n=30` is still the right floor — a larger available pool might justify tightening it, since the heuristic was chosen for a small-set constraint that may not hold forever.
- Any revision should update this document and say so plainly, not silently drift the constant in code without a record of why.

## 6. Current status (as of this checkpoint)

`checkpoints/month2_asllrp/best_val.pt` (step 4000): ECE = 0.9814, n_samples = 21, n_correct = 0. Per §3, this is **`INSUFFICIENT_EVIDENCE`**, not a pass or fail of the 0.10 threshold — the checkpoint's total blank-collapse (0/21 exact-match decodes) makes this reading uninformative about calibration quality, not proof either that calibration is or isn't needed. This is exactly the state `evaluate_calibration_decision()` reports today; it will automatically resolve to a real `NEEDS_CALIBRATION`/`ACCEPTABLE` verdict the next time this runs against a checkpoint that clears the evidence guard.
