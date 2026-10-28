# Decision Log

| Decision | Alternatives | Evidence | Date |
|---|---|---|---|
| At eval, cap clips to 256 frames with evenly spaced indices (`eval/subsample.py`) before the Uni-Sign loader. | Keep Uni-Sign's `random.sample`; fix its seed. | 85 of 100 `pose_t100` clips exceed 256 frames (min 102, median 479.5, max 1056); `random.sample` is used at `datasets.py:473`, `:573`, `:641`; the old `pred_openasl.csv` was not reproducible exactly (replaying the original RNG sequence reproduced 3 of 5 real outputs). | 2026-10-14 |
| Shuffled control copied from `run_batch_unisign.py` (seed 42, derangement over clip ids). | The `eval_translation.py` variant (file-based, default seed 0). | The mapping equals the original script's mapping on all 100 ids; that script produced `pred_openasl.csv`. | 2026-10-14 |
| Metrics use sacrebleu corpus BLEU and chrF with default settings, no lowercasing, scores on a 0-100 scale; ROUGE-L from the `rouge` package (F-measure x100). | Uni-Sign's `SLRT_metrics.py` (vendored sacrebleu, chrF scaled differently). | Matches `eval_translation.py` calls; our values equal sacrebleu 2.2.0 and 2.6.0 on the same fixed example. | 2026-10-14 |
| Eval dtype fp32. | bf16 (what `run_batch_unisign.py` uses). | First-100, seed 42, real control: bf16 BLEU 6.49, chrF 27.74; fp32 BLEU 6.68, chrF 27.82; no significant difference (paired p = 0.076 BLEU, 0.416 chrF); 59 of 100 texts identical; peak GPU memory about 2.5 GB in both; fp32 wall time 586 s vs 444 s for bf16. T4 has no native bf16. | 2026-10-28 |
| `manifests/eval300.csv` is the eval set. | The old `eval_main_ids.csv`. | The two sets have 0 clips in common (reported by Manasi). | 2026-10-28 |
| Pose files are stored as float32. | float64. | Identical outputs on 5 clips (reported by Manasi). | 2026-10-28 |
