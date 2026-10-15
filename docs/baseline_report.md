# Baseline report (zero-shot, OpenASL pose-only checkpoint)

## 1. Status

First-100 baseline from pose_t100. Provisional: not a random sample. Random-300 baseline pending P1's manifest and extraction.

## 2. Scores (100 clips, 95% bootstrap CI, 1000 resamples)

| control | dtype | BLEU | chrF | ROUGE-L |
|---|---|---|---|---|
| real | bf16 | 6.49 [4.77, 8.28] | 27.74 [25.84, 29.79] | 21.64 [18.96, 24.80] |
| real | fp32 | 6.68 [4.92, 8.48] | 27.82 [25.99, 29.86] | 21.48 [18.91, 24.51] |
| zeroed | bf16 | 0.00 [0.00, 0.00] | 9.58 [9.18, 9.94] | 0.00 [0.00, 0.00] |
| zeroed | fp32 | 0.00 [0.00, 0.00] | 9.58 [9.18, 9.94] | 0.00 [0.00, 0.00] |
| shuffled | bf16 | 0.21 [0.17, 0.24] | 15.74 [14.73, 16.72] | 8.57 [7.50, 9.61] |
| shuffled | fp32 | 0.22 [0.19, 0.26] | 15.86 [14.84, 16.85] | 8.70 [7.56, 9.83] |

## 3. Real vs shuffled

| dtype | chrF gap (real − shuffled) | 95% CI of gap | BLEU ratio (real / shuffled) | paired p |
|---|---|---|---|---|
| bf16 | +12.00 | [9.82, 14.27] | 30.6x | 0.0000 |
| fp32 | +11.96 | [9.86, 14.25] | 29.8x | 0.0000 |

The BLEU ratio is unstable because shuffled BLEU is near zero.

## 4. bf16 vs fp32 (real control)

| metric | bf16 | fp32 | paired p (fp32 better) |
|---|---|---|---|
| BLEU | 6.49 | 6.68 | 0.076 |
| chrF | 27.74 | 27.82 | 0.416 |

59 of 100 texts are identical between the two dtypes. There is no significant difference. fp32 is proposed as the eval dtype, pending team agreement (see `docs/decision_log.md`).

## 5. Setup (from the metrics JSONs)

| field | value |
|---|---|
| clips | 100 |
| seed | 42 |
| max_frames | 256 |
| num_beams | 4 |
| max_new_tokens | 100 |
| bootstrap resamples | 1000 |
| dtypes | bf16 and fp32 (one run each per control) |
| device | cuda:0 |
| torch | 2.11.0+cu128 |
| transformers | 4.40.0 |
| sacrebleu | 2.6.0 |
| git hash | cc1c05ed666c8bbfd771874fc53680e75fcaf252 (identical in all 6 JSONs) |
| git_dirty | false in all 6 JSONs |

## 6. Latency and memory

| run | wall time (s) | peak GPU memory (MB) |
|---|---|---|
| real bf16 | 444 | 2600 |
| real fp32 | 586 | 2487 |
| zeroed bf16 | 1545 | 2600 |
| zeroed fp32 | 1821 | 2541 |
| shuffled bf16 | 1006 | 2600 |
| shuffled fp32 | 879 | 2487 |

Wall time covers the whole run, including the model build, and peak memory is from the start of the run. Other programs may have been using the GPU, so wall times are not clean latency measurements.

## 7. Caveats

- The old `pred_openasl.csv` is not reproducible exactly.
- Frame subsampling differs from the old run: even spacing is used here, not `random.sample`.
- `unisign_env` uses torch 2.11.0+cu128 and sacrebleu 2.6.0, while the team pins differ.
- 100 clips give wide confidence intervals.
- Per-signer and per-article tables are pending the manifest.
