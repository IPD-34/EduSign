# RTMW extraction speed

- Date: 2026-10-15
- Model: rtmlib Wholebody `lightweight` (YOLOX-tiny + RTMW-dw-l-m 256x192), onnxruntime
- Device: `cuda` (NVIDIA GeForce RTX 3050 6GB Laptop GPU)
- Clips timed: 20 (seed 42, sampled from `eval300.csv`), 1 extra warm-up clip excluded
- Timed work per clip: video decode + detection + pose for every frame; model load excluded
- onnxruntime providers in use (before / after the timed run): `{'detector': ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'pose': ['CUDAExecutionProvider', 'CPUExecutionProvider']}` / `{'detector': ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'pose': ['CUDAExecutionProvider', 'CPUExecutionProvider']}`
- GPU utilisation from nvidia-smi during the timed run (461 samples at 1/s): mean 41%, max 64%, memory 443-443 MiB (idle before run: 17%)

| Metric | Value |
|---|---|
| Mean seconds per clip | 25.90 |
| Median seconds per clip | 24.71 |
| Min / max seconds per clip | 6.18 / 72.57 |
| Mean seconds per frame | 0.0583 |
| Total frames | 8892 |
| Total seconds | 518.0 |

## Per clip

| clip_id | frames | seconds |
|---|---|---|
| 07271540624354089-14579_29_14 | 745 | 28.31 |
| 04292615380240461-27609_6_1 | 377 | 13.99 |
| 25066732372640965-12438552_3_3 | 737 | 26.94 |
| 17823842107610255-5180_7_15 | 336 | 12.12 |
| 7322011977986957-46989_6_6 | 487 | 17.76 |
| 06440459788487662-12438552_0_4 | 692 | 25.67 |
| 3417711780830395-19004_5_11 | 589 | 34.05 |
| 05880382618831348-6613_10_15 | 274 | 18.26 |
| 3398063531739528-736_5_6 | 393 | 25.81 |
| 7441102052287127-698_16_14 | 245 | 16.25 |
| 3504359507547228-85232_1_28 | 623 | 41.69 |
| 1596308500897099-184729_2_2 | 299 | 20.88 |
| 8685614912872266-357353_15_5 | 416 | 31.42 |
| 46020335317111627-15944_5_5 | 332 | 30.20 |
| 36742401858522156-5165_3_12 | 285 | 23.74 |
| 3125233426242333-6613_15_12 | 374 | 21.90 |
| 1368310638213508-12438552_3_3 | 877 | 72.57 |
| 3991387397913413-46989_6_7 | 569 | 40.96 |
| 14323415480403745-19004_12_0 | 143 | 9.28 |
| 9318309466509407-184729_25_0 | 99 | 6.18 |

## Kaggle T4

Source: figures from Kaggle runs, entered here by hand; not produced by this laptop's `data/time_rtmw.py` run above.
Model: rtmlib Wholebody `lightweight` (YOLOX-tiny + RTMW-dw-l-m 256x192), onnxruntime.

### Single GPU

- Device: `cuda` (Tesla T4); both the detector and the pose model sessions on `CUDAExecutionProvider`
- Clips timed: 20 from `eval300.csv` (seed 42), 1 extra warm-up clip excluded
- GPU utilisation during the timed run: mean 77% (idle: 24%)

| Metric | Value |
|---|---|
| Mean seconds per clip | 14.68 |
| Median seconds per clip | 12.67 |
| Min / max seconds per clip | 3.29 / 28.95 |
| Mean seconds per frame | 0.0330 |
| Total frames | 8892 |
| Total seconds | 293.6 |

### Two GPUs (`--shard 0/2` and `--shard 1/2`)

- One process per GPU, `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`
- The same 16 clips, 8 per shard

| Metric | Shard 0 | Shard 1 |
|---|---|---|
| Mean seconds per clip | 15.83 | 13.88 |
| Mean seconds per frame (per process) | about 0.0338 | about 0.0338 |
| GPU utilisation | 75% | 75% |

| Run | Wall-clock (16 clips) |
|---|---|
| Single GPU | 232.4 s |
| Two GPUs | 126.6 s |
| Speed-up | about 1.84x |

The 2-GPU test used only 8 clips per shard, so treat 1.84x as approximate.

### Subset cap

This exceeds the 5 s/clip threshold, so the subset cap is 3k clips.

## Kaggle T4, capped at 256 frames

Source: figures from a Kaggle run, entered here by hand; not produced by this laptop's `data/time_rtmw.py` run above.

- Device: `cuda` (Tesla T4); both the detector and the pose model sessions on `CUDAExecutionProvider`, before and after the timed run
- Model: rtmlib Wholebody `lightweight` (YOLOX-tiny + RTMW-dw-l-m 256x192), onnxruntime
- Clips timed: 20 from `eval300.csv` (seed 42), 1 extra warm-up clip excluded
- Frame cap: at most 256 frames per clip (evenly spaced, first and last included); frame counts below are kept frames
- GPU utilisation during the timed run: mean 73% (idle: 26%)

| Metric | Value |
|---|---|
| Mean seconds per clip | 8.39 |
| Median seconds per clip | 8.81 |
| Min / max seconds per clip | 3.37 / 9.65 |
| Mean seconds per frame | 0.0347 |
| Total frames (kept) | 4839 |
| Total seconds | 167.9 |

Not comparable with the uncapped 14.68 s/clip figure above, which processed full clips.

The two-GPU speed-up (1.84x) was measured before the cap was added and should be re-measured.
