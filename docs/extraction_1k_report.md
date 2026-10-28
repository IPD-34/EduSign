# Pose extraction report: eval300 and train_1k

Source: figures reported from a Kaggle run (branch `data-week2`, two Tesla T4 shards, `--stats`), entered here by hand.
They were not produced on this machine and have not been re-run here. This report contains no sentence text and no clip IDs.

## 1. Pose file check (`data/check_poses.py`)

| Manifest | expected | found | missing | bad | extra |
|---|---|---|---|---|---|
| `eval300` | 300 | 300 | 0 | 0 | not reported |
| `train_1k` | 1000 | 1000 | 0 | 0 | 300 |

The 300 extras for `train_1k` are the `eval300` poses, which sit in the same folder. They are expected and do not fail the check.

## 2. Extraction time

Measured from the modification time of the first `.pkl` file to the last.

| Set | Clips | Time | Seconds per clip |
|---|---|---|---|
| `eval300` | 300 | 1,424 s | 4.75 |
| `train_1k` | 1000 | 4,546 s | 4.55 |
| Total | 1300 | 5,970 s (1.66 h) | 4.59 |

Caveats:
- The first-to-last file times exclude model load and the first clip.
- The original fetch output and the provider and GPU-utilisation output were not kept, so this run has no recorded
  confirmation that both sessions were on `CUDAExecutionProvider`, and no GPU-utilisation figure.
- This is about 1.8x faster than the single-GPU capped figure of 8.39 s/clip in `docs/rtmw_speed.md`
  (8.39 / 4.59 = 1.83), but that figure was measured on different clips, so the ratio is approximate.

## 3. Shard equivalence

5 clips were extracted with `--shard 0/1` and again with `--shard 0/2` plus `--shard 1/2`. The maximum absolute difference
between the two results was 0 for keypoints and 0 for scores. Sharding does not change the poses.

## 4. Statistics files

`clip_stats.csv` and `frame_counts.csv` each have 1,301 lines: a header plus 1,300 rows (300 `eval300` + 1000 `train_1k`),
one row per extracted clip.

## 5. Fetch

Fetching with `support_suffix_range=False` in `data/fetch_clips.py` worked on Kaggle for all 1,300 clips.

## 6. Projection for `train_3k` (estimate)

This is an estimate, not a measurement.

| Step | Estimate | Basis |
|---|---|---|
| Extraction, 3,000 clips | about 3.8 h | 3,000 clips x 4.55 s/clip (the `train_1k` rate) = 13,650 s |
| Fetching, 3,000 clips | about 40 min (about 0.7 h) | A 21-clip fetch measured 65.8 MB in 17 s (3.87 MB/s), about 3.1 MB per clip; 3,000 x 3.1 MB = about 9.4 GB, and 9.4 GB / 3.87 MB/s = about 2,430 s |
| Total | about 4.5 h | About 3.8 h extraction + about 0.7 h fetch |

The earlier fetch estimate of 1.7 h was wrong and is replaced by the figure above.

Extraction time is expected to scale roughly linearly with the clip count, so it will shift if the clip mix differs
(clip length varies a lot, and clips are capped at 256 frames).
