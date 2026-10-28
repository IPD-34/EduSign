# Edge-case flag report

Sources: `manifests/edge_text_flags.csv` (text flags, all 64,266 clips) and
`manifests/edge_eval300_pose_flags.csv` (pose and video flags, the 300 eval300 clips). The stat distributions are
the ones printed by `data/make_pose_flags.py`. This report holds counts only: no sentence text and no clip IDs.

> **All flags are heuristics, and the thresholds are not validated.** Nothing here has been checked against labels
> or human review. The pose-flag thresholds were first guessed, then partly adjusted after looking at the distributions
> of the same 300 eval300 clips they are applied to, so their rates are in-sample and should not be read as findings about
> the dataset. Use the flags to slice results and to pick examples to look at, not as ground truth.

## 1. Text flags per split

Rates are the share of the clips in that split. The text flags come from the sentence text, which is not reproduced here.

| Split | Clips | `has_digit` | `has_acronym` | `long_sentence` | `short_sentence` |
|---|---|---|---|---|---|
| train | 53,328 | 13,315 (25.0%) | 4,629 (8.7%) | 5,322 (10.0%) | 5,111 (9.6%) |
| val | 4,432 | 1,014 (22.9%) | 473 (10.7%) | 494 (11.1%) | 373 (8.4%) |
| test | 6,506 | 1,809 (27.8%) | 1,196 (18.4%) | 564 (8.7%) | 623 (9.6%) |
| all | 64,266 | 16,138 (25.1%) | 6,298 (9.8%) | 6,380 (9.9%) | 6,107 (9.5%) |

The test split has a noticeably higher acronym rate than train (see the `has_acronym` column), so acronym results on test
are not directly comparable with train-side rates.

Definitions (see `data/make_edge_flags.py`):
- `has_digit`: the sentence contains a digit (regex `\d`, so other Unicode decimal digits count too).
- `has_acronym`: the sentence contains a run of 2 or more capital ASCII letters not touching other letters
  (`DNA`, `DNA's` and `CO2` match; `mRNA` and `iPhone` do not).
- `long_sentence`: word count (whitespace split) strictly above the 90th percentile of the train split, which is 35 words.
- `short_sentence`: word count strictly below the 10th percentile of the train split, which is 7 words.

## 2. Pose and video flags per split

These exist only for the eval300 clips, so only the splits that eval300 contains appear (test).

| Split | Clips | `long_clip` | `short` | `no_person_frames` | `multi_person` | `hand_dropout` | `left_dominant` | `nonstandard_fps` | `nonstandard_resolution` |
|---|---|---|---|---|---|---|---|---|---|
| test | 300 | 30 (10.0%) | 26 (8.7%) | 13 (4.3%) | 41 (13.7%) | 26 (8.7%) | 68 (22.7%) | 100 (33.3%) | 0 (0.0%) |

Thresholds (named constants at the top of `data/make_pose_flags.py`):

| Flag | Rule |
|---|---|
| `long_clip` | `n_frames` > 842 (about the 90th percentile of eval300 clip length) |
| `short` | `n_frames` < 150 |
| `no_person_frames` | `frames_no_person_frac` > 0 (any kept frame with no detected person) |
| `multi_person` | `max_persons_detected` > 1 |
| `hand_dropout` | `hand_low_conf_frac` > 0.2 |
| `left_dominant` | `left_path_len` > 1.5 x `right_path_len` |
| `nonstandard_fps` | the absolute difference from the most common fps in the stats file is above 0.5 |
| `nonstandard_resolution` | `(width, height)` differs from the most common pair; never fires on this data (all clips are 640x480) |

## 3. Stat distributions (eval300, 300 clips)

As printed by `data/make_pose_flags.py`: minimum, median, 90th percentile and maximum of each stat in `clip_stats.csv`.

| Stat | min | median | p90 | max |
|---|---|---|---|---|
| `n_frames` | 46 | 428 | 841.8 | 2667 |
| `fps` | 14.99 | 30 | 31.23 | 31.5 |
| `width` | 640 | 640 | 640 | 640 |
| `height` | 480 | 480 | 480 | 480 |
| `frames_no_person_frac` | 0 | 0 | 0 | 0.007812 |
| `max_persons_detected` | 1 | 1 | 2 | 2 |
| `left_hand_mean_score` | 0.199 | 0.682 | 0.8033 | 0.9401 |
| `right_hand_mean_score` | 0.2793 | 0.7266 | 0.8293 | 0.8994 |
| `hand_low_conf_frac` | 0 | 0.04883 | 0.1832 | 0.8261 |
| `left_path_len` | 0.4828 | 6.572 | 13.62 | 29.26 |
| `right_path_len` | 1.05 | 6.591 | 11.72 | 24.13 |

Most common values in the stats file (1,300 rows: eval300 and train_1k): fps 30 (tolerance 0.5), resolution 640x480.

## 4. Caveats

- **Not validated.** None of the thresholds has been checked against human judgement. The text flags rest on simple
  regular expressions and a whitespace word count, so they will miss or over-match some sentences.
- **`nonstandard_fps` depends on its tolerance.** The recorded fps jitters around 30, so an exact match would flag most
  clips; the 0.5 tolerance is a choice, not a measured cut-off.
- **Small values near the limits.** `no_person_frames` fires on any kept frame without a person, so it also catches clips
  with a single such frame. Stats are computed on the kept frames (at most 256 per clip), so `frames_no_person_frac`
  moves in steps of about 1/256 on long clips.
- **Path lengths** (`left_dominant`) are measured on the kept, possibly subsampled, frames in normalised coordinates
  and include per-frame keypoint jitter, so they are a rough proxy for movement.
- **`long_clip` and `short` describe the original clip length** (`n_frames`), not the 256-frame subsample used for pose.
