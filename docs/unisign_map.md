# Uni-Sign code map (read-only)

Source: `C:\Users\l\Desktop\personb\Uni-Sign`. Paths below are relative to that folder. Line numbers are from the files as read on 2026-10-14. Nothing in Uni-Sign was modified.

## 1. Model class and wiring

| What | Where | Note |
|---|---|---|
| Model class `Uni_Sign(nn.Module)` | `models.py:72` | Single class holding pose encoders and mT5 |
| Part list `['body','left','right','face_all']` | `models.py:77` | 4 parts, one ST-GCN chain each |
| Spatial + temporal ST-GCN chains per part | `models.py:92-93` | `gcn_modules[mode]` (spatial), `fusion_gcn_modules[mode]` (temporal) |
| `left` shares weights with `right` | `models.py:95-97` | `gcn_modules`, `fusion_gcn_modules`, `proj_linear` are aliased |
| mT5 model and tokenizer | `models.py:139-140` | `MT5ForConditionalGeneration.from_pretrained(mt5_path)`; `mt5_path` is `./pretrained_weight/mt5-base` (`config.py:2`) |
| Pose branch forward (per part: proj -> spatial GCN -> add body feature -> temporal GCN -> mean over joints) | `models.py:206-264` | `left`/`right`/`face_all` add a detached body-joint feature (`models.py:240`, `:252`, `:255`) |
| 4 part features concatenated + `part_para`, then `pose_proj` | `models.py:266-267` | Result is used as `inputs_embeds` |
| Text prefix "Translate sign language video to English: " embedded and prepended | `models.py:269-278` | Uses `mt5_model.encoder.embed_tokens` |
| mT5 called with `inputs_embeds`, `labels` | `models.py:291-295` | The pose tokens are the encoder input; decoder is stock mT5 |
| Loss: CrossEntropy with `label_smoothing=args.label_smoothing`, `ignore_index=-100` | `models.py:300-301` | Default `--label_smoothing` is 0.2 (`utils.py:532`) |
| Labels truncated to `max_length=50` tokens | `models.py:286` | Target side in forward |
| RGB branch (only if `--rgb_support`) | `models.py:109-135`, `models.py:208-216` | Not used in pose-only |

## 2. Pose projection layers

| Layer | Where | Shape |
|---|---|---|
| `proj_linear[mode]` (x, y, conf -> 64) | `models.py:86` (applied inside the part loop, `models.py:220-223`) | `nn.Linear(3, 64)`, one per part; `left` aliases `right` (`models.py:97`) |
| `part_para` | `models.py:99` | Learned bias vector, size `hidden_dim*4` = 1024 |
| `pose_proj` (to mT5 embedding size) | `models.py:100` (applied `models.py:267`) | `nn.Linear(256*4, 768)` |

`args.hidden_dim` default is 256 (`utils.py`, `--hidden_dim`, `utils.py:455`).

## 3. Dataset loader and frame subsampling

### Pose loading and the 69 kept points
- `load_part_kp` is at `datasets.py:18`. Confidence threshold `thr = 0.3` (`datasets.py:19`).
- Points kept per part (inside `load_part_kp`, `datasets.py:23-48`):
  - body: indices `[0] + 3..10` = 9 points (`datasets.py:31-33`)
  - left hand: `91:112` = 21 points; right hand: `112:133` = 21 points (`datasets.py:34-41`)
  - face_all: `range(23, 40)[::2]` (9 points) + `83..90` (8 points) + `53` (1 point) = 18 points (`datasets.py:42-47`)
- Normalisation: `crop_scale` at `datasets.py:80` computes the scale from the body part and applies it to all parts; points with confidence <= thr are zeroed (`datasets.py:71`, `datasets.py:104`).

### Datasets
| Class | Where | Used by |
|---|---|---|
| `Base_Dataset.collate_fn` (pads by repeating last frame, builds `attention_mask`) | `datasets.py:336-403` | all |
| `S2T_Dataset` (`load_pose` at `datasets.py:461`) | `datasets.py:406` | `fine_tuning.py:25,39,52` (train / test / dev) |
| `S2T_Dataset_news` (`load_pose` at `datasets.py:566`) | `datasets.py:502` | only for `CSL_News` |
| `S2T_Dataset_online` (`load_pose` at `datasets.py:634`) | `datasets.py:606` | `run_batch_unisign.py:41,72` (single in-memory pose) |

### Frame subsampling
Every `load_pose` caps at `args.max_length` (default 256, `utils.py:523`). If `duration > max_length`, frames are chosen with `random.sample`, then sorted; otherwise all frames are used.

### EVERY use of `random` in the loader / eval path

| # | File:line | Call | Effect |
|---|---|---|---|
| 1 | `datasets.py:473` | `sorted(random.sample(range(duration), k=self.max_length))` | `S2T_Dataset.load_pose`: random frame subset for clips > 256 frames, **train and eval** |
| 2 | `datasets.py:573` | `sorted(random.sample(range(duration), k=self.max_length))` | `S2T_Dataset_news.load_pose`: same, CSL_News only |
| 3 | `datasets.py:641` | `sorted(random.sample(range(duration), k=self.max_length))` | `S2T_Dataset_online.load_pose`: same, used by `run_batch_unisign.py` |
| 4 | `datasets.py:556` | `random.randint(0, len(self) - 1)` | `S2T_Dataset_news.__getitem__`: replacement sample on load failure (CSL_News only) |
| 5 | `datasets.py:181-182` | `np.random.choice(..., size=left_sample_size, replace=False, p=...)` | `load_support_rgb_dict`: left-hand frame selection, **only if `--rgb_support`** |
| 6 | `datasets.py:206-207` | `np.random.choice(..., size=right_sample_size, replace=False, p=...)` | same, right hand, **only if `--rgb_support`** |
| 7 | `datasets.py:93` | `# ratio = np.random.uniform(...)` | commented out; `ratio = 1` is used (`datasets.py:94`) |
| 8 | `utils.py:289-291` | `sampler_func(..., random_choice=True)` uses `np.random.choice` | defined; **no call site found** (`grep sampler_func` returns only the definition) |
| 9 | `fine_tuning.py:30` | `DistributedSampler(train_data, shuffle=True)` | train-time shuffling only; test/dev use `SequentialSampler` (`fine_tuning.py:43`, `:56`) |

Seeding: `utils.set_seed(args.seed)` (`utils.py:431-437`) seeds torch, torch.cuda, `np.random` and `random`; called at `fine_tuning.py:23` and `run_batch_unisign.py:64`. Default `--seed` is 42 (`utils.py:486`). I did not verify whether DataLoader worker processes (`--num_workers` default 8, `utils.py:488`; used at `fine_tuning.py:46`) reproduce the same `random.sample` result across runs.

Outside Uni-Sign's own files: `run_batch_unisign.py:88` uses `random.Random(args.seed)` to build the shuffled-pose control (`run_batch_unisign.py:88-93`); `make_zeroed` is at `run_batch_unisign.py:44`.

## 4. Generation call at eval

| Where | Settings |
|---|---|
| `models.py:313-322` `Uni_Sign.generate` | Wraps `mt5_model.generate(inputs_embeds, attention_mask, max_new_tokens, num_beams)`; no other generate kwargs |
| `fine_tuning.py:263-266` (`evaluate`) | `max_new_tokens=100`, `num_beams=4` (hard-coded) |
| `run_batch_unisign.py:59` | `max_new_tokens=100`, `num_beams=4` (hard-coded) |

Decoding of predictions: `tokenizer.batch_decode(..., skip_special_tokens=True)` at `fine_tuning.py:280`. First prediction is padded to 150 tokens at `fine_tuning.py:276-279` before `pad_sequence`.

Metrics: `translation_performance(tgt_refs, tgt_pres)` at `fine_tuning.py:288`, defined in `SLRT_metrics.py` (uses `external_metrics.sacrebleu`, `SLRT_metrics.py:6,22,37`). BLEURT only if `--eval` and dataset is How2Sign or OpenASL (`fine_tuning.py:292-299`, needs `./BLEURT-20`).

## 5. Checkpoint loading

| Where | What |
|---|---|
| `fine_tuning.py:75-84` | `torch.load(args.finetune, map_location='cpu')['model']` then `model.load_state_dict(state_dict, strict=True)`; prints missing/unexpected keys |
| `run_batch_unisign.py:82-84` | Same pattern, `strict=True` |
| CLI flag | `--finetune` (`utils.py:458`; default `''`) |
| Order | Model is built (`fine_tuning.py:67`), moved to cuda, trainable params cast to fp32 (`fine_tuning.py:71-73`), then the checkpoint is loaded, then DeepSpeed init (`fine_tuning.py:101`) |
| Checkpoint saved format | `{'model': ...}` via `utils.save_on_master` (`fine_tuning.py:133` (per epoch), best at `:148`, `:161`, `:174`); contents of the dict beyond `'model'` not read |

Local checkpoints seen: `personb\ckpt\openasl_pose_only_slt.pth`, `personb\ckpt\how2sign_pose_only_slt.pth`.

## 6. Entry points

| Entry | Where | Notes |
|---|---|---|
| Train + eval | `fine_tuning.py`, `main(args)` at `fine_tuning.py:19`, invoked at `:334` | Argument parser is `utils.get_args_parser` (`utils.py:442`) |
| Eval only | same script with `--eval` | `fine_tuning.py:113-123`; runs dev (not for How2Sign) then test, then returns |
| Train loop | `train_one_epoch` at `fine_tuning.py:194` | Uses DeepSpeed `model.backward` / `model.step` (`fine_tuning.py:219-220`) |
| Eval function | `evaluate` at `fine_tuning.py:235` | Writes `{phase}_tmp_pres.txt` / `_refs.txt` only if `--eval` and world size 1 (`fine_tuning.py:317-323`) |
| Optimizer / scheduler | `create_optimizer` (timm) `fine_tuning.py:93`; cosine scheduler via `transformers.get_scheduler` `fine_tuning.py:94-99` | |
| Precision | `--dtype` default `bf16` (`utils.py:501`); DeepSpeed init `utils.py:404` | `evaluate` casts inputs to bf16 when `model.bfloat16_enabled()` (`fine_tuning.py:241-254`) |
| Custom batch inference (not upstream) | `run_batch_unisign.py`, `main` at `:63` | Loads a folder of pose `.pkl`, runs real / zeroed / shuffled, writes CSV (`--pose_dir`, `--out_csv`, `--max_clips`). Model cast to bf16 at `:85` |

## Not found
- A LoRA/PEFT implementation: not found in the files read (`models.py`, `fine_tuning.py`, `datasets.py`, `utils.py`).
- A deterministic (non-random) frame subsampling path: not found; all three `load_pose` implementations use `random.sample`.
