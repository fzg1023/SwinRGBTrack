# SwinRGBTrack

**RGB-T object tracking: shared hierarchical representations + cross-modal adaptation within relation layers + post-encoder fusion**

[Pretrained weights,models and results](https://pan.baidu.com/s/1luaTWQ1YzmU-EwEmX9uKzw?pwd=SWIN)

[简体中文](./README_CN.md) ｜ **English**

---

## 1. Method Overview

SwinRGBTrack uses a single shared hierarchical visual front end to extract RGB and thermal infrared (TIR) features, performs lightweight cross-modal adaptation while template–search relations are being encoded, fuses the two modalities at the encoder output, and finally produces classification and bounding-box regression through a unified decoder.

| Component | Implementation | Description |
|---|---|---|
| Shared Swin-B first three stages | `models/backbone/swin_transformer.py` | Stride 16, 512 channels; template grid 12×12, search grid 24×24; only the third-stage output enters the relation encoder |
| 8-layer two-stream relation encoder | `models/methods/SwinTrack/modules/encoder/` | Template and search tokens are concatenated within each modality; the two streams share their core parameters |
| Bidirectional bottleneck prompts | `models/methods/SwinRGBTrack/network_enc_promptlora.py` | 512→16→512 bottleneck; injects the other modality state into each layer input as a zero-initialized residual |
| TIR-specific QKV LoRA | `models/methods/SwinTrack/modules/lora.py` | Added only to the attention QKV projections of the TIR stream, rank 8 with scale α=16; the RGB stream keeps the shared core mapping |
| Post-encoder fusion | `network_enc_fuse.py`, `network_enc_fusemamba.py` | Equal-weight 0.5 fusion + shared channel bias, plus a forward/backward Mamba spatial-context residual branch |
| Single relation decoder layer + Mlp head | `models/methods/SwinTrack/modules/decoder/`, `models/head/mlp.py` | Search tokens as queries, concatenated template–search sequence as keys/values |
| Training losses | `criterion/modules/varifocal_loss.py`, `criterion/modules/iou_loss.py` | Varifocal classification loss + GIoU regression loss, weight 2 each |

The adaptation modules add about **0.402 M** parameters in total (≈0.271 M for prompts, ≈0.131 M for LoRA).

### Main results

| Dataset | Metrics |
|---|---|
| LasHeR (245 common test sequences) | PR **72.68** / NPR **68.70** / SR **57.67** |
| GTOT (full 50 sequences) | MPR **91.72** / MSR **78.68** |
| RGBT210 | PR **84.19** / SR **62.02** |
| RGBT234 (full set) | MPR **85.67** / MSR **64.13** |
| VTUAV | ST **84.86 / 77.00**; LT **63.24 / 55.61** |

---

## 2. Environment Setup

All experiments use the conda environment **`mambavision_rgbt`** (the activation path is hard-coded in the scripts).

```bash
# Option 1: use requirements.txt
conda create -y -n mambavision_rgbt python=3.9
conda activate mambavision_rgbt
pip install -r requirements.txt
# required by the Mamba fusion residual
pip install mamba_ssm

# Option 2: legacy helper (note it creates an environment named SwinTrack;
# experiments should use mambavision_rgbt instead)
bash conda_init.sh
```

> ⚠️ Every `.sh` script contains `source /home/fzg/anaconda3/etc/profile.d/conda.sh`.
> When moving to another machine, edit that line to match your local conda installation.

Main dependencies are listed in [`requirements.txt`](./requirements.txt): `torch`, `torchvision`, `timm`, `fvcore`, `wandb`, `shapely`, `numpy`, `scipy`, `matplotlib`, `pyyaml`, etc.

### Pretrained weights

| File | Purpose | Location |
|---|---|---|
| `SwinTrack-B-384.pth` | Backbone / training warm-start weights | Project root |

---

## 3. Data Preparation

### 3.1 LasHeR (primary local dataset)

The root directory is controlled by `LASHER_ROOT`, default `/home/fzg/data/lasher`:

```
/home/fzg/data/lasher
├── train/                 # 979 training sequences (used by train_lasher_ft_from_vtuav.sh)
│   └── <seq>/visible/  infrared/  ...
└── testingset/            # 245 test sequences
    └── <seq>/visible/  infrared/  ...
```

### 3.2 Server-side multi-dataset (VTUAV / GTOT / RGBT210 / RGBT234)

Controlled by `SGTEST_DATA_ROOT`, default `/root/RGBTData`:

```
/root/RGBTData
├── VTUAV/test_ST  VTUAV/test_LT
├── GTOT    (v/ i/ groundTruth_v.txt)
├── RGBT210 (visible/ infrared/ visible.txt)
└── RGBT234 (visible/ infrared/ visible.txt)
```

### 3.3 Official evaluation toolkit (RGBT toolkit 1.0.1)

The official metrics for GTOT / RGBT210 / RGBT234 / LasHeR are aggregated with the official
RGBT toolkit 1.0.1, whose root directory is given by `RGBT_TOOLKIT_HOME` (default `/root`).

### 3.4 Path template

`path.template.yaml` is the path template used by the config-driven framework entry (`main.py`).
Copy it to `path.yaml` and fill in the paths:

```yaml
LASHER_PATH: '/home/fzg/data/lasher'
```

---

## 4. Script Overview

The repository contains **33 `.sh` scripts**, grouped into four categories:

| Group | Scripts | Description |
|---|---|---|
| **A. Final method EncPromptLoRA** | `train_rgbt_promptlora*.sh`, `train_lasher_ft_from_vtuav.sh`, `train_vtuav_promptlora.sh`, `test_lasher_rgbt_promptlora*.sh`, `test_vtuav_*.sh`, `test_encpromptlora_*.sh` | Training and testing for the official results |
| **B. Comparison / ablation** | `train_rgbt.sh`, `train_rgbt_enc_fuse.sh`, `train_rgbt_concat_fuse.sh`, `train_rgbt_dec_fuse.sh`, `train_rgbt_dec_fuse_adaptive.sh`, `train_rgbt_layer_fuse.sh` + matching `test_*` | Comparison experiments over fusion positions and structural variants |
| **C. Evaluation tools** | `recompute_official_rgbt_metrics.sh`, `test_lasher_rgb.sh`, `test_lasher_tir.sh` | Official metric recomputation, single-modality comparison |
| **D. Framework entry points** | `conda_init.sh`, `run.sh` | Environment creation and the native config-driven entry |

---

## 5. Training Scripts

### 5.1 Final method — EncPromptLoRA (★ core)

**Training chain (FP32 + AdamW + frozen backbone, only Prompt/LoRA/head are trained):**

```
SwinTrack-B-384.pth
  └─(enc_fuse fine-tune)→ enc_fusemamba_fp32 ep2
       └─ stage1 (FP32, LR=1e-4, 3 epochs)  → output/enc_promptlora_fp32/
            └─ stage2 (FP32, LR=5e-5, 2 epochs) → output/enc_promptlora_fp32_stage2/
                 └─ stage3 (FP32, LR=2.5e-5, 2 epochs) → output/enc_promptlora_fp32_stage3/
```

| Script | Warm-start weights | Training / freezing | Output directory |
|---|---|---|---|
| `train_rgbt_promptlora.sh` | `output/enc_fusemamba/checkpoint_epoch001.pth` | **AMP**, 3 epochs, constant LR=1e-4; backbone + original encoder + decoder + mamba residual frozen | `output/enc_promptlora/` |
| `train_rgbt_promptlora_fp32_stage1.sh` | `output/enc_fusemamba_fp32/checkpoint_epoch002.pth` | **FP32 (`--no_amp`)**, 3 epochs, LR=1e-4 | `output/enc_promptlora_fp32/` |
| `train_rgbt_promptlora_fp32_stage2.sh` | `output/enc_promptlora_fp32/checkpoint_epoch003.pth` | FP32, 2 epochs, LR=5e-5 (fully frozen again) | `output/enc_promptlora_fp32_stage2/` |
| `train_rgbt_promptlora_fp32_stage3.sh` | `output/enc_promptlora_fp32_stage2/checkpoint_epoch001.pth` | FP32, 2 epochs, LR=2.5e-5 | `output/enc_promptlora_fp32_stage3/` |

**Usage** (the scripts take no arguments; key hyperparameters are constants at the top of each script):

```bash
bash train_rgbt_promptlora_fp32_stage1.sh
bash train_rgbt_promptlora_fp32_stage2.sh   # requires stage1 to be finished first
bash train_rgbt_promptlora_fp32_stage3.sh   # requires stage2 to be finished first
```

Shared training hyperparameters (hard-coded in the scripts): `--batch_size 16 --samples_per_epoch 60000 --lr ... --backbone_lr 1e-5
--freeze_backbone_epochs 100 (forever) --freeze_stem_epochs 100 --freeze_fusion_epochs 100 --const_lr --seed 42`.
They all call `train_rgbt_enc_fuse_v1.py --model_type enc_promptlora` underneath.

**VTUAV branch:**

| Script | Description |
|---|---|
| `train_vtuav_promptlora.sh` | Trains EncPromptLoRA **from scratch** on VTUAV (backbone initialized from `SwinTrack-B-384.pth`, everything else random). Config: `config/SwinRGBTrack/Base-384-enc-promptlora-vtuav`; automatically runs a full `vtuav_st` evaluation every 5 epochs. Default output: `output/enc_promptlora_vtuav/` |
| `train_lasher_ft_from_vtuav.sh` | Uses a VTUAV checkpoint as a **warm start** and incrementally fine-tunes on the LasHeR training set (979 sequences) to adapt the cross-domain ability to the LasHeR domain; automatically runs a full LasHeR test(245) every epoch and prints `[RESULT]` to help pick the best epoch |

```bash
# VTUAV from scratch (server)
bash train_vtuav_promptlora.sh
VTUAV_HOME=/data/VTUAV bash train_vtuav_promptlora.sh

# VTUAV → LasHeR incremental fine-tuning
bash train_lasher_ft_from_vtuav.sh -c output/enc_promptlora_vtuav/checkpoint_epoch010.pth

#  -c, --checkpoint PATH   VTUAV checkpoint (first-round warm start)
#  -o, --output-dir PATH   output dir (default <proj>/output/enc_promptlora_lasher_ft)
#  -e, --epochs N          target total epochs including resume (default 6)
#  -d, --data-root PATH    LasHeR root (default /root/RGBTData/LasHeR)
#  -b, --batch-size N      default 8
#  -l, --lr FLOAT          prompt/lora/head learning rate (default 5e-5)
#  -w, --workers N         default 4
# env vars: AUTO_TEST / TEST_INTERVAL / TEST_WORKERS / TEST_TIMEOUT /
#           BACKBONE_LR / FREEZE_STEM / SAMPLES_PER_EPOCH
```

> `train_lasher_ft_from_vtuav.sh` supports **resume**: if the output directory already contains
> `checkpoint_epochXXX.pth` and the target epoch count has not been reached, it automatically
> resumes from the latest checkpoint — just re-run the same command after an interruption.

### 5.2 Comparison experiments (fusion positions and structural variants)

| Script | Configuration | Description |
|---|---|---|
| `train_rgbt.sh` | Fixed mean fusion after the backbone | Base `RGBTSwinTrack`, runs through the config-driven framework entry |
| `train_rgbt_enc_fuse.sh` | Mean fusion after the encoder (EncFuse) | Fine-tunes EncFuse on LasHeR |
| `train_rgbt_concat_fuse.sh` | Concat fusion after the backbone | Freezes Backbone+Encoder+Decoder, trains only `fusion_proj` + head (LR=1e-3) |
| `train_rgbt_dec_fuse.sh` | Post-decoder fusion baseline | Produces `output/dec_fuse_difnet/checkpoint_epoch005.pth`, used as the warm start for Adaptive / LayerFuse |
| `train_rgbt_dec_fuse_adaptive.sh` | Post-decoder adaptive gated fusion | Initialized from DecFuse ep5, scene-adaptive gating (≈0.4 M new parameters) |
| `train_rgbt_layer_fuse.sh` | Explicit fusion at every encoder layer | Layer-wise fusion + cross-layer aggregation + decoder; initializing from DecFuse ep5 is recommended |

**Usage:**

```bash
# Base RGBTSwinTrack (config-driven)
bash train_rgbt.sh <workspace_dir> [--resume PATH] [--weight_path PATH] \
    [--device_ids "0,1"] [--workers N] [--offline] [--evaluation_only]

# EncFuse fine-tuning (named arguments, defaults at the top of the script)
bash train_rgbt_enc_fuse.sh \
    --weight ./SwinTrack-B-384.pth --output_dir ./output/rgbt_finetune \
    --batch_size 32 --epochs 50 --lr 1e-4 --backbone_lr 1e-5 \
    --freeze_backbone_epochs 3 --warmup_epochs 2 --workers 4 --seed 42 --num_gpus 1

# ConcatFuse (positional arguments)
bash train_rgbt_concat_fuse.sh [weight] [output_dir] [batch_size] [epochs]

# DecFuse-Adaptive
bash train_rgbt_dec_fuse_adaptive.sh [--weight P] [--output_dir P] [--batch_size N] \
    [--epochs N] [--lr F] [--backbone_lr F] [--freeze_epochs N] [--workers N] [--resume P]

# LayerFuse (named arguments, same style as enc_fuse)
bash train_rgbt_layer_fuse.sh [--weight P] [--output_dir P] [--batch_size N] \
    [--epochs N] [--lr F] [--backbone_lr F] [--freeze_backbone_epochs N] \
    [--warmup_epochs N] [--workers N] [--seed N] [--resume P]
```

---

## 6. Testing Scripts

### 6.1 Final method — LasHeR

| Script | Usage | Description |
|---|---|---|
| `test_lasher_rgbt_promptlora.sh` | `./test_lasher_rgbt_promptlora.sh [epoch ...]` | Tests `output/enc_promptlora/`; with no arguments only `final_model.pth` is evaluated; `1 2 3` evaluates several epochs; results → `output/enc_promptlora/testresults/` |
| `test_lasher_rgbt_promptlora_fp32_stage1.sh` | same | Tests `output/enc_promptlora_fp32/` |
| `test_lasher_rgbt_promptlora_fp32_stage2.sh` | same | Tests `output/enc_promptlora_fp32_stage2/` |
| `test_lasher_rgbt_promptlora_fp32_stage3.sh` | same | Tests `output/enc_promptlora_fp32_stage3/` |

```bash
bash test_lasher_rgbt_promptlora.sh              # final only
bash test_lasher_rgbt_promptlora.sh 1 2 3        # epoch001/002/003
LASHER_ROOT=/home/fzg/data/lasher WORKERS=4 bash test_lasher_rgbt_promptlora_fp32_stage2.sh final
```

All of them call `test_lasher_rgbt_enc_fuse.py --model_type enc_promptlora` underneath.

### 6.2 Final method — VTUAV / GTOT / RGBT210 / RGBT234

| Script | Usage | Description |
|---|---|---|
| `test_vtuav_promptlora.sh` | `bash test_vtuav_promptlora.sh [dataset] [epoch\|latest\|final] [all]` | Main test script; defaults to `vtuav_st` + latest epoch; for GTOT/RGBT210/RGBT234 it automatically calls the RGBT toolkit to produce official metrics |
| `test_vtuav_checkpoint.sh` | `bash test_vtuav_checkpoint.sh <ckpt.pth> [dataset] [workers]` | Single test for an **arbitrary** checkpoint; `dataset` ∈ `vtuav_st` (default) / `vtuav_lt` / `gtot` / `rgbt210` / `rgbt234` / `all` |
| `test_vtuav_multi_checkpoints.sh` | `bash ... -c CKPT -c CKPT [-d DATASET ...] [-w N] [-o DIR]` | Batch test over multiple checkpoints × multiple datasets |
| `test_lasher_vtuav_checkpoints.sh` | `bash ... -c CKPT [-c CKPT ...] [-d LASHER_ROOT] [-w N] [-o DIR] [-m MODEL_TYPE]` | Batch test on **LasHeR** with VTUAV weights (`enc_promptlora_vtuav`) |
| `test_encpromptlora_server_bench.sh` | `bash ... [-d CKPT_DIR] [-e EPOCH]... [-s DATASETS] [-w N] [-o DIR] [-m TYPE]` | Server batch: evaluates **every** `checkpoint_epoch*.pth` in the output directory on all 5 datasets and aggregates official metrics automatically |
| `test_encpromptlora_fp32_bench.sh` | same (defaults `-d output/enc_promptlora_fp32`, `-w 1`) | FP32 variant of the batch bench |

```bash
# Main script: common combinations
bash test_vtuav_promptlora.sh                          # vtuav_st, latest epoch
bash test_vtuav_promptlora.sh vtuav_lt 30              # dataset + epoch
bash test_vtuav_promptlora.sh gtot final               # use final_model.pth
bash test_vtuav_promptlora.sh vtuav_st 1 2 3           # several epochs
bash test_vtuav_promptlora.sh vtuav_st latest all      # all four datasets

# Arbitrary checkpoint
bash test_vtuav_checkpoint.sh output/enc_promptlora_vtuav/checkpoint_epoch005.pth all 4

# Batch bench (-s accepts comma / space / repeated forms)
bash test_encpromptlora_server_bench.sh -d output/enc_promptlora_stage2 -e 001 -e 002
bash test_encpromptlora_fp32_bench.sh -s "gtot rgbt234" -w 2
# dataset choices: vtuav_lt | vtuav_st | gtot | rgbt210 | rgbt234 | all
```

### 6.3 Comparison / ablation experiments

| Script | Usage | Underlying script |
|---|---|---|
| `test_lasher_rgbt_dec_fuse.sh` | `bash ... [weight] [workers]` | `test_lasher_rgbt_dec_fuse.py` (post-decoder fusion baseline) |
| `test_lasher_rgbt_dec_fuse_adaptive.sh` | `bash ... [weight] [workers]` | `test_lasher_rgbt_dec_fuse_adaptive.py` |
| `test_lasher_rgbt_enc_fuse.sh` | controlled by env vars (`LASHER_ROOT`/`WORKERS`) | `test_lasher_rgbt_enc_fuse.py` |
| `test_lasher_rgbt_concat_fuse.sh` | `bash ... [weight] [workers]` | `test_lasher_rgbt_concat_fuse.py` |
| `test_lasher_rgbt_layer_fuse.sh` | `bash ... [weight] [workers]` | `test_lasher_rgbt_layer_fuse.py` |
| `test_lasher_rgbt.sh` | controlled by env vars | `test_lasher_rgbt.py` (base RGBTSwinTrack) |
| `test_lasher_rgb.sh` | controlled by env vars | `test_lasher.py --modality rgb` (**RGB only**) |
| `test_lasher_tir.sh` | controlled by env vars | `test_lasher.py --modality tir` (**TIR only**) |

```bash
bash test_lasher_rgbt_layer_fuse.sh ./my_checkpoint.pth 8
bash test_lasher_rgb.sh         # RGB-only
bash test_lasher_tir.sh         # TIR-only
# common env vars: LASHER_ROOT (default /home/fzg/data/lasher), WORKERS
```

> Results are written to `./test_results/` by default.

---

## 7. Official Metric Recomputation (RGBT toolkit 1.0.1)

`recompute_official_rgbt_metrics.sh` recomputes the official metrics offline from already
tracked predictions, without running inference again:

```bash
bash recompute_official_rgbt_metrics.sh \
  gtot    output/batch_testresults/checkpoint_epoch005/gtot/swintrack_b384_enc_promptlora_vtuav \
  rgbt210 output/batch_testresults/checkpoint_epoch005/rgbt210/swintrack_b384_enc_promptlora_vtuav \
  rgbt234 output/batch_testresults/checkpoint_epoch005/rgbt234/swintrack_b384_enc_promptlora_vtuav \
  lasher  output/enc_promptlora_vtuav/lasher_testresults/checkpoint_epoch005/lasher/swintrack_b384_enc_promptlora_vtuav
# arguments: <dataset> <result_dir>, any number of pairs; dataset ∈ {gtot, rgbt210, rgbt234, lasher}
# env vars: RGBT_TOOLKIT_HOME (default /root), SUMMARY_FILE
# each result directory gets: official_metrics.json, official_metrics.csv, summary.json
```

> Official LasHeR re-evaluation requires the file names inside the result directory to match
> the toolkit's own `lashertest.txt` (245 sequences) exactly.

---

## 8. Framework Entry Points (config-driven, native SwinTrack)

| Script | Usage |
|---|---|
| `run.sh` | `./run.sh <method_name> <config_name> [options]`, supporting `--output_dir`, `--device_ids`, `-W/--workers`, `--mixin`, `--resume`, `--weight_path`, `--offline`, `--evaluation_only`, etc. |
| `main.py` | Direct call: `python main.py <method_name> <config_name> [options]` |
| `sweep.py` | Hyper-parameter search together with `run.sh --do_sweep` |
| `conda_init.sh` | Creates a conda environment (note: it is named `SwinTrack`; use `mambavision_rgbt` for experiments) |

`method_name` and `config_name` map to `config/<method_name>/<config_name>/`. Available combinations:

```bash
# Base RGBTSwinTrack (0.5 mean fusion after the backbone)
python main.py RGBTSwinTrack Base-384 --output_dir /path/to/output --num_workers 4

# EncFuse (fusion after the encoder)
./run.sh RGBTSwinTrackEncFuse Base-384-enc-fuse --output_dir /path/to/output -W 4

# DecFuse / ConcatFuse
./run.sh RGBTSwinTrackDecFuse   Base-384-dec-fuse-adaptive --output_dir /path/to/output
./run.sh RGBTSwinTrackConcatFuse Base-384-concat-fuse      --output_dir /path/to/output
```

> ⚠️ **EncPromptLoRA is not registered in the framework's `models/methods/builder.py`**,
> so it **cannot** be started through `main.py`/`run.sh`. Use the standalone scripts from §5.1
> (`train_rgbt_enc_fuse_v1.py --model_type enc_promptlora`) instead.

---

## 9. Environment Variables

| Variable | Default | Purpose | Used by |
|---|---|---|---|
| `LASHER_ROOT` | `/home/fzg/data/lasher` | LasHeR data root (`train/`, `testingset/`) | All LasHeR training/testing scripts |
| `SGTEST_DATA_ROOT` | `/root/RGBTData` | Server data root for VTUAV/GTOT/RGBT210/RGBT234 | `test_vtuav_*`, `test_encpromptlora_*` |
| `RGBT_TOOLKIT_HOME` | `/root` | Root of the official RGBT toolkit 1.0.1 | Any script producing official GTOT/RGBT210/RGBT234/LasHeR metrics |
| `VTUAV_HOME` | `/root/RGBTData/VTUAV` | VTUAV data root (`train`/`test_ST`/`test_LT`) | `train_vtuav_promptlora.sh` |
| `WORKERS` | 4 (1 for some `test_lasher_*`) | Number of parallel test workers | All `test_lasher_*` |
| `SKIP_EXISTING` | `1` | `0` = force full re-test; `1` = skip inference for sequences that already have complete `.txt` results | `test_encpromptlora_*_bench.sh` |
| `MODEL_TYPE` | `enc_promptlora_vtuav` | Model type | `test_vtuav_multi_checkpoints.sh` |
| `AUTO_TEST` | `1` | `0` = disable the automatic full test after each epoch | `train_lasher_ft_from_vtuav.sh` |
| `TEST_INTERVAL` / `TEST_WORKERS` / `TEST_TIMEOUT` | `1` / `1` / `7200` | Auto-test interval / processes / timeout in seconds | `train_lasher_ft_from_vtuav.sh` |
| `BACKBONE_LR` / `FREEZE_STEM` / `SAMPLES_PER_EPOCH` | `0` / `1000` / `60000` | Backbone LR / stem freeze epochs / samples per epoch | `train_lasher_ft_from_vtuav.sh` |
| `SUMMARY_FILE` | `output/official_recompute_*.csv` | Status summary CSV for recomputation jobs | `recompute_official_rgbt_metrics.sh` |

---

## 10. Notes

1. **Hard-coded conda path**: every `.sh` depends on `/home/fzg/anaconda3/etc/profile.d/conda.sh` and the
   `mambavision_rgbt` environment; adjust these when moving to another machine.
2. **EncPromptLoRA does not go through `main.py`**: it can only be run with the standalone scripts such as
   `train_rgbt_enc_fuse_v1.py` / `test_lasher_rgbt_enc_fuse.py`.
3. **GPU memory advice**: for the 245 LasHeR test sequences each worker loads a full model independently;
   on a 24 GB GPU prefer `workers=1~2` and run multiple checkpoints sequentially.

---

## Acknowledgements

This project extends the code framework of [SwinTrack](https://arxiv.org/abs/2112.00995).
Evaluation relies on the LasHeR, GTOT, RGBT210, RGBT234 and VTUAV datasets and on RGBT toolkit 1.0.1.
