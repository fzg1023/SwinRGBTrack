# SwinRGBTrack

**RGB-T 目标跟踪：共享层次化表示 + 关系层内跨模态自适应 + 编码器后融合**

[Pretrained weights,models and results](https://pan.baidu.com/s/1luaTWQ1YzmU-EwEmX9uKzw?pwd=SWIN 提取码: SWIN)

**简体中文** ｜ [English](./README_EN.md)

---

## 1. 方法概览

SwinRGBTrack 用一条共享的层次化视觉前端提取 RGB 与热红外（TIR）特征，在模板–搜索关系编码过程中进行轻量跨模态自适应，并在编码器输出处融合两种模态，最后经统一解码器完成分类与回归。

| 组件 | 实现位置 | 说明 |
|---|---|---|
| 共享 Swin-B 前 3 阶段 | `models/backbone/swin_transformer.py` | stride 16，512 通道；模板网格 12×12、搜索网格 24×24；仅第 3 阶段输出进入关系编码器 |
| 8 层双流关系编码器 | `models/methods/SwinTrack/modules/encoder/` | 每个模态内模板与搜索 token 拼接；两流共享核心参数 |
| 双向瓶颈 Prompt | `models/methods/SwinRGBTrack/network_enc_promptlora.py` | 512→16→512 瓶颈；把另一模态状态以零初始化残差注入每一层输入 |
| TIR 专属 QKV LoRA | `models/methods/SwinTrack/modules/lora.py` | 仅加在 TIR 流的注意力 QKV 投影上，rank 8、缩放 α=16；RGB 流保留共享核心映射 |
| 编码器后融合 | `network_enc_fuse.py`、`network_enc_fusemamba.py` | 等权 0.5 融合 + 共享通道 bias；另有前向/反向 Mamba 空间上下文残差分支 |
| 单层关系解码器 + Mlp 预测头 | `models/methods/SwinTrack/modules/decoder/`、`models/head/mlp.py` | 搜索 token 作 query，模板–搜索拼接序列作 key/value |
| 训练损失 | `criterion/modules/varifocal_loss.py`、`criterion/modules/iou_loss.py` | Varifocal 分类损失 + GIoU 回归损失，各权重 2 |

自适应模块共新增约 **0.402 M** 参数（Prompt 约 0.271 M，LoRA 约 0.131 M）。

### 主要结果

| 数据集 | 指标 |
|---|---|
| LasHeR（245 条公共测试序列） | PR **72.68** / NPR **68.70** / SR **57.67** |
| GTOT（全量 50 序列） | MPR **91.72** / MSR **78.68** |
| RGBT210 | PR **84.19** / SR **62.02** |
| RGBT234（全量） | MPR **85.67** / MSR **64.13** |
| VTUAV | ST **84.86 / 77.00**；LT **63.24 / 55.61** |

---

## 2. 环境准备

实验统一使用 conda 环境 **`mambavision_rgbt`**（脚本内已硬编码激活路径）。

```bash
# 方式一：使用 requirements.txt
conda create -y -n mambavision_rgbt python=3.9
conda activate mambavision_rgbt
pip install -r requirements.txt
# Mamba 融合残差需要
pip install mamba_ssm

# 方式二：参考旧脚本（注意它会新建名为 SwinTrack 的环境，实验请改用 mambavision_rgbt）
bash conda_init.sh
```

> ⚠️ 所有 `.sh` 脚本中都写有 `source /home/fzg/anaconda3/etc/profile.d/conda.sh`。
> 迁移到其它机器时需按实际 conda 安装路径修改该行。

主要依赖见 [`requirements.txt`](./requirements.txt)：`torch`、`torchvision`、`timm`、`fvcore`、`wandb`、`shapely`、`numpy`、`scipy`、`matplotlib`、`pyyaml` 等。

### 预训练权重

| 文件 | 用途 | 放置位置 |
|---|---|---|
| `SwinTrack-B-384.pth` | 骨干/模型训练热启动权重 | 项目根目录 |

---

## 3. 数据准备

### 3.1 LasHeR（本地主力数据集）

默认根目录由环境变量 `LASHER_ROOT` 控制，默认 `/home/fzg/data/lasher`：

```
/home/fzg/data/lasher
├── train/                 # 979 条训练序列（train_lasher_ft_from_vtuav.sh 使用）
│   └── <seq>/visible/  infrared/  ...
└── testingset/            # 245 条测试序列
    └── <seq>/visible/  infrared/  ...
```

### 3.2 服务器多数据集（VTUAV / GTOT / RGBT210 / RGBT234）

由 `SGTEST_DATA_ROOT` 控制，默认 `/root/RGBTData`：

```
/root/RGBTData
├── VTUAV/test_ST  VTUAV/test_LT
├── GTOT    (v/ i/ groundTruth_v.txt)
├── RGBT210 (visible/ infrared/ visible.txt)
└── RGBT234 (visible/ infrared/ visible.txt)
```

### 3.3 官方评测工具包（RGBT toolkit 1.0.1）

GTOT / RGBT210 / RGBT234 / LasHeR 的正式指标由官方 RGBT toolkit 1.0.1 汇总，
其根目录由 `RGBT_TOOLKIT_HOME` 指定（默认 `/root`）。

### 3.4 路径模板

`path.template.yaml` 是框架（`main.py`）配置驱动的路径模板，复制为 `path.yaml` 后填写：

```yaml
LASHER_PATH: '/home/fzg/data/lasher'
```

---

## 4. 脚本总览

仓库共 **33 个 `.sh`**，按用途分为四组：

| 分组 | 脚本 | 说明 |
|---|---|---|
| **A. 最终方法 EncPromptLoRA** | `train_rgbt_promptlora*.sh`、`train_lasher_ft_from_vtuav.sh`、`train_vtuav_promptlora.sh`、`test_lasher_rgbt_promptlora*.sh`、`test_vtuav_*.sh`、`test_encpromptlora_*.sh` | 正式结果的训练与测试 |
| **B. 对比 / 消融** | `train_rgbt.sh`、`train_rgbt_enc_fuse.sh`、`train_rgbt_concat_fuse.sh`、`train_rgbt_dec_fuse.sh`、`train_rgbt_dec_fuse_adaptive.sh`、`train_rgbt_layer_fuse.sh` + 对应 `test_*` | 融合位置与结构变体的对比实验 |
| **C. 评测工具** | `recompute_official_rgbt_metrics.sh`、`test_lasher_rgb.sh`、`test_lasher_tir.sh` | 官方指标重评、单模态对比 |
| **D. 框架入口** | `conda_init.sh`、`run.sh` | 环境创建与原生配置驱动入口 |

---

## 5. 训练脚本

### 5.1 最终方法 —— EncPromptLoRA（★ 核心）

**训练链路（FP32 + AdamW + 冻结骨干，仅训 Prompt/LoRA/head）：**

```
SwinTrack-B-384.pth
  └─(enc_fuse 微调)→ enc_fusemamba_fp32 ep2
       └─ stage1 (FP32, LR=1e-4, 3 epochs)  → output/enc_promptlora_fp32/
            └─ stage2 (FP32, LR=5e-5, 2 epochs) → output/enc_promptlora_fp32_stage2/
                 └─ stage3 (FP32, LR=2.5e-5, 2 epochs) → output/enc_promptlora_fp32_stage3/
```

| 脚本 | 热启动权重 | 训练/冻结设置 | 输出目录 |
|---|---|---|---|
| `train_rgbt_promptlora.sh` | `output/enc_fusemamba/checkpoint_epoch001.pth` | **AMP**，3 epochs，LR=1e-4 恒定；冻结 backbone+encoder(原始)+decoder+mamba 残差 | `output/enc_promptlora/` |
| `train_rgbt_promptlora_fp32_stage1.sh` | `output/enc_fusemamba_fp32/checkpoint_epoch002.pth` | **FP32 (`--no_amp`)**，3 epochs，LR=1e-4 | `output/enc_promptlora_fp32/` |
| `train_rgbt_promptlora_fp32_stage2.sh` | `output/enc_promptlora_fp32/checkpoint_epoch003.pth` | FP32，2 epochs，LR=5e-5（重新全程冻结） | `output/enc_promptlora_fp32_stage2/` |
| `train_rgbt_promptlora_fp32_stage3.sh` | `output/enc_promptlora_fp32_stage2/checkpoint_epoch001.pth` | FP32，2 epochs，LR=2.5e-5 | `output/enc_promptlora_fp32_stage3/` |

**用法**（脚本基本无参数，直接运行；关键参数在脚本顶部常量中）：

```bash
bash train_rgbt_promptlora_fp32_stage1.sh
bash train_rgbt_promptlora_fp32_stage2.sh   # 需先完成 stage1
bash train_rgbt_promptlora_fp32_stage3.sh   # 需先完成 stage2
```

共同的训练超参（写死在脚本里）：`--batch_size 16 --samples_per_epoch 60000 --lr ... --backbone_lr 1e-5
--freeze_backbone_epochs 100(全程) --freeze_stem_epochs 100 --freeze_fusion_epochs 100 --const_lr --seed 42`，
底层调用 `train_rgbt_enc_fuse_v1.py --model_type enc_promptlora`。

**VTUAV 分支：**

| 脚本 | 说明 |
|---|---|
| `train_vtuav_promptlora.sh` | 在 VTUAV 上**从头训练** EncPromptLoRA（骨干用 `SwinTrack-B-384.pth` 初始化，其余随机）。配置 `config/SwinRGBTrack/Base-384-enc-promptlora-vtuav`；每 5 轮自动在 `vtuav_st` 全量测试。输出默认 `output/enc_promptlora_vtuav/` |
| `train_lasher_ft_from_vtuav.sh` | 用 VTUAV checkpoint **热启动**，在 LasHeR 训练集（979 序列）上增量微调，把跨域能力补到 LasHeR 域；每轮自动全量测 LasHeR test(245)，输出 `[RESULT]` 便于挑最优 epoch |

```bash
# VTUAV 从头训练（服务器）
bash train_vtuav_promptlora.sh
VTUAV_HOME=/data/VTUAV bash train_vtuav_promptlora.sh

# VTUAV → LasHeR 增量微调
bash train_lasher_ft_from_vtuav.sh -c output/enc_promptlora_vtuav/checkpoint_epoch010.pth

#  -c, --checkpoint PATH   VTUAV checkpoint（首轮热启动权重）
#  -o, --output-dir PATH   输出目录（默认 <proj>/output/enc_promptlora_lasher_ft）
#  -e, --epochs N          目标总轮数（含续训，默认 6）
#  -d, --data-root PATH    LasHeR 根目录（默认 /root/RGBTData/LasHeR）
#  -b, --batch-size N      默认 8
#  -l, --lr FLOAT          prompt/lora/head 学习率（默认 5e-5）
#  -w, --workers N         默认 4
# 环境变量：AUTO_TEST / TEST_INTERVAL / TEST_WORKERS / TEST_TIMEOUT /
#          BACKBONE_LR / FREEZE_STEM / SAMPLES_PER_EPOCH
```

> `train_lasher_ft_from_vtuav.sh` 支持**断点续训**：输出目录已有 `checkpoint_epochXXX.pth` 且未达目标轮数时自动从最新检查点续训，中断后重跑同一条命令即可。

### 5.2 对比实验（融合位置与结构变体）

| 脚本 | 对应配置 | 说明 |
|---|---|---|
| `train_rgbt.sh` | 骨干后固定均值融合 | 基础 `RGBTSwinTrack`，走框架配置驱动 |
| `train_rgbt_enc_fuse.sh` | 编码器后均值融合（EncFuse） | 在 LasHeR 上微调 EncFuse |
| `train_rgbt_concat_fuse.sh` | 骨干后 concat 融合 | 冻结 Backbone+Encoder+Decoder，只训 `fusion_proj` + head（LR=1e-3） |
| `train_rgbt_dec_fuse.sh` | 解码器后融合基线 | 产出 `output/dec_fuse_difnet/checkpoint_epoch005.pth`，供 Adaptive / LayerFuse 热启动 |
| `train_rgbt_dec_fuse_adaptive.sh` | 解码器后自适应门控融合 | 从 DecFuse ep5 初始化，场景自适应门控（~0.4M 新参数） |
| `train_rgbt_layer_fuse.sh` | 逐编码层显式融合 | Encoder 逐层融合 + 跨层聚合 + Decoder，推荐从 DecFuse ep5 初始化 |

**用法：**

```bash
# 基础 RGBTSwinTrack（框架配置驱动）
bash train_rgbt.sh <workspace_dir> [--resume PATH] [--weight_path PATH] \
    [--device_ids "0,1"] [--workers N] [--offline] [--evaluation_only]

# EncFuse 微调（命名参数，默认见脚本顶部）
bash train_rgbt_enc_fuse.sh \
    --weight ./SwinTrack-B-384.pth --output_dir ./output/rgbt_finetune \
    --batch_size 32 --epochs 50 --lr 1e-4 --backbone_lr 1e-5 \
    --freeze_backbone_epochs 3 --warmup_epochs 2 --workers 4 --seed 42 --num_gpus 1

# ConcatFuse（位置参数）
bash train_rgbt_concat_fuse.sh [weight] [output_dir] [batch_size] [epochs]

# DecFuse-Adaptive
bash train_rgbt_dec_fuse_adaptive.sh [--weight P] [--output_dir P] [--batch_size N] \
    [--epochs N] [--lr F] [--backbone_lr F] [--freeze_epochs N] [--workers N] [--resume P]

# LayerFuse（命名参数，同 enc_fuse 风格）
bash train_rgbt_layer_fuse.sh [--weight P] [--output_dir P] [--batch_size N] \
    [--epochs N] [--lr F] [--backbone_lr F] [--freeze_backbone_epochs N] \
    [--warmup_epochs N] [--workers N] [--seed N] [--resume P]
```

---

## 6. 测试脚本

### 6.1 最终方法 —— LasHeR

| 脚本 | 用法 | 说明 |
|---|---|---|
| `test_lasher_rgbt_promptlora.sh` | `./test_lasher_rgbt_promptlora.sh [epoch ...]` | 测 `output/enc_promptlora/`；不带参数只测 `final_model.pth`；`1 2 3` 测多个 epoch；结果 → `output/enc_promptlora/testresults/` |
| `test_lasher_rgbt_promptlora_fp32_stage1.sh` | 同上 | 测 `output/enc_promptlora_fp32/` |
| `test_lasher_rgbt_promptlora_fp32_stage2.sh` | 同上 | 测 `output/enc_promptlora_fp32_stage2/` |
| `test_lasher_rgbt_promptlora_fp32_stage3.sh` | 同上 | 测 `output/enc_promptlora_fp32_stage3/` |

```bash
bash test_lasher_rgbt_promptlora.sh              # 只测 final
bash test_lasher_rgbt_promptlora.sh 1 2 3        # 测 epoch001/002/003
LASHER_ROOT=/home/fzg/data/lasher WORKERS=4 bash test_lasher_rgbt_promptlora_fp32_stage2.sh final
```

底层统一调用 `test_lasher_rgbt_enc_fuse.py --model_type enc_promptlora`。

### 6.2 最终方法 —— VTUAV / GTOT / RGBT210 / RGBT234

| 脚本 | 用法 | 说明 |
|---|---|---|
| `test_vtuav_promptlora.sh` | `bash test_vtuav_promptlora.sh [dataset] [epoch\|latest\|final] [all]` | 主测试脚本，默认 `vtuav_st` + 最新 epoch；GTOT/RGBT210/RGBT234 完成后自动调用 RGBT toolkit 输出官方指标 |
| `test_vtuav_checkpoint.sh` | `bash test_vtuav_checkpoint.sh <ckpt.pth> [dataset] [workers]` | 对**任意** checkpoint 单次测试；`dataset` 取 `vtuav_st`(默认)/`vtuav_lt`/`gtot`/`rgbt210`/`rgbt234`/`all` |
| `test_vtuav_multi_checkpoints.sh` | `bash ... -c CKPT -c CKPT [-d DATASET ...] [-w N] [-o DIR]` | 多 checkpoint × 多数据集批量测试 |
| `test_lasher_vtuav_checkpoints.sh` | `bash ... -c CKPT [-c CKPT ...] [-d LASHER_ROOT] [-w N] [-o DIR] [-m MODEL_TYPE]` | 用 VTUAV 权重（`enc_promptlora_vtuav`）在 **LasHeR** 上批量测试 |
| `test_encpromptlora_server_bench.sh` | `bash ... [-d CKPT_DIR] [-e EPOCH]... [-s DATASETS] [-w N] [-o DIR] [-m TYPE]` | 服务器批量：对输出目录下**所有** `checkpoint_epoch*.pth` 逐个在 5 个数据集上测试并自动汇总官方指标 |
| `test_encpromptlora_fp32_bench.sh` | 同上（默认 `-d output/enc_promptlora_fp32`，`-w 1`） | FP32 版本的批量 bench |

```bash
# 主脚本：常用组合
bash test_vtuav_promptlora.sh                          # vtuav_st, 最新 epoch
bash test_vtuav_promptlora.sh vtuav_lt 30              # 指定数据集 + epoch
bash test_vtuav_promptlora.sh gtot final               # 用 final_model.pth
bash test_vtuav_promptlora.sh vtuav_st 1 2 3           # 多 epoch
bash test_vtuav_promptlora.sh vtuav_st latest all      # 四个数据集全测

# 任意 checkpoint
bash test_vtuav_checkpoint.sh output/enc_promptlora_vtuav/checkpoint_epoch005.pth all 4

# 批量 bench（-s 支持逗号/空格/重复三种写法）
bash test_encpromptlora_server_bench.sh -d output/enc_promptlora_stage2 -e 001 -e 002
bash test_encpromptlora_fp32_bench.sh -s "gtot rgbt234" -w 2
# 数据集可选值: vtuav_lt | vtuav_st | gtot | rgbt210 | rgbt234 | all
```

### 6.3 对比 / 消融实验

| 脚本 | 用法 | 底层脚本 |
|---|---|---|
| `test_lasher_rgbt_dec_fuse.sh` | `bash ... [weight] [workers]` | `test_lasher_rgbt_dec_fuse.py`（解码器后融合基线） |
| `test_lasher_rgbt_dec_fuse_adaptive.sh` | `bash ... [weight] [workers]` | `test_lasher_rgbt_dec_fuse_adaptive.py` |
| `test_lasher_rgbt_enc_fuse.sh` | 环境变量控制（`LASHER_ROOT`/`WORKERS`） | `test_lasher_rgbt_enc_fuse.py` |
| `test_lasher_rgbt_concat_fuse.sh` | `bash ... [weight] [workers]` | `test_lasher_rgbt_concat_fuse.py` |
| `test_lasher_rgbt_layer_fuse.sh` | `bash ... [weight] [workers]` | `test_lasher_rgbt_layer_fuse.py` |
| `test_lasher_rgbt.sh` | 环境变量控制 | `test_lasher_rgbt.py`（基础 RGBTSwinTrack） |
| `test_lasher_rgb.sh` | 环境变量控制 | `test_lasher.py --modality rgb`（**仅 RGB**） |
| `test_lasher_tir.sh` | 环境变量控制 | `test_lasher.py --modality tir`（**仅 TIR**） |

```bash
bash test_lasher_rgbt_layer_fuse.sh ./my_checkpoint.pth 8
bash test_lasher_rgb.sh         # RGB-only
bash test_lasher_tir.sh         # TIR-only
# 通用环境变量：LASHER_ROOT（默认 /home/fzg/data/lasher）、WORKERS
```

> 结果默认保存在 `./test_results/`。

---

## 7. 官方指标重评（RGBT toolkit 1.0.1）

`recompute_official_rgbt_metrics.sh` 对**已完成跟踪**的预测结果离线重算官方指标，不重新推理：

```bash
bash recompute_official_rgbt_metrics.sh \
  gtot    output/batch_testresults/checkpoint_epoch005/gtot/swintrack_b384_enc_promptlora_vtuav \
  rgbt210 output/batch_testresults/checkpoint_epoch005/rgbt210/swintrack_b384_enc_promptlora_vtuav \
  rgbt234 output/batch_testresults/checkpoint_epoch005/rgbt234/swintrack_b384_enc_promptlora_vtuav \
  lasher  output/enc_promptlora_vtuav/lasher_testresults/checkpoint_epoch005/lasher/swintrack_b384_enc_promptlora_vtuav
# 参数：<dataset> <result_dir>，可连续传多组；dataset ∈ {gtot, rgbt210, rgbt234, lasher}
# 环境变量：RGBT_TOOLKIT_HOME（默认 /root）、SUMMARY_FILE
# 每个结果目录生成/更新：official_metrics.json、official_metrics.csv、summary.json
```

> LasHeR 官方重评要求结果目录内的文件名与 toolkit 自带 `lashertest.txt`（245 条序列）完全一致。

---

## 8. 框架入口（配置驱动，SwinTrack 原生）

| 脚本 | 用法 |
|---|---|
| `run.sh` | `./run.sh <method_name> <config_name> [options]`，支持 `--output_dir`、`--device_ids`、`-W/--workers`、`--mixin`、`--resume`、`--weight_path`、`--offline`、`--evaluation_only` 等 |
| `main.py` | 直接调用：`python main.py <method_name> <config_name> [options]` |
| `sweep.py` | 配合 `run.sh --do_sweep` 做超参搜索 |
| `conda_init.sh` | 创建 conda 环境（注意环境名为 `SwinTrack`，实验请用 `mambavision_rgbt`） |

`method_name` 与 `config_name` 对应 `config/<method_name>/<config_name>/`。可用组合：

```bash
# 基础 RGBTSwinTrack（骨干后 0.5 融合）
python main.py RGBTSwinTrack Base-384 --output_dir /path/to/output --num_workers 4

# EncFuse（编码器后融合）
./run.sh RGBTSwinTrackEncFuse Base-384-enc-fuse --output_dir /path/to/output -W 4

# DecFuse / ConcatFuse
./run.sh RGBTSwinTrackDecFuse   Base-384-dec-fuse-adaptive --output_dir /path/to/output
./run.sh RGBTSwinTrackConcatFuse Base-384-concat-fuse      --output_dir /path/to/output
```

> ⚠️ **EncPromptLoRA 未注册到框架的 `models/methods/builder.py`**，
> 因此它**不能**通过 `main.py`/`run.sh` 启动，必须使用 §5.1 的独立脚本
> （`train_rgbt_enc_fuse_v1.py --model_type enc_promptlora`）。

---

## 9. 环境变量汇总

| 变量 | 默认值 | 用途 | 涉及脚本 |
|---|---|---|---|
| `LASHER_ROOT` | `/home/fzg/data/lasher` | LasHeR 数据根（含 `train/`、`testingset/`） | 全部 LasHeR 训练/测试脚本 |
| `SGTEST_DATA_ROOT` | `/root/RGBTData` | 服务器 VTUAV/GTOT/RGBT210/RGBT234 数据根 | `test_vtuav_*`、`test_encpromptlora_*` |
| `RGBT_TOOLKIT_HOME` | `/root` | 官方 RGBT toolkit 1.0.1 根目录 | 凡涉及 GTOT/RGBT210/RGBT234/LasHeR 官方指标 |
| `VTUAV_HOME` | `/root/RGBTData/VTUAV` | VTUAV 数据根（`train`/`test_ST`/`test_LT`） | `train_vtuav_promptlora.sh` |
| `WORKERS` | 4（`test_lasher_*` 部分为 1） | 测试并行 worker 数 | 各 `test_lasher_*` |
| `SKIP_EXISTING` | `1` | `0`=强制全量重测；`1`=已有完整 `.txt` 结果的序列跳过推理 | `test_encpromptlora_*_bench.sh` |
| `MODEL_TYPE` | `enc_promptlora_vtuav` | 模型类型 | `test_vtuav_multi_checkpoints.sh` |
| `AUTO_TEST` | `1` | `0`=关闭每轮自动全量测试 | `train_lasher_ft_from_vtuav.sh` |
| `TEST_INTERVAL` / `TEST_WORKERS` / `TEST_TIMEOUT` | `1` / `1` / `7200` | 自动测试间隔 / 进程数 / 超时秒 | `train_lasher_ft_from_vtuav.sh` |
| `BACKBONE_LR` / `FREEZE_STEM` / `SAMPLES_PER_EPOCH` | `0` / `1000` / `60000` | 骨干学习率 / stem 冻结轮数 / 每轮样本数 | `train_lasher_ft_from_vtuav.sh` |
| `SUMMARY_FILE` | `output/official_recompute_*.csv` | 重评任务状态汇总 CSV | `recompute_official_rgbt_metrics.sh` |

---

## 10. 注意事项

1. **conda 路径硬编码**：所有 `.sh` 依赖 `/home/fzg/anaconda3/etc/profile.d/conda.sh` 与 `mambavision_rgbt` 环境，迁移时需修改。
2. **EncPromptLoRA 不走 `main.py`**：该模型只能通过 `train_rgbt_enc_fuse_v1.py` / `test_lasher_rgbt_enc_fuse.py` 等独立脚本运行。
3. **显存建议**：LasHeR 245 条测试序列，每个 worker 会独立加载完整模型；24 GB 卡建议 `workers=1~2`，多 checkpoint 顺序执行。

---

## 致谢

本项目基于 [SwinTrack](https://arxiv.org/abs/2112.00995) 的代码框架扩展而来，
评测依赖 LasHeR、GTOT、RGBT210、RGBT234、VTUAV 数据集及 RGBT toolkit 1.0.1。
