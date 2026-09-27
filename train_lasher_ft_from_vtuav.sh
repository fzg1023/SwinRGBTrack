#!/bin/bash
# =============================================================================
# VTUAV checkpoint 热启动 → LasHeR 训练集增量微调 EncPromptLoRA (服务器)
# =============================================================================
# 背景:
#   VTUAV 训练的 checkpoint (enc_promptlora_vtuav) 直接在 LasHeR 上零样本测试
#   只有 ~0.60 (跨域)。本脚本用该 checkpoint 作为热启动权重, 在 LasHeR 训练集
#   (train/, 979 序列) 上增量微调 PromptLoRA, 把跨域能力补到 LasHeR 域。
#   微调期间每轮自动在 LasHeR test (245) 全量测试并输出 [RESULT],
#   用于逐轮观察过拟合、挑选最优 epoch。
#
# 特性:
#   * 一次可跑多轮 (默认 6), 每轮自动全量测 test (test_interval=1)
#   * 自动断点续训: 输出目录已有 checkpoint_epochXXX.pth 且未达到目标轮数时,
#     自动从最新 checkpoint 续训 (中断后重跑同一条命令即可续上)
#   * 显存安全默认 (24GB 卡): 全程冻结 backbone + stem(encoder/decoder),
#     只训 prompt/LoRA + head —— 这正是本项目 LasHeR 最优 EncPromptLoRA
#     (stage1→2, PS 0.7117→0.7243) 采用的安全范式。
#     警告: 若把 stem/backbone 解冻 (FREEZE_STEM 小 + BACKBONE_LR>0),
#     24GB 卡必须把 batch 降到 4~8, 否则 OOM (已在 epoch3 实测)。
#
# 用法 (服务器):
#   bash train_lasher_ft_from_vtuav.sh \
#       -c output/enc_promptlora_vtuav/checkpoint_epoch010.pth
#
# 参数:
#   -c, --checkpoint PATH   VTUAV checkpoint (首轮热启动权重)
#   -o, --output-dir PATH   输出目录, 默认 <proj>/output/enc_promptlora_lasher_ft
#   -e, --epochs N          目标总轮数 (含续训), 默认 6
#   -d, --data-root PATH    LasHeR 根目录 (含 train/ + test/ 或 trainingset+testingset),
#                           默认 /root/RGBTData/LasHeR
#   -b, --batch-size N      默认 8 (24GB 卡; 解冻 stem/backbone 时需更小)
#   -l, --lr FLOAT          prompt/lora/head 学习率, 默认 5e-5
#   -w, --workers N         训练 dataloader worker, 默认 4
#   -h, --help
#
# 环境变量:
#   AUTO_TEST=0             关闭每轮自动全量测试 (默认 1=开)
#   TEST_INTERVAL=N         每隔 N 轮全量测一次 (默认 1=每轮)
#   TEST_WORKERS=N          自动测试进程数 (默认 1; 需与训练进程共享显存)
#   TEST_TIMEOUT=N          单次测试超时秒 (默认 7200)
#   BACKBONE_LR             全程冻结 backbone 时设 0 (默认 0, 最省显存)
#   FREEZE_STEM=N           stem 冻结轮数 (默认 1000≈全程; 想解冻 encoder 再调小)
#   SAMPLES_PER_EPOCH=N     每轮样本数 (默认 60000, 时间紧可 30000)
# =============================================================================
set -o pipefail

usage() { sed -n '3,55p' "$0"; }

CHECKPOINT=""
OUTPUT_DIR=""
EPOCHS="${EPOCHS:-6}"
DATA_ROOT="${LASHER_ROOT:-/root/RGBTData/LasHeR}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-5}"
# 全程冻结 backbone: 最省显存且与本项目 LasHeR 最优训练范式一致
BACKBONE_LR="${BACKBONE_LR:-0}"
FREEZE_STEM="${FREEZE_STEM:-1000}"
FREEZE_BB="${FREEZE_BB:-1000}"
WORKERS="${WORKERS:-4}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-60000}"
SEED="${SEED:-42}"
AUTO_TEST="${AUTO_TEST:-1}"
TEST_INTERVAL="${TEST_INTERVAL:-1}"
TEST_WORKERS="${TEST_WORKERS:-1}"
TEST_TIMEOUT="${TEST_TIMEOUT:-7200}"

while [ $# -gt 0 ]; do
    case "$1" in
        -c|--checkpoint) CHECKPOINT="$2"; shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -e|--epochs) EPOCHS="$2"; shift 2 ;;
        -d|--data-root) DATA_ROOT="$2"; shift 2 ;;
        -b|--batch-size) BATCH_SIZE="$2"; shift 2 ;;
        -l|--lr) LR="$2"; shift 2 ;;
        -w|--workers) WORKERS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] 未知参数: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "[ERROR] 必须用 -c/--checkpoint 指定 VTUAV checkpoint。" >&2
    usage >&2
    exit 2
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_ROOT"

[ -f "$CHECKPOINT" ] || { echo "[ERROR] checkpoint 不存在: $CHECKPOINT" >&2; exit 1; }
[ -d "$DATA_ROOT" ]  || { echo "[ERROR] LasHeR 根目录不存在: $DATA_ROOT" >&2; exit 1; }
if [ ! -d "$DATA_ROOT/test" ] && [ ! -d "$DATA_ROOT/train" ] \
   && [ ! -d "$DATA_ROOT/testingset" ] && [ ! -d "$DATA_ROOT/trainingset" ]; then
    echo "[ERROR] $DATA_ROOT 不是有效 LasHeR 根 (需含 train/test 或 trainingset/testingset)" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/output/enc_promptlora_lasher_ft}"
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/train_log_$(date +%Y%m%d_%H%M%S).txt"

# ── 自动断点续训: 若输出目录已有 checkpoint_epochXXX.pth, 从最新续训 ──
LATEST_CKPT="$(ls -1 "$OUTPUT_DIR"/checkpoint_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)"
RESUME_FLAG=()
if [ -n "$LATEST_CKPT" ]; then
    latest_num="$(basename "$LATEST_CKPT" | sed -E 's/checkpoint_epoch([0-9]+)\.pth/\1/')"
    if [ "$((10#$latest_num))" -ge "$((10#$EPOCHS))" ]; then
        echo "[INFO] 已训练到 epoch $latest_num (>= 目标 $EPOCHS), 无需再训。"
        echo "      若要多训, 请用 -e 提高目标轮数。"
        exit 0
    fi
    echo "[INFO] 检测到已有 checkpoint: $LATEST_CKPT → 自动续训 (从 epoch $((10#$latest_num + 1)) 继续)"
    RESUME_FLAG=(--resume "$LATEST_CKPT")
fi

echo "=========================================================="
echo "  VTUAV → LasHeR 增量微调 EncPromptLoRA (多轮)"
echo "=========================================================="
echo "  热启动 checkpoint : $CHECKPOINT"
echo "  LasHeR 根目录     : $DATA_ROOT"
echo "  输出目录          : $OUTPUT_DIR"
echo "  目标轮数          : $EPOCHS"
echo "  Batch / 每轮样本  : $BATCH_SIZE / $SAMPLES_PER_EPOCH"
echo "  LR / Backbone LR  : $LR / $BACKBONE_LR"
echo "  Stem freeze       : ${FREEZE_STEM} epochs (prompt/LoRA 始终可训练)"
echo "  Backbone freeze   : ${FREEZE_BB} epochs"
echo "  自动全量测 test   : $([ "$AUTO_TEST" = "1" ] && echo "开 (每 ${TEST_INTERVAL} 轮)" || echo "关")"
echo "  数据              : $DATA_ROOT/train (979) 训练, test (245) 每轮评测"
echo "=========================================================="

EXTRA_ARGS=("${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}")
if [ "$AUTO_TEST" = "1" ]; then
    EXTRA_ARGS+=(--auto_test --test_interval "$TEST_INTERVAL"
                 --test_timeout "$TEST_TIMEOUT" --test_workers "$TEST_WORKERS")
else
    EXTRA_ARGS+=(--no_auto_test)
fi

# 默认走 warmup(1) + cosine 全程调度, 多轮安全 (LR 逐轮自然衰减)。
python -u train_rgbt_enc_fuse_v1.py \
    --model_type enc_promptlora_vtuav \
    --weight "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --lasher_root "$DATA_ROOT" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --samples_per_epoch "$SAMPLES_PER_EPOCH" \
    --lr "$LR" \
    --backbone_lr "$BACKBONE_LR" \
    --freeze_backbone_epochs "$FREEZE_BB" \
    --freeze_stem_epochs "$FREEZE_STEM" \
    --warmup_epochs 1 \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --amp \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    2>&1 | tee "$LOG"

echo ""
echo "=========================================================="
echo "  微调完成! 输出: $OUTPUT_DIR"
echo "  日志: $LOG"
echo ""
echo "  逐轮 [RESULT] (LasHeR test) 见日志; 挑选最优 epoch 后:"
echo "    bash test_lasher_vtuav_checkpoints.sh \\"
echo "      -c $OUTPUT_DIR/checkpoint_epochXXX.pth -w 2"
echo "=========================================================="
