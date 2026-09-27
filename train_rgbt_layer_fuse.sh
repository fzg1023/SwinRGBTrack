#!/bin/bash
# =============================================================================
# RGBTSwinTrack-LayerFuse — LasHeR 微调训练脚本
# Encoder逐层融合 + 跨层聚合 + Decoder
#
# 推荐: 从 DecFuse epoch5 最优权重初始化
#
# 用法:
#   bash train_rgbt_layer_fuse.sh                                    # 默认参数
#   bash train_rgbt_layer_fuse.sh --output_dir ./output/layer_fuse   # 自定义输出
# =============================================================================

set -e

source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── 推荐: 从 DecFuse epoch5 最优权重初始化 ──
DEC_FUSE_CKPT="${PROJ_ROOT}/output/dec_fuse_difnet/checkpoint_epoch005.pth"

# ── 默认参数 ──
WEIGHT="${DEC_FUSE_CKPT}"
LASHER_ROOT="${LASHER_ROOT:-/home/fzg/data/lasher}"
OUTPUT_DIR="${PROJ_ROOT}/output/layer_fuse"
BATCH_SIZE=16
EPOCHS=30
LR=1e-4
BACKBONE_LR=1e-5
FREEZE_BACKBONE_EPOCHS=2
WARMUP_EPOCHS=2
GRAD_ACCUM=2
AMP_FLAG="--amp"
WORKERS=4
SEED=42
NUM_GPUS=1
RESUME=""

# ── 解析参数 ──
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --weight)        WEIGHT="$2"; shift ;;
        --output_dir)    OUTPUT_DIR="$2"; shift ;;
        --lasher_root)   LASHER_ROOT="$2"; shift ;;
        --batch_size)    BATCH_SIZE="$2"; shift ;;
        --resume)        RESUME="--resume $2"; shift ;;
        --epochs)        EPOCHS="$2"; shift ;;
        --lr)            LR="$2"; shift ;;
        --backbone_lr)   BACKBONE_LR="$2"; shift ;;
        --freeze_backbone_epochs) FREEZE_BACKBONE_EPOCHS="$2"; shift ;;
        --warmup_epochs) WARMUP_EPOCHS="$2"; shift ;;
        --workers)       WORKERS="$2"; shift ;;
        --seed)          SEED="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================"
echo "  RGBTSwinTrack-LayerFuse LasHeR 微调训练"
echo "  (Encoder逐层融合 + 跨层聚合 + Decoder)"
echo "========================================"
echo "  初始化权重:         ${WEIGHT}"
echo "  输出目录:           ${OUTPUT_DIR}"
echo "  LasHeR 路径:        ${LASHER_ROOT}"
echo "  Batch Size:         ${BATCH_SIZE}"
echo "  Epochs:             ${EPOCHS}"
echo "  LR:                 ${LR}"
echo "  Backbone LR:        ${BACKBONE_LR}"
echo "  Freeze:             ${FREEZE_BACKBONE_EPOCHS} epochs"
echo "  Warmup:             ${WARMUP_EPOCHS} epochs"
echo "  Grad Accum:         ${GRAD_ACCUM}"
echo "  Workers:            ${WORKERS}"
echo "  GPUs:               ${NUM_GPUS}"
echo "========================================"

LOG_FILE="${OUTPUT_DIR}/train_log_$(date +%Y%m%d_%H%M%S).txt"

cd "${PROJ_ROOT}"

python -u train_rgbt_layer_fuse.py \
    --weight "${WEIGHT}" \
    --output_dir "${OUTPUT_DIR}" \
    --lasher_root "${LASHER_ROOT}" \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --backbone_lr ${BACKBONE_LR} \
    --freeze_backbone_epochs ${FREEZE_BACKBONE_EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --grad_accum ${GRAD_ACCUM} \
    --workers ${WORKERS} \
    --seed ${SEED} \
    ${AMP_FLAG} \
    ${RESUME} \
    2>&1 | tee "${LOG_FILE}"
