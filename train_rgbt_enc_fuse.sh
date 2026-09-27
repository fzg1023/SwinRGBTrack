#!/bin/bash
# =============================================================================
# RGBTSwinTrack — LasHeR 训练集微调 训练脚本
# 参考 OmniAdapt 的数据管线 (mmap 缓存 / 文件索引 / bbox 缓冲)
# =============================================================================

set -e

# Conda 环境设置
source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

# 项目根目录
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── 默认参数 ──
WEIGHT="${PROJ_ROOT}/SwinTrack-B-384.pth"
LASHER_ROOT="${LASHER_ROOT:-/home/fzg/data/lasher}"
OUTPUT_DIR="${PROJ_ROOT}/output/rgbt_finetune"
BATCH_SIZE=32
EPOCHS=50
LR=1e-4
BACKBONE_LR=1e-5
FREEZE_BACKBONE_EPOCHS=3
WARMUP_EPOCHS=2
WORKERS=4
SEED=42
NUM_GPUS=1

# ── 解析参数 ──
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --weight)        WEIGHT="$2"; shift ;;
        --output_dir)    OUTPUT_DIR="$2"; shift ;;
        --lasher_root)   LASHER_ROOT="$2"; shift ;;
        --batch_size)    BATCH_SIZE="$2"; shift ;;
        --epochs)        EPOCHS="$2"; shift ;;
        --lr)            LR="$2"; shift ;;
        --backbone_lr)   BACKBONE_LR="$2"; shift ;;
        --freeze_backbone_epochs) FREEZE_BACKBONE_EPOCHS="$2"; shift ;;
        --warmup_epochs) WARMUP_EPOCHS="$2"; shift ;;
        --workers)       WORKERS="$2"; shift ;;
        --seed)          SEED="$2"; shift ;;

        --num_gpus)      NUM_GPUS="$2"; shift ;;
        *) echo "Unknown param: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================"
echo "  RGBTSwinTrack LasHeR 微调训练"
echo "========================================"
echo "  预训练权重:         ${WEIGHT}"
echo "  输出目录:           ${OUTPUT_DIR}"
echo "  LasHeR 路径:        ${LASHER_ROOT}"
echo "  Batch Size:         ${BATCH_SIZE}"
echo "  Epochs:             ${EPOCHS}"
echo "  LR:                 ${LR}"
echo "  Backbone LR:        ${BACKBONE_LR}"
echo "  Freeze Backbone:    ${FREEZE_BACKBONE_EPOCHS} epochs"
echo "  Warmup:             ${WARMUP_EPOCHS} epochs"
echo "  Workers:            ${WORKERS}"
echo "  GPUs:               ${NUM_GPUS}"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

export OPENCV_LOG_LEVEL=OFF

LOG_FILE="${OUTPUT_DIR}/train_log_$(date +%Y%m%d_%H%M%S).txt"

cd "${PROJ_ROOT}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    # 多 GPU 分布式训练
    python -u -m torch.distributed.run \
        --nproc_per_node "${NUM_GPUS}" \
        --master_port $((RANDOM % 10000 + 20000)) \
        train_rgbt_enc_fuse.py \
        --weight "${WEIGHT}" \
        --output_dir "${OUTPUT_DIR}" \
        --lasher_root "${LASHER_ROOT}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --backbone_lr "${BACKBONE_LR}" \
        --freeze_backbone_epochs "${FREEZE_BACKBONE_EPOCHS}" \
        --warmup_epochs "${WARMUP_EPOCHS}" \
        --workers "${WORKERS}" \
        --seed "${SEED}" \
        2>&1 | tee "${LOG_FILE}"
else
    # 单 GPU 训练
    python -u train_rgbt_enc_fuse.py \
        --weight "${WEIGHT}" \
        --output_dir "${OUTPUT_DIR}" \
        --lasher_root "${LASHER_ROOT}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --backbone_lr "${BACKBONE_LR}" \
        --freeze_backbone_epochs "${FREEZE_BACKBONE_EPOCHS}" \
        --warmup_epochs "${WARMUP_EPOCHS}" \
        --workers "${WORKERS}" \
        --seed "${SEED}" \
        2>&1 | tee "${LOG_FILE}"
fi
