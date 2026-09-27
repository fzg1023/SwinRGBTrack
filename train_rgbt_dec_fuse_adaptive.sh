#!/bin/bash
# =============================================================================
# DecFuse-Adaptive Step 2 — 场景自适应门控融合
# 从 DecFuse ep5 初始化, ~0.4M 新参数
# =============================================================================
set -e
source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEC_FUSE_CKPT="${PROJ_ROOT}/output/dec_fuse_difnet/checkpoint_epoch005.pth"
WEIGHT="${DEC_FUSE_CKPT}"; LASHER_ROOT="${LASHER_ROOT:-/home/fzg/data/lasher}"
OUTPUT_DIR="${PROJ_ROOT}/output/dec_fuse_adaptive"
BATCH_SIZE=16; EPOCHS=10; LR=1e-4; BACKBONE_LR=1e-5
FREEZE_EPOCHS=2; WARMUP_EPOCHS=1; GRAD_ACCUM=2; WORKERS=4; SEED=42

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --weight) WEIGHT="$2"; shift ;; --output_dir) OUTPUT_DIR="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;; --epochs) EPOCHS="$2"; shift ;;
        --lr) LR="$2"; shift ;; --backbone_lr) BACKBONE_LR="$2"; shift ;;
        --freeze_epochs) FREEZE_EPOCHS="$2"; shift ;; --workers) WORKERS="$2"; shift ;;
        --resume) RESUME="--resume $2"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac; shift
done

echo "========================================"
echo "  DecFuse-Adaptive Step 2 训练"
echo "  场景自适应门控融合 (~0.4M 新参数)"
echo "========================================"
echo "  初始化: ${WEIGHT}  输出: ${OUTPUT_DIR}"
echo "  BS=${BATCH_SIZE} Epochs=${EPOCHS} Freeze=${FREEZE_EPOCHS}"
echo "========================================"

LOG_FILE="${OUTPUT_DIR}/train_log_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJ_ROOT}"
python -u train_rgbt_dec_fuse_adaptive.py \
    --weight "${WEIGHT}" --output_dir "${OUTPUT_DIR}" --lasher_root "${LASHER_ROOT}" \
    --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} --backbone_lr ${BACKBONE_LR} \
    --freeze_epochs ${FREEZE_EPOCHS} --warmup_epochs ${WARMUP_EPOCHS} \
    --grad_accum ${GRAD_ACCUM} --workers ${WORKERS} --seed ${SEED} --amp \
    ${RESUME} 2>&1 | tee "${LOG_FILE}"
