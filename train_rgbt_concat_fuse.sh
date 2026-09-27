#!/bin/bash
# ConcatFuse 微调: 冻结 Backbone+Encoder+Decoder, 只训 concat融合+Head
set -e
source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"

WEIGHT="${1:-${PROJ_ROOT}/SwinTrack-B-384.pth}"
OUTPUT_DIR="${2:-${PROJ_ROOT}/output/concat_fuse}"
BATCH_SIZE="${3:-32}"
EPOCHS="${4:-30}"

echo "============================================"
echo "  ConcatFuse 微调 (fusion_proj + head only)"
echo "============================================"
echo "  预训练权重: ${WEIGHT}"
echo "  输出目录:   ${OUTPUT_DIR}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Epochs:     ${EPOCHS}"
echo "  LR:         1e-3"
echo "  可训参数:   fusion_proj + head (其余冻结)"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJ_ROOT}"
python -u train_rgbt_concat_fuse.py \
    --weight "${WEIGHT}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --workers 4 \
    2>&1 | tee "${OUTPUT_DIR}/train_log_$(date +%Y%m%d_%H%M%S).txt"
