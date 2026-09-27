#!/bin/bash
# =============================================================================
# RGBTSwinTrack-LayerFuse — LasHeR 测试脚本
# Encoder逐层融合 + 跨层聚合 + Decoder
#
# 用法:
#   bash test_lasher_rgbt_layer_fuse.sh                          # 默认权重
#   bash test_lasher_rgbt_layer_fuse.sh ./my_checkpoint.pth      # 指定权重
#   bash test_lasher_rgbt_layer_fuse.sh ./my_checkpoint.pth 8    # 指定权重+workers
# =============================================================================

set -e

source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"

WEIGHT="${1:-${PROJ_ROOT}/SwinTrack-B-384.pth}"
WORKERS="${2:-4}"
LASHER_ROOT="${LASHER_ROOT:-/home/fzg/data/lasher}"
SAVE_DIR="${PROJ_ROOT}/test_results"

echo "========================================"
echo "  RGBTSwinTrack-LayerFuse LasHeR 测试"
echo "  (Encoder逐层融合 + 跨层聚合 + Decoder)"
echo "========================================"
echo "  权重文件: ${WEIGHT}"
echo "  数据集:   ${LASHER_ROOT}/testingset"
echo "  结果目录: ${SAVE_DIR}"
echo "  Workers:  ${WORKERS}"
echo "========================================"

cd "${PROJ_ROOT}"

python test_lasher_rgbt_layer_fuse.py \
    --weight "${WEIGHT}" \
    --dataset_root "${LASHER_ROOT}/testingset" \
    --save_dir "${SAVE_DIR}" \
    --workers "${WORKERS}"
