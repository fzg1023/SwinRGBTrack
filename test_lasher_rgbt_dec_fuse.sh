#!/bin/bash
# =============================================================================
# RGBTSwinTrack-DecFuse — LasHeR 测试脚本
# Decoder后融合, Head前0.5fuse, 直接加载预训练权重
#
# 用法:
#   bash test_lasher_rgbt_dec_fuse.sh                          # 使用默认权重
#   bash test_lasher_rgbt_dec_fuse.sh ./my_checkpoint.pth      # 指定权重
#   bash test_lasher_rgbt_dec_fuse.sh ./my_checkpoint.pth 8    # 指定权重+workers
# =============================================================================

set -e

source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate DSwinTrack

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── 默认参数 ──
WEIGHT="${1:-${PROJ_ROOT}/SwinTrack-B-384.pth}"
WORKERS="${2:-4}"
LASHER_ROOT="${LASHER_ROOT:-/home/fzg/data/lasher}"
SAVE_DIR="${PROJ_ROOT}/test_results"

echo "========================================"
echo "  RGBTSwinTrack-DecFuse LasHeR 测试"
echo "  (Decoder后融合, Head前0.5fuse)"
echo "========================================"
echo "  项目目录: ${PROJ_ROOT}"
echo "  权重文件: ${WEIGHT}"
echo "  数据集:   ${LASHER_ROOT}/testingset"
echo "  结果目录: ${SAVE_DIR}"
echo "  Workers:  ${WORKERS}"
echo "========================================"

cd "${PROJ_ROOT}"

python test_lasher_rgbt_dec_fuse.py \
    --weight "${WEIGHT}" \
    --dataset_root "${LASHER_ROOT}/testingset" \
    --save_dir "${SAVE_DIR}" \
    --workers "${WORKERS}"
