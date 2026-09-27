#!/bin/bash
# =============================================================================
# RGBTSwinTrack LasHeR RGBT (RGB+TIR) 双模态测试脚本
# =============================================================================

set -e

# Conda 环境设置
source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

# 项目根目录
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 权重文件
WEIGHT="${PROJ_ROOT}/SwinTrack-B-384.pth"

# LasHeR 测试集路径
LASHER_ROOT="${LASHER_ROOT:-/home/fzg/data/lasher}"

# 结果保存目录
SAVE_DIR="${PROJ_ROOT}/test_results"

# Worker 数量（默认 4，多GPU可调高）
WORKERS="${WORKERS:-4}"

echo "========================================"
echo "  RGBTSwinTrack-EncFuse LasHeR 测试 (Encoder后融合)"
echo "========================================"
echo "  项目目录: ${PROJ_ROOT}"
echo "  权重文件: ${WEIGHT}"
echo "  数据集:   ${LASHER_ROOT}/testingset"
echo "  结果目录: ${SAVE_DIR}"
echo "  Workers:  ${WORKERS}"
echo "========================================"

cd "${PROJ_ROOT}"

python test_lasher_rgbt_enc_fuse.py \
    --weight "${WEIGHT}" \
    --dataset_root "${LASHER_ROOT}/testingset" \
    --save_dir "${SAVE_DIR}" \
    --workers "${WORKERS}"
