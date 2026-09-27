#!/bin/bash
# ConcatFuse 测试脚本
set -e
source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"

WEIGHT="${1:-${PROJ_ROOT}/SwinTrack-B-384.pth}"
WORKERS="${2:-4}"
LASHER_ROOT="${LASHER_ROOT:-/home/fzg/data/lasher}"

echo "========================================"
echo "  ConcatFuse LasHeR 测试 (concat融合)"
echo "========================================"
echo "  权重: ${WEIGHT}"
echo "  数据: ${LASHER_ROOT}/testingset"
echo "  Workers: ${WORKERS}"
echo "========================================"

cd "${PROJ_ROOT}"
python test_lasher_rgbt_concat_fuse.py \
    --weight "${WEIGHT}" \
    --dataset_root "${LASHER_ROOT}/testingset" \
    --save_dir "${PROJ_ROOT}/test_results" \
    --workers "${WORKERS}"
