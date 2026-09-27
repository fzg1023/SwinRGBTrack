#!/bin/bash
set -e
source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
WEIGHT="${1:-${PROJ_ROOT}/SwinTrack-B-384.pth}"; WORKERS="${2:-4}"
LASHER_ROOT="${LASHER_ROOT:-/home/fzg/data/lasher}"; SAVE_DIR="${PROJ_ROOT}/test_results"
echo "========================================"
echo "  DecFuse-Adaptive Step2 LasHeR 测试"
echo "========================================"
echo "  权重: ${WEIGHT}  Workers: ${WORKERS}"
echo "========================================"
cd "${PROJ_ROOT}"
python -u test_lasher_rgbt_dec_fuse_adaptive.py --weight "${WEIGHT}" --dataset_root "${LASHER_ROOT}/testingset" --save_dir "${SAVE_DIR}" --workers "${WORKERS}"
