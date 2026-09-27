#!/bin/bash
# =============================================================================
# RGBTSwinTrack-EncPromptLoRA — 对任意 checkpoint 进行测试 (服务器)
# =============================================================================
# 用法:
#   bash test_vtuav_checkpoint.sh <checkpoint.pth> [dataset] [workers]
#
#   dataset: vtuav_st (默认) | vtuav_lt | gtot | rgbt210 | rgbt234 | all
#   RGBT_TOOLKIT_HOME: RGBT toolkit 1.0.1 根目录；GTOT/RGBT210/RGBT234
#                      自动用该工具包输出论文正式指标。
#
# 示例:
#   bash test_vtuav_checkpoint.sh output/enc_promptlora_vtuav/checkpoint_epoch005.pth
#   bash test_vtuav_checkpoint.sh output/enc_promptlora_vtuav/checkpoint_epoch005.pth gtot
#   bash test_vtuav_checkpoint.sh output/enc_promptlora_vtuav/checkpoint_epoch005.pth all 4
#
# 结果保存到 <checkpoint所在目录>/testresults/<checkpoint名>/<dataset>/
# =============================================================================
set -eo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJ_ROOT}"

# ── 参数 ──
CHECKPOINT="${1:?用法: bash test_vtuav_checkpoint.sh <checkpoint.pth> [dataset] [workers]}"
DATASET="${2:-vtuav_st}"
WORKERS="${3:-${WORKERS:-4}}"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "[ERROR] checkpoint 不存在: ${CHECKPOINT}"
    exit 1
fi

SGTEST_DATA_ROOT="${SGTEST_DATA_ROOT:-/root/RGBTData}"
export RGBT_TOOLKIT_HOME="${RGBT_TOOLKIT_HOME:-/root}"

CKPT_ABS="$(cd "$(dirname "${CHECKPOINT}")" && pwd)/$(basename "${CHECKPOINT}")"
CKPT_TAG="$(basename "${CHECKPOINT}" .pth)"
SAVE_DIR="$(dirname "${CKPT_ABS}")/testresults/${CKPT_TAG}"

if [ "${DATASET}" = "all" ]; then
    DATASETS=("vtuav_st" "vtuav_lt" "gtot" "rgbt210" "rgbt234")
else
    DATASETS=("${DATASET}")
fi

echo "========================================"
echo "  EncPromptLoRA 单 checkpoint 测试"
echo "========================================"
echo "  Checkpoint: ${CKPT_ABS}"
echo "  Datasets:   ${DATASETS[*]}"
echo "  数据根目录: ${SGTEST_DATA_ROOT}"
echo "  官方评测包: ${RGBT_TOOLKIT_HOME}"
echo "  结果目录:   ${SAVE_DIR}/<dataset>/"
echo "  Workers:    ${WORKERS}"
echo "========================================"

mkdir -p "${SAVE_DIR}"

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "────────────────────────────────────────"
    echo "  测试 ${ds} @ ${CKPT_TAG}"
    echo "────────────────────────────────────────"
    python -u test_vtuav_promptlora.py \
        --weight "${CKPT_ABS}" \
        --model_type enc_promptlora_vtuav \
        --dataset "${ds}" \
        --data_root "${SGTEST_DATA_ROOT}" \
        --save_dir "${SAVE_DIR}" \
        --workers "${WORKERS}"
    echo "[DONE] ${ds} 测试完成"
done

echo ""
echo "全部测试完成! 结果: ${SAVE_DIR}/"
