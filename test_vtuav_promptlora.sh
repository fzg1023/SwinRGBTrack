#!/bin/bash
# =============================================================================
# RGBTSwinTrack-EncPromptLoRA — VTUAV / GTOT / RGBT210 / RGBT234 测试脚本
# =============================================================================
# 数据目录与 SGTrack 服务器组织一致 (可用环境变量覆盖):
#   SGTEST_DATA_ROOT = /root/RGBTData
#     vtuav_st : <root>/VTUAV/test_ST
#     vtuav_lt : <root>/VTUAV/test_LT
#     gtot     : <root>/GTOT
#     rgbt210  : <root>/RGBT210
#     rgbt234  : <root>/RGBT234
#   RGBT_TOOLKIT_HOME = /root
#     用于 GTOT/RGBT210/RGBT234 的官方指标汇总；服务器可覆盖此路径。
#
# 用法:
#   bash test_vtuav_promptlora.sh                     # vtuav_st, 最新 epoch
#   bash test_vtuav_promptlora.sh vtuav_lt 30         # 指定数据集 + epoch
#   bash test_vtuav_promptlora.sh gtot final          # 用 final_model.pth
#   bash test_vtuav_promptlora.sh vtuav_st 1 2 3      # 测试多个 epoch
#   bash test_vtuav_promptlora.sh vtuav_st latest all # 四个数据集全测
# =============================================================================
set -eo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJ_ROOT}"

CKPT_DIR="${CKPT_DIR:-${PROJ_ROOT}/output/enc_promptlora_vtuav}"
SGTEST_DATA_ROOT="${SGTEST_DATA_ROOT:-/root/RGBTData}"
export RGBT_TOOLKIT_HOME="${RGBT_TOOLKIT_HOME:-/root}"
SAVE_DIR="${SAVE_DIR:-${CKPT_DIR}/testresults}"
WORKERS="${WORKERS:-4}"

DATASET="${1:-vtuav_st}"
shift || true

# ── epoch 列表: 参数 / final / latest (最后一个 checkpoint) ──
if [ $# -eq 0 ]; then
    EPOCHS=("latest")
else
    EPOCHS=()
    for arg in "$@"; do
        case "${arg}" in
            final)   EPOCHS+=("final") ;;
            latest)  EPOCHS+=("latest") ;;
            all)     EPOCHS=("all") ;;
            *)       arg_clean="${arg#epoch}"
                     EPOCHS+=("$(printf "%03d" "$((10#$arg_clean))")") ;;
        esac
    done
fi

if [ "${DATASET}" = "all" ]; then
    DATASETS=("vtuav_st" "vtuav_lt" "gtot" "rgbt210" "rgbt234")
else
    DATASETS=("${DATASET}")
fi

echo "========================================"
echo "  EncPromptLoRA VTUAV/GTOT/RGBT 测试"
echo "========================================"
echo "  Checkpoint 目录: ${CKPT_DIR}"
echo "  Epochs:          ${EPOCHS[*]}"
echo "  Datasets:        ${DATASETS[*]}"
echo "  数据根目录:      ${SGTEST_DATA_ROOT}"
echo "  官方评测包:      ${RGBT_TOOLKIT_HOME}"
echo "========================================"

mkdir -p "${SAVE_DIR}"

for ds in "${DATASETS[@]}"; do
    for ep in "${EPOCHS[@]}"; do
        if [ "${ep}" = "final" ]; then
            CKPT="${CKPT_DIR}/final_model.pth"
            TAG="final"
        elif [ "${ep}" = "latest" ]; then
            CKPT=$(ls -t "${CKPT_DIR}"/checkpoint_epoch*.pth 2>/dev/null | head -1)
            if [ -z "${CKPT}" ]; then
                echo "[WARN] 无 checkpoint, 跳过 (${CKPT_DIR})"
                continue
            fi
            TAG="epoch$(basename "${CKPT}" | grep -oE '[0-9]{3}' | head -1)"
        else
            CKPT="${CKPT_DIR}/checkpoint_epoch${ep}.pth"
            TAG="epoch${ep}"
        fi
        if [ ! -f "${CKPT}" ]; then
            echo "[WARN] checkpoint 不存在, 跳过: ${CKPT}"
            continue
        fi

        echo ""
        echo "────────────────────────────────────────"
        echo "  测试 ${ds} @ ${TAG}"
        echo "────────────────────────────────────────"
        python -u test_vtuav_promptlora.py \
            --weight "${CKPT}" \
            --model_type enc_promptlora_vtuav \
            --dataset "${ds}" \
            --data_root "${SGTEST_DATA_ROOT}" \
            --save_dir "${SAVE_DIR}/${TAG}" \
            --workers "${WORKERS}"
        echo "[DONE] ${ds} @ ${TAG} 测试完成"
    done
done

echo ""
echo "全部测试完成! 结果: ${SAVE_DIR}/"
