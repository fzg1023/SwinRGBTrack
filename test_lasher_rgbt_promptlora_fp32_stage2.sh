#!/bin/bash
# =============================================================================
# PromptTrack fp32 Stage2 — LasHeR 测试脚本
# =============================================================================
set -eo pipefail
source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
CKPT_DIR="${PROJ_ROOT}/output/enc_promptlora_fp32_stage2"
SAVE_DIR="${CKPT_DIR}/testresults"
LASHER_ROOT="${LASHER_ROOT:-/home/fzg/data/lasher}"
WORKERS="${WORKERS:-4}"

if [ $# -eq 0 ]; then
    EPOCHS=("final")
else
    EPOCHS=()
    for arg in "$@"; do
        if [ "${arg}" = "final" ]; then
            EPOCHS+=("final")
        else
            arg_clean="${arg#epoch}"
            EPOCHS+=("$(printf "%03d" "$((10#$arg_clean))")")
        fi
    done
fi

echo "========================================"
echo "  PromptTrack fp32 Stage2 LasHeR 测试"
echo "========================================"
echo "  Checkpoint 目录: ${CKPT_DIR}"
echo "  Epochs:          ${EPOCHS[*]}"
echo "  目标参考: fp32 Stage1 ep3 seq_PS=0.7263 / seq_AO=0.5902"
echo "  对比参考: AMP 版最优 seq_PS=0.7243"
echo "  判定线:   ≥0.7263=继续涨; 0.724~0.7263=噪声带; <0.724=无增益"
echo "========================================"

mkdir -p "${SAVE_DIR}"

for ep in "${EPOCHS[@]}"; do
    if [ "${ep}" = "final" ]; then
        CKPT="${CKPT_DIR}/final_model.pth"
        TAG="final"
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
    echo "  测试 ${TAG}"
    echo "────────────────────────────────────────"
    cd "${PROJ_ROOT}"
    python -u test_lasher_rgbt_enc_fuse.py \
        --weight "${CKPT}" \
        --model_type enc_promptlora \
        --dataset_root "${LASHER_ROOT}/testingset" \
        --save_dir "${SAVE_DIR}/${TAG}" \
        --workers "${WORKERS}"
    echo "[DONE] ${TAG} 测试完成"
done

echo ""
echo "全部测试完成! 结果: ${SAVE_DIR}/"
