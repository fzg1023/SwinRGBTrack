#!/bin/bash
# =============================================================================
# PromptTrack Stage1 fp32 (AMP=False) — Encoder 内跨模态 Prompt + LoRA
# =============================================================================
# 与 AMP 版 Stage1 同构, 两个区别:
#   1. --no_amp (fp32)
#   2. 热启动: enc_fusemamba_fp32 ep2 (PS≈0.715, 死链 bias 机制, fp32 训练)
# 链条: enc_fuse_v1_noamp ep8 (0.7143) → enc_fusemamba_fp32 ep2 → 本阶段。
# 融合层 (in_proj=0 死链 + 已学 bias) 在本阶段全程冻结, bias 常数被保留。
# 输出: output/enc_promptlora_fp32/
# =============================================================================
set -eo pipefail
source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJ_ROOT}"

WEIGHT="${PROJ_ROOT}/output/enc_fusemamba_fp32/checkpoint_epoch002.pth"
OUTPUT_DIR="${PROJ_ROOT}/output/enc_promptlora_fp32"
EPOCHS=3
LR=1e-4

if [ ! -f "${WEIGHT}" ]; then
    echo "[ERROR] 热启动权重不存在: ${WEIGHT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/train_log_$(date +%Y%m%d_%H%M%S).txt"

echo "========================================"
echo "  PromptTrack Stage1 fp32 训练"
echo "========================================"
echo "  热启动:   ${WEIGHT} (fp32 enc_fusemamba ep2, PS≈0.715)"
echo "  输出:     ${OUTPUT_DIR}"
echo "  训练:     PromptGen(16) + LoRA(24) + head, LR=${LR} 恒定"
echo "  冻结:     backbone ep1-2, ep3 起 1e-5; stem/fusion 全程冻结"
echo "  Epochs:   ${EPOCHS} | AMP=False (fp32)"
echo "========================================"

python -u train_rgbt_enc_fuse_v1.py \
    --model_type enc_promptlora \
    --weight "${WEIGHT}" \
    --output_dir "${OUTPUT_DIR}" \
    --lasher_root /home/fzg/data/lasher \
    --batch_size 16 \
    --epochs ${EPOCHS} \
    --samples_per_epoch 60000 \
    --lr ${LR} \
    --backbone_lr 1e-5 \
    --freeze_backbone_epochs 2 \
    --freeze_stem_epochs 100 \
    --freeze_fusion_epochs 100 \
    --const_lr \
    --warmup_epochs 0 \
    --workers 4 \
    --seed 42 \
    --no_amp \
    --no_auto_test \
    2>&1 | tee "${LOG}"
