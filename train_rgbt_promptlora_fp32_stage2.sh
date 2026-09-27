#!/bin/bash
# =============================================================================
# PromptTrack Stage2 fp32 (AMP=False) — 降学习率安全续训
# =============================================================================
# 热启动: Stage1 fp32 的 ep3 (seq_PS=0.7263, 已超越 AMP 版 0.7243)。
# 注意: Stage1 ep3 时 backbone 曾解冻 (1e-5), 本阶段重新全程冻结,
#       延续"安全区间"动力学再训 2 轮 (lr 减半至 5e-5)。
# 止损: PS < 0.724 判定无增益。
# 输出: output/enc_promptlora_fp32_stage2/
# =============================================================================
set -eo pipefail
source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJ_ROOT}"

WEIGHT="${PROJ_ROOT}/output/enc_promptlora_fp32/checkpoint_epoch003.pth"
OUTPUT_DIR="${PROJ_ROOT}/output/enc_promptlora_fp32_stage2"
EPOCHS=2
LR=5e-5

if [ ! -f "${WEIGHT}" ]; then
    echo "[ERROR] 热启动权重不存在: ${WEIGHT} (先跑 stage1)"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/train_log_$(date +%Y%m%d_%H%M%S).txt"

echo "========================================"
echo "  PromptTrack Stage2 fp32 训练"
echo "========================================"
echo "  热启动:   ${WEIGHT} (Stage1 fp32 ep3, PS=0.7263)"
echo "  输出:     ${OUTPUT_DIR}"
echo "  训练:     prompt/LoRA + head, LR=${LR} 恒定"
echo "  冻结:     backbone+stem+fusion 全程冻结 (重新冻结)"
echo "  Epochs:   ${EPOCHS} | AMP=False (fp32)"
echo "  目标:     超过 0.7263; 止损: PS < 0.724"
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
    --freeze_backbone_epochs 100 \
    --freeze_stem_epochs 100 \
    --freeze_fusion_epochs 100 \
    --const_lr \
    --warmup_epochs 0 \
    --workers 4 \
    --seed 42 \
    --no_amp \
    --no_auto_test \
    2>&1 | tee "${LOG}"
