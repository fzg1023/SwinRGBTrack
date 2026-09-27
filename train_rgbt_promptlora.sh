#!/bin/bash
# =============================================================================
# EncPromptLoRA — Encoder 内跨模态 Prompt 注入 + TIR 专属 LoRA (Pilot)
# =============================================================================
# 背景: 空间/时序 mamba (数百万新增参数, 加在 encoder 之后) 均因 978 序列小
#       数据过拟合而失败。本方案改为在 encoder 8 层内部逐层做跨模态早融合,
#       新增参数量小一个数量级 (~450K, PromptGen + LoRA)。
# 热启动: output/enc_fusemamba/checkpoint_epoch001.pth (PS=0.7117 基准)。
#       PromptGen.up / LoRALinear.lora_B 均零初始化, 初始严格等价该 checkpoint。
# 冻结:  backbone + 原始 encoder 8 层 + decoder + 已训练的 mamba 融合残差 (全程)
# 训练:  16 个 PromptGen + 24 个 LoRALinear (q/k/v x 8 层) + head, LR=1e-4 恒定
# 输出:  output/enc_promptlora/
# =============================================================================
set -eo pipefail
source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJ_ROOT}"

WEIGHT="${PROJ_ROOT}/output/enc_fusemamba/checkpoint_epoch001.pth"
OUTPUT_DIR="${PROJ_ROOT}/output/enc_promptlora"
EPOCHS=3
LR=1e-4

if [ ! -f "${WEIGHT}" ]; then
    echo "[ERROR] 热启动权重不存在: ${WEIGHT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/train_log_$(date +%Y%m%d_%H%M%S).txt"

echo "========================================"
echo "  EncPromptLoRA Pilot 训练"
echo "========================================"
echo "  热启动:   ${WEIGHT}"
echo "  输出:     ${OUTPUT_DIR}"
echo "  冻结:     backbone+encoder(原始权重)+decoder+mamba融合 (全程)"
echo "  训练:     PromptGen(16) + LoRA(24) + head, LR=${LR} 恒定"
echo "  Epochs:   ${EPOCHS}"
echo "  基准参考: FuseMamba ep1 PS=0.7117 / AO=0.5797"
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
    --amp \
    --no_auto_test \
    2>&1 | tee "${LOG}"
