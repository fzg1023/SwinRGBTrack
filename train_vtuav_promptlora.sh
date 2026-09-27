#!/bin/bash
# =============================================================================
# RGBTSwinTrack-EncPromptLoRA — VTUAV 从头训练脚本 (服务器)
# =============================================================================
# 配置: config/SwinRGBTrack/Base-384-enc-promptlora-vtuav
# 数据 (与 SGTrack 服务器目录组织一致, 可用环境变量覆盖):
#   VTUAV_HOME      = /root/RGBTData/VTUAV   (train / test_ST / test_LT)
#   SGTEST_DATA_ROOT= /root/RGBTData         (auto_test 用数据根)
# 从头训练: backbone 用 SwinTrack-B-384.pth (ImageNet 预训练) 初始化,
#           encoder/decoder/head/prompt/LoRA 随机初始化。
# 每 5 轮自动对 vtuav_st 全量测试。
#
# 用法:
#   bash train_vtuav_promptlora.sh
#   VTUAV_HOME=/data/VTUAV bash train_vtuav_promptlora.sh
# =============================================================================
set -eo pipefail

# 服务器环境: mambavision_rgbt
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJ_ROOT}"

# ── 参数 (可环境变量覆盖) ──
WEIGHT="${WEIGHT:-${PROJ_ROOT}/SwinTrack-B-384.pth}"
VTUAV_HOME="${VTUAV_HOME:-/root/RGBTData/VTUAV}"
SGTEST_DATA_ROOT="${SGTEST_DATA_ROOT:-/root/RGBTData}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/output/enc_promptlora_vtuav}"
# 显存说明: 前 3 轮 backbone 冻结; 解冻后需额外保存 backbone 激活+梯度+AdamW
# state (~3-4GB @24GB 卡)。batch 8 对齐 SGTrack VTUAV 配置, 解冻后约 80% 显存。
# 24GB 卡可 BATCH_SIZE=12 尝试, 48GB 卡可 BATCH_SIZE=16。
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-50}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-20000}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
FREEZE_BB_EPOCHS="${FREEZE_BB_EPOCHS:-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"
TEST_INTERVAL="${TEST_INTERVAL:-5}"
# 每 N 轮自动测试默认关闭 (vtuav_st 全量 176 序列易超时)。
# 需要时 AUTO_TEST=1 开启, 训练完成后可用 test_vtuav_checkpoint.sh 单独测。
AUTO_TEST="${AUTO_TEST:-0}"
RESUME="${RESUME:-}"

if [ ! -f "${WEIGHT}" ]; then
    echo "[ERROR] 预训练权重不存在: ${WEIGHT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/train_log_$(date +%Y%m%d_%H%M%S).txt"

echo "========================================"
echo "  EncPromptLoRA VTUAV 从头训练"
echo "========================================"
echo "  配置:       config/SwinRGBTrack/Base-384-enc-promptlora-vtuav"
echo "  权重:       ${WEIGHT}"
echo "  训练数据:   ${VTUAV_HOME}/train"
echo "  验证数据:   ${VTUAV_HOME}/test_ST (每轮 val)"
echo "  自动测试:   $([ "${AUTO_TEST}" = "1" ] && echo "vtuav_st (每 ${TEST_INTERVAL} 轮)" || echo "关闭 (训练后用 test_vtuav_checkpoint.sh 单测)")"
echo "  输出目录:   ${OUTPUT_DIR}"
echo "  Batch:      ${BATCH_SIZE}"
echo "  Epochs:     ${EPOCHS}"
echo "  LR:         ${LR}  Backbone LR: ${BACKBONE_LR}"
echo "  Workers:    ${WORKERS}"
echo "========================================"

EXTRA_ARGS=()
[ -n "${RESUME}" ] && EXTRA_ARGS+=(--resume "${RESUME}")
[ "${AUTO_TEST}" = "1" ] || EXTRA_ARGS+=(--no_auto_test)

python -u train_vtuav_promptlora.py \
    --weight "${WEIGHT}" \
    --output_dir "${OUTPUT_DIR}" \
    --vtuav_root "${VTUAV_HOME}" \
    --data_root "${SGTEST_DATA_ROOT}" \
    --model_type enc_promptlora_vtuav \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --samples_per_epoch ${SAMPLES_PER_EPOCH} \
    --lr ${LR} \
    --backbone_lr ${BACKBONE_LR} \
    --freeze_backbone_epochs ${FREEZE_BB_EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --workers ${WORKERS} \
    --seed ${SEED} \
    --test_interval ${TEST_INTERVAL} \
    --amp \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    2>&1 | tee "${LOG}"

echo ""
echo "训练完成! 日志: ${LOG}"
