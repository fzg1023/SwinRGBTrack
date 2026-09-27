#!/bin/bash
# =============================================================================
# RGBTSwinTrack RGBT 训练脚本
# 使用 RGBT 配置文件进行训练
#
# 用法:
#   bash train_rgbt.sh <workspace_dir>
#
# 示例:
#   bash train_rgbt.sh /home/fzg/experiments/rgbt_track
#
# 可选参数:
#   --resume <path>           恢复训练
#   --weight_path <path>      加载预训练权重
#   --device_ids <ids>        GPU 设备 ID (如 "0,1,2,3")
#   --workers <num>           DataLoader worker 数量
#   --offline                 WandB 离线模式
#   --evaluation_only         仅评估
# =============================================================================

set -e

# Conda 环境设置
source /home/fzg/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

# 项目根目录
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"

workspace_dir=''
resume_file_path=''
weight_path=''
device_ids=''
num_workers=4
pin_memory=true
wandb_offline=false
evaluation_only=false

method_name="RGBTSwinTrack"
config_name="Base-384"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -R|--resume) resume_file_path="${2}"; shift ;;
        -W|--workers) num_workers=$2; shift ;;
        --output_dir) workspace_dir=$2; shift ;;
        --no_pin_memory) pin_memory=false ;;
        --device_ids) device_ids="$2"; shift ;;
        --offline) wandb_offline=true ;;
        --weight_path) weight_path="$2"; shift ;;
        --evaluation_only) evaluation_only=true ;;
        *) if [[ -z "$workspace_dir" ]]; then
               workspace_dir="$1"
           else
               echo "Unknown parameter passed: $1"; exit 1
           fi ;;
    esac
    shift
done

if [[ -z "$workspace_dir" ]]; then
    echo "Usage: bash train_rgbt.sh <workspace_dir> [options]"
    echo ""
    echo "Options:"
    echo "  --resume <path>            Resume from checkpoint"
    echo "  --weight_path <path>       Load pretrained weights"
    echo "  --device_ids <ids>         GPU device IDs (e.g. '0,1,2,3')"
    echo "  --workers <num>            DataLoader workers (default: 4)"
    echo "  --offline                  WandB offline mode"
    echo "  --evaluation_only          Evaluation only"
    exit 1
fi

set -o pipefail

# ── GPU 检测 ──
if [[ -z "$device_ids" ]]; then
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    nvidia-smi
else
    num_gpus=$(nvidia-smi -i "$device_ids" --query-gpu=name --format=csv,noheader | wc -l)
    nvidia-smi -i "$device_ids"
    export CUDA_VISIBLE_DEVICES="$device_ids"
fi

echo "========================================"
echo "  RGBTSwinTrack RGBT 训练"
echo "========================================"
echo "  项目目录:      ${PROJ_ROOT}"
echo "  工作目录:      ${workspace_dir}"
echo "  方法:          ${method_name}"
echo "  配置:          ${config_name}"
echo "  GPU 数量:      ${num_gpus}"
echo "  Workers:       ${num_workers}"
echo "  Pin Memory:    ${pin_memory}"
echo "  WandB Offline: ${wandb_offline}"
echo "  仅评估:        ${evaluation_only}"
echo "========================================"

# ── 构建运行 ID ──
DATE_WITH_TIME=$(date "+%Y.%m.%d-%H.%M.%S-%6N")
run_id="RGBTSwinTrack-${config_name}-${DATE_WITH_TIME}"

output_dir="$workspace_dir/$run_id"
mkdir -p "$output_dir"

export OMP_NUM_THREADS=1

target_options=("$method_name" "$config_name")

common_options=("--run_id" "$run_id" "--output_dir" "$workspace_dir" "--num_workers" "$num_workers")

if [[ "$pin_memory" == true ]]; then
    common_options+=("--pin_memory")
fi
if [[ "$wandb_offline" == true ]]; then
    common_options+=("--wandb_run_offline")
fi
if [[ "$num_gpus" -gt 1 ]]; then
    common_options+=("--distributed_nproc_per_node" "$num_gpus")
    common_options+=("--distributed_do_spawn_workers")
fi
if [[ -n "$resume_file_path" ]]; then
    common_options+=("--resume" "$resume_file_path")
fi
if [[ -n "$weight_path" ]]; then
    common_options+=("--weight_path" "$weight_path")
fi
if [[ "$evaluation_only" == true ]]; then
    common_options+=("--mixin_config" "${PROJ_ROOT}/config/mixin/evaluation.yaml")
fi

# ── 启动训练 ──
output_log="$output_dir/train_stdout.log"
echo "[INFO] 训练日志: $output_log"

cd "${PROJ_ROOT}"
python main.py "${target_options[@]}" "${common_options[@]}" 2>&1 | tee -a "$output_log"
