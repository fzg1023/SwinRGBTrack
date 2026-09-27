#!/bin/bash
# =============================================================================
# CIPTrack / EncPromptLoRA — 多 checkpoint、多数据集批量测试（服务器）
# =============================================================================
# 用法：
#   bash test_vtuav_multi_checkpoints.sh \
#     -c output/enc_promptlora_vtuav/checkpoint_epoch005.pth \
#     -c output/enc_promptlora_vtuav/checkpoint_epoch010.pth \
#     -d vtuav_st -d gtot -d rgbt234 -w 2
#
# 参数：
#   -c, --checkpoint PATH   checkpoint 路径；可重复指定，至少一次
#   -d, --dataset NAME      数据集；可重复指定。支持：
#                             vtuav_st | vtuav_lt | gtot | rgbt210 | rgbt234 | all
#                             未指定时使用 vtuav_st
#   -w, --workers N         每次测试的并行 worker 数，默认 4
#   -o, --output-dir PATH   批量结果根目录，默认 <第一个checkpoint目录>/batch_testresults
#   -h, --help              显示帮助
#
# 环境变量：
#   SGTEST_DATA_ROOT=/root/RGBTData   数据集根目录
#   MODEL_TYPE=enc_promptlora_vtuav   模型类型（默认值）
#   RGBT_TOOLKIT_HOME=/path/to/rgbt-1.0.1
#                                     GTOT/RGBT210/RGBT234 官方评测包目录
#
# 输出结构：
#   <output-dir>/<checkpoint文件名>/<dataset>/swintrack_b384_enc_promptlora_vtuav/
#   <output-dir>/batch_summary.csv
#
# 注意：每个 worker 都会独立加载完整模型。24GB GPU 建议 workers=1 或 2；
# 多 checkpoint 按顺序执行，避免多个模型组同时占用显存。
# =============================================================================
set -o pipefail

usage() {
    sed -n '3,32p' "$0"
}

CHECKPOINTS=()
DATASETS=()
WORKERS="${WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        -c|--checkpoint)
            [ $# -ge 2 ] || { echo "[ERROR] $1 缺少 checkpoint 路径" >&2; exit 2; }
            CHECKPOINTS+=("$2")
            shift 2
            ;;
        -d|--dataset)
            [ $# -ge 2 ] || { echo "[ERROR] $1 缺少数据集名称" >&2; exit 2; }
            DATASETS+=("$2")
            shift 2
            ;;
        -w|--workers)
            [ $# -ge 2 ] || { echo "[ERROR] $1 缺少 worker 数" >&2; exit 2; }
            WORKERS="$2"
            shift 2
            ;;
        -o|--output-dir)
            [ $# -ge 2 ] || { echo "[ERROR] $1 缺少输出目录" >&2; exit 2; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] 未知参数: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "[ERROR] 至少要用 -c/--checkpoint 指定一个 checkpoint。" >&2
    usage >&2
    exit 2
fi

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] workers 必须是正整数，当前值: $WORKERS" >&2
    exit 2
fi

if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("vtuav_st")
fi

# 展开 all、校验名称，并去重以避免重复的长时间测试。
VALID_DATASETS=("vtuav_st" "vtuav_lt" "gtot" "rgbt210" "rgbt234")
EXPANDED_DATASETS=()
for ds in "${DATASETS[@]}"; do
    if [ "$ds" = "all" ]; then
        EXPANDED_DATASETS=("${VALID_DATASETS[@]}")
        break
    fi
    is_valid=0
    for valid in "${VALID_DATASETS[@]}"; do
        [ "$ds" = "$valid" ] && is_valid=1 && break
    done
    if [ "$is_valid" -ne 1 ]; then
        echo "[ERROR] 不支持的数据集: $ds" >&2
        echo "        可选: ${VALID_DATASETS[*]} 或 all" >&2
        exit 2
    fi
    already_added=0
    for added in "${EXPANDED_DATASETS[@]}"; do
        [ "$ds" = "$added" ] && already_added=1 && break
    done
    [ "$already_added" -eq 1 ] || EXPANDED_DATASETS+=("$ds")
done
DATASETS=("${EXPANDED_DATASETS[@]}")

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_ROOT"
SGTEST_DATA_ROOT="${SGTEST_DATA_ROOT:-/root/RGBTData}"
export RGBT_TOOLKIT_HOME="${RGBT_TOOLKIT_HOME:-/root}"
MODEL_TYPE="${MODEL_TYPE:-enc_promptlora_vtuav}"

# 在切换目录前解析 checkpoint，保证相对路径也可靠。
CHECKPOINT_ABS=()
CHECKPOINT_TAGS=()
for checkpoint in "${CHECKPOINTS[@]}"; do
    if [ ! -f "$checkpoint" ]; then
        echo "[ERROR] checkpoint 不存在: $checkpoint" >&2
        exit 1
    fi
    abs_path="$(cd "$(dirname "$checkpoint")" && pwd)/$(basename "$checkpoint")"
    tag="$(basename "$checkpoint" .pth)"
    for old_tag in "${CHECKPOINT_TAGS[@]}"; do
        if [ "$tag" = "$old_tag" ]; then
            echo "[ERROR] 存在同名 checkpoint：$tag。请用 -o 分开运行，或先重命名文件。" >&2
            exit 2
        fi
    done
    CHECKPOINT_ABS+=("$abs_path")
    CHECKPOINT_TAGS+=("$tag")
done

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$(dirname "${CHECKPOINT_ABS[0]}")/batch_testresults"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

SUMMARY_CSV="$OUTPUT_DIR/batch_summary.csv"
if [ ! -f "$SUMMARY_CSV" ]; then
    printf 'checkpoint,dataset,status,result_dir\n' > "$SUMMARY_CSV"
fi

printf '\n========================================\n'
printf '  CIPTrack 多 checkpoint 批量测试\n'
printf '========================================\n'
printf '  Checkpoints : %s\n' "${#CHECKPOINT_ABS[@]}"
printf '  Datasets    : %s\n' "${DATASETS[*]}"
printf '  数据根目录  : %s\n' "$SGTEST_DATA_ROOT"
printf '  官方评测包  : %s\n' "$RGBT_TOOLKIT_HOME"
printf '  输出根目录  : %s\n' "$OUTPUT_DIR"
printf '  Workers     : %s\n' "$WORKERS"
printf '========================================\n'

failed=0
for i in "${!CHECKPOINT_ABS[@]}"; do
    checkpoint="${CHECKPOINT_ABS[$i]}"
    tag="${CHECKPOINT_TAGS[$i]}"
    checkpoint_dir="$OUTPUT_DIR/$tag"
    mkdir -p "$checkpoint_dir"

    for ds in "${DATASETS[@]}"; do
        printf '\n────────────────────────────────────────\n'
        printf '  测试 %s @ %s\n' "$ds" "$tag"
        printf '────────────────────────────────────────\n'

        if python -u test_vtuav_promptlora.py \
            --weight "$checkpoint" \
            --model_type "$MODEL_TYPE" \
            --dataset "$ds" \
            --data_root "$SGTEST_DATA_ROOT" \
            --save_dir "$checkpoint_dir" \
            --workers "$WORKERS"; then
            printf '%s,%s,success,%s\n' "$tag" "$ds" "$checkpoint_dir/$ds" >> "$SUMMARY_CSV"
            echo "[DONE] $ds @ $tag"
        else
            printf '%s,%s,failed,%s\n' "$tag" "$ds" "$checkpoint_dir/$ds" >> "$SUMMARY_CSV"
            echo "[ERROR] $ds @ $tag 失败，继续下一项。" >&2
            failed=1
        fi
    done
done

printf '\n========================================\n'
printf '批量测试结束。汇总: %s\n' "$SUMMARY_CSV"
printf '========================================\n'

exit "$failed"
