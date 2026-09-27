#!/bin/bash
# =============================================================================
# CIPTrack / EncPromptLoRA (VTUAV 权重) — 在 LasHeR 上批量测试多个 checkpoint
# =============================================================================
# 用途：用 VTUAV 数据集训练的 enc_promptlora_vtuav checkpoint，在 LasHeR
#       测试集上推理并计算指标 (AO/SS/SR50/SR75/PS/NPS)，支持同时指定多个
#       checkpoint，每个 checkpoint 独立输出，最后汇总。
#
# 用法：
#   bash test_lasher_vtuav_checkpoints.sh \
#     -c output/enc_promptlora_vtuav/checkpoint_epoch005.pth \
#     -c output/enc_promptlora_vtuav/checkpoint_epoch010.pth \
#     -w 2
#
# 参数：
#   -c, --checkpoint PATH   checkpoint 路径；可重复指定，至少一次
#   -d, --data-root PATH    LasHeR 数据根目录，默认 /root/RGBTData/LasHeR
#                           脚本会自动在该目录下寻找 testingset
#   -w, --workers N         并行 worker 数，默认 4
#   -o, --output-dir PATH   批量结果根目录
#                           默认 <第一个checkpoint所在目录>/lasher_testresults
#   -m, --model-type NAME   模型类型，默认 enc_promptlora_vtuav
#   -h, --help              显示帮助
#
# 环境变量：
#   LASHER_ROOT=...         等价于 -d
#   SGTEST... 不适用（LasHeR 是 RGBT 内部评测，不走官方 RGBT toolkit）
#
# 输出结构：
#   <output-dir>/<checkpoint文件名>/lasher/swintrack_b384_enc_promptlora_vtuav/
#       per_seq_metrics.csv  summary.json  （每个序列的 .txt 预测文件）
#   <output-dir>/batch_summary.csv
#
# 说明：LasHeR 测试集 245 条序列，单卡多 worker 会各自加载完整模型，
# 24GB GPU 建议 workers=1~2；多 checkpoint 顺序执行以免显存超限。
# =============================================================================
set -o pipefail

usage() {
    sed -n '3,38p' "$0"
}

CHECKPOINTS=()
WORKERS="${WORKERS:-4}"
DATA_ROOT="${LASHER_ROOT:-/root/RGBTData/LasHeR}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MODEL_TYPE="${MODEL_TYPE:-enc_promptlora_vtuav}"

while [ $# -gt 0 ]; do
    case "$1" in
        -c|--checkpoint)
            [ $# -ge 2 ] || { echo "[ERROR] $1 缺少 checkpoint 路径" >&2; exit 2; }
            CHECKPOINTS+=("$2")
            shift 2
            ;;
        -d|--data-root)
            [ $# -ge 2 ] || { echo "[ERROR] $1 缺少数据根目录" >&2; exit 2; }
            DATA_ROOT="$2"
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
        -m|--model-type)
            [ $# -ge 2 ] || { echo "[ERROR] $1 缺少模型类型" >&2; exit 2; }
            MODEL_TYPE="$2"
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

# ── 定位 LasHeR 测试集目录（兼容多种布局） ──
#   * <root>/test         官方 LasHeR 布局 (train/ + test/)  —— 服务器实测
#   * <root>/testingset   老式布局
#   * <root>              本身就是测试序列集目录
if [ -d "${DATA_ROOT}/test" ]; then
    LASHER_TEST="${DATA_ROOT}/test"
elif [ -d "${DATA_ROOT}/testingset" ]; then
    LASHER_TEST="${DATA_ROOT}/testingset"
elif [ -d "${DATA_ROOT}" ]; then
    LASHER_TEST="${DATA_ROOT}"
else
    echo "[ERROR] 未找到 LasHeR 测试集目录: ${DATA_ROOT} (期望存在 test 或 testingset 子目录)" >&2
    exit 1
fi
if [ ! -d "${LASHER_TEST}" ]; then
    echo "[ERROR] LasHeR 测试目录无效: ${LASHER_TEST}" >&2
    exit 1
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_ROOT"

# 解析绝对路径（切目录前）
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
    OUTPUT_DIR="$(dirname "${CHECKPOINT_ABS[0]}")/lasher_testresults"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

SUMMARY_CSV="$OUTPUT_DIR/batch_summary.csv"
if [ ! -f "$SUMMARY_CSV" ]; then
    printf 'checkpoint,dataset,n_valid,n_total,mean_fps,seq_AO,seq_SS,seq_SR50,seq_SR75,seq_PS,seq_NPS,status,result_dir\n' > "$SUMMARY_CSV"
fi

printf '\n========================================\n'
printf '  VTUAV 权重 → LasHeR 批量测试\n'
printf '========================================\n'
printf '  Checkpoints : %s\n' "${#CHECKPOINT_ABS[@]}"
printf '  LasHeR 测试 : %s\n' "$LASHER_TEST"
printf '  输出根目录  : %s\n' "$OUTPUT_DIR"
printf '  Workers     : %s\n' "$WORKERS"
printf '========================================\n'

failed=0
for i in "${!CHECKPOINT_ABS[@]}"; do
    checkpoint="${CHECKPOINT_ABS[$i]}"
    tag="${CHECKPOINT_TAGS[$i]}"
    ckpt_out_dir="$OUTPUT_DIR/$tag"
    mkdir -p "$ckpt_out_dir"

    printf '\n────────────────────────────────────────\n'
    printf '  LasHeR 测试 %s\n' "$tag"
    printf '────────────────────────────────────────\n'

    # 单条推理错误不应中断整批；由 python 返回码决定本 checkpoint 状态。
    if python -u test_lasher_rgbt_enc_fuse.py \
        --weight "$checkpoint" \
        --model_type "$MODEL_TYPE" \
        --dataset_root "$LASHER_TEST" \
        --save_dir "$ckpt_out_dir" \
        --workers "$WORKERS"; then
        # 汇总结果目录名（与脚本内 save_name 保持一致）
        RESULT_DIR="$ckpt_out_dir/lasher/swintrack_b384_enc_promptlora_vtuav"
        if [ -f "$RESULT_DIR/summary.json" ]; then
            # 从 summary.json 提取关键指标写入汇总 CSV
            summary=$(python - "$RESULT_DIR/summary.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    s = json.load(f)
sm = s.get('seq_means', {})
def g(k, d=0.0):
    v = sm.get(k, d)
    return f'{v:.4f}' if isinstance(v, (int, float)) else f'{d:.4f}'
print(f"{s.get('n_sequences', 0)},{s.get('n_valid', 0)},{s.get('mean_fps', 0):.2f},{g('AO')},{g('SS')},{g('SR50')},{g('SR75')},{g('PS')},{g('NPS')}")
PY
)
            printf '%s,lasher,%s,success,%s\n' "$tag" "$summary" "$RESULT_DIR" >> "$SUMMARY_CSV"
            echo "[DONE] $tag → $RESULT_DIR"
        else
            printf '%s,lasher,0,0,0,0,0,0,0,0,0,no_summary,%s\n' "$tag" "$RESULT_DIR" >> "$SUMMARY_CSV"
            echo "[WARN] $tag 未生成 summary.json" >&2
        fi
    else
        printf '%s,lasher,0,0,0,0,0,0,0,0,0,failed,%s\n' "$tag" "$ckpt_out_dir/lasher" >> "$SUMMARY_CSV"
        echo "[ERROR] $tag LasHeR 测试失败，继续下一项。" >&2
        failed=1
    fi
done

printf '\n========================================\n'
printf '批量测试结束。汇总: %s\n' "$SUMMARY_CSV"
printf '========================================\n'

exit "$failed"
