#!/bin/bash
# =============================================================================
# EncPromptLoRA fp32 (LasHeR 训练, enc_promptlora, fp32) — 服务器
# VTUAV/GTOT/RGBT210/RGBT234 批量测试
# =============================================================================
# 用途: 对 output/enc_promptlora_fp32 下所有 checkpoint_epoch*.pth
#       (LasHeR 训练的 EncPromptLoRA fp32, model_type=enc_promptlora) 逐个在
#       服务器的 VTUAV / GTOT / RGBT210 / RGBT234 上测试, 并自动用官方 RGBT
#       toolkit 汇总 GTOT/RGBT210/RGBT234 指标。
#
# 用法 (服务器, 在 /root/SwinRGBTrack 下运行):
#   bash test_encpromptlora_fp32_bench.sh
#   bash test_encpromptlora_fp32_bench.sh -s vtuav_st,gtot      # 只测部分数据集
#   bash test_encpromptlora_fp32_bench.sh -e 003 -s "gtot rgbt234"
#
# 参数:
#   -d, --ckpt-dir PATH    checkpoint 目录 (默认 output/enc_promptlora_fp32)
#   -e, --epoch N         只测指定 epoch (可重复); 默认=目录下全部 checkpoint_epoch*.pth
#   -s, --datasets LIST    要测试的数据集子集, 支持三种写法:
#                            逗号分隔:  -s vtuav_st,gtot
#                            空格分隔:  -s "vtuav_st gtot rgbt234"
#                            重复使用:  -s vtuav_st -s gtot
#                          可选值: vtuav_lt | vtuav_st | gtot | rgbt210 | rgbt234
#                          默认 all (5 个全测); 输入非法值会立即报错
#   -w, --workers N        每次测试并行 worker 数, 默认 1 (fp32 显存占用更高)
#   -o, --output-dir PATH  批量结果根目录 (默认 <ckpt-dir>/server_bench_testresults)
#   -m, --model-type NAME  默认 enc_promptlora (fp32 训练, 结构同 enc_promptlora)
#   -h, --help
#
# 环境变量:
#   SGTEST_DATA_ROOT=/root/RGBTData   服务器 RGBT 数据根目录
#   RGBT_TOOLKIT_HOME=/root           官方 RGBT toolkit 目录
#   SKIP_EXISTING=0                   强制全量重测 (默认=1: 已存在完整 .txt
#                                     结果的序列直接读取显示, 不重新推理)
#
# 输出结构:
#   <output-dir>/<checkpoint名>/<dataset>/swintrack_b384_enc_promptlora/
#       per_seq_metrics.csv  summary.json  official_metrics.* (gtot/rgbt210/rgbt234)
#   <output-dir>/batch_summary.csv
# =============================================================================
set -o pipefail

usage() { sed -n '3,40p' "$0"; }

CKPT_DIR="${CKPT_DIR:-output/enc_promptlora_fp32}"
EPOCHS=()
DS_ARGS=()
if [ -n "${DATASETS:-}" ]; then
    # 兼容旧的环境变量逗号写法
    IFS=',' read -r -a DS_ARGS <<< "$DATASETS"
fi
WORKERS="${WORKERS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MODEL_TYPE="${MODEL_TYPE:-enc_promptlora}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

while [ $# -gt 0 ]; do
    case "$1" in
        -d|--ckpt-dir) CKPT_DIR="$2"; shift 2 ;;
        -e|--epoch)
            [ $# -ge 2 ] || { echo "[ERROR] $1 缺少 epoch" >&2; exit 2; }
            EPOCHS+=("$2"); shift 2 ;;
        -s|--datasets)
            [ $# -ge 2 ] || { echo "[ERROR] $1 缺少数据集列表" >&2; exit 2; }
            DS_ARGS+=("$2"); shift 2 ;;
        -w|--workers)
            [ $# -ge 2 ] || { echo "[ERROR] $1 缺少 worker 数" >&2; exit 2; }
            WORKERS="$2"; shift 2 ;;
        -o|--output-dir)
            [ $# -ge 2 ] || { echo "[ERROR] $1 缺少输出目录" >&2; exit 2; }
            OUTPUT_DIR="$2"; shift 2 ;;
        -m|--model-type)
            [ $# -ge 2 ] || { echo "[ERROR] $1 缺少模型类型" >&2; exit 2; }
            MODEL_TYPE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] 未知参数: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${CKPT_DIR}/server_bench_testresults"
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_ROOT"

CKPT_DIR_ABS="$(cd "$CKPT_DIR" && pwd)"
SGTEST_DATA_ROOT="${SGTEST_DATA_ROOT:-/root/RGBTData}"
export RGBT_TOOLKIT_HOME="${RGBT_TOOLKIT_HOME:-/root}"

# ── 解析数据集列表 ──
ALL_DS=(vtuav_lt vtuav_st gtot rgbt210 rgbt234)
DS_LIST=()
if [ ${#DS_ARGS[@]} -eq 0 ]; then
    DS_LIST=("${ALL_DS[@]}")
else
    # 每个 -s 值可能同时含逗号和空格: 先全部压扁成一个数组再统一展开
    RAW=()
    for arg in "${DS_ARGS[@]}"; do
        # 把逗号统一替换为空格再按空白切分
        arg_sp="${arg//,/ }"
        # shellcheck disable=SC2206
        RAW+=($arg_sp)
    done
    # 去重
    for ds in "${RAW[@]}"; do
        found=0
        for prev in "${DS_LIST[@]}"; do
            [ "$prev" = "$ds" ] && found=1 && break
        done
        [ "$found" -eq 0 ] && DS_LIST+=("$ds")
    done
fi

# ── 校验数据集名, 拼错立即报错 ──
for ds in "${DS_LIST[@]}"; do
    ok=0
    for a in "${ALL_DS[@]}"; do
        [ "$ds" = "$a" ] && ok=1 && break
    done
    if [ "$ok" -eq 0 ]; then
        echo "[ERROR] 未知数据集: ${ds} (可选: ${ALL_DS[*]})" >&2
        exit 2
    fi
done

# ── 解析要测试的 checkpoint ──
if [ ${#EPOCHS[@]} -eq 0 ]; then
    CKPTS=()
    while IFS= read -r f; do CKPTS+=("$f"); done < \
        <(ls -1 "${CKPT_DIR_ABS}"/checkpoint_epoch*.pth 2>/dev/null | sort -V)
else
    CKPTS=()
    for ep in "${EPOCHS[@]}"; do
        ep="${ep#epoch}"
        CKPTS+=("${CKPT_DIR_ABS}/checkpoint_epoch$(printf '%03d' "$((10#$ep))").pth")
    done
fi

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "[ERROR] 目录下没有 checkpoint_epoch*.pth: ${CKPT_DIR_ABS}" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
SUMMARY_CSV="$OUTPUT_DIR/batch_summary.csv"
if [ ! -f "$SUMMARY_CSV" ]; then
    printf 'checkpoint,dataset,n_valid,n_total,mean_fps,seq_AO,seq_SS,seq_SR50,seq_SR75,seq_PS,seq_NPS,status,result_dir\n' > "$SUMMARY_CSV"
fi

echo "=========================================================="
echo "  EncPromptLoRA fp32 (enc_promptlora) 服务器评测"
echo "=========================================================="
echo "  Checkpoint 目录: ${CKPT_DIR_ABS}"
echo "  Checkpoints   : ${#CKPTS[@]}"
echo "  Datasets      : ${DS_LIST[*]}"
echo "  数据根目录    : ${SGTEST_DATA_ROOT}"
echo "  官方评测包    : ${RGBT_TOOLKIT_HOME}"
echo "  模型类型      : ${MODEL_TYPE}"
echo "  输出目录      : ${OUTPUT_DIR}"
echo "  Workers       : ${WORKERS}"
echo "=========================================================="

failed=0
for ckpt in "${CKPTS[@]}"; do
    [ -f "$ckpt" ] || { echo "[WARN] 跳过不存在: $ckpt"; continue; }
    ckpt_name="$(basename "$ckpt" .pth)"
    for ds in "${DS_LIST[@]}"; do
        ckpt_out="$OUTPUT_DIR/$ckpt_name"
        mkdir -p "$ckpt_out"
        echo ""
        echo "────────────────────────────────────────"
        echo "  测试 ${ds} @ ${ckpt_name}"
        echo "────────────────────────────────────────"

        SKIP_ARGS=()
        if [ "${SKIP_EXISTING}" = "1" ]; then
            SKIP_ARGS=(--skip_existing)
            echo "  [模式] 复用已有 .txt 结果 (SKIP_EXISTING=1); 强制全量: SKIP_EXISTING=0"
        fi

        if python -u test_vtuav_promptlora.py \
            --weight "$ckpt" \
            --model_type "$MODEL_TYPE" \
            --dataset "$ds" \
            --data_root "$SGTEST_DATA_ROOT" \
            --save_dir "$ckpt_out" \
            --workers "$WORKERS" \
            "${SKIP_ARGS[@]}"; then
            result_dir="$ckpt_out/$ds/swintrack_b384_enc_promptlora"
            # 尝试提取 summary.json 关键指标写入批量汇总
            summary=""
            if [ -f "$result_dir/summary.json" ]; then
                summary=$(python - "$result_dir/summary.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    s = json.load(f)
sm = s.get('seq_means', {})
def g(k, d=0.0):
    v = sm.get(k, d)
    return f'{v:.4f}' if isinstance(v, (int, float)) else f'{d:.4f}'
print(f"{s.get('n_sequences',0)},{s.get('n_valid',0)},{s.get('mean_fps',0):.2f},{g('AO')},{g('SS')},{g('SR50')},{g('SR75')},{g('PS')},{g('NPS')}")
PY
)
            fi
            if [ -n "$summary" ]; then
                printf '%s,%s,%s,success,%s\n' "$ckpt_name" "$ds" "$summary" "$result_dir" >> "$SUMMARY_CSV"
            else
                printf '%s,%s,0,0,0,0,0,0,0,0,0,no_summary,%s\n' "$ckpt_name" "$ds" "$result_dir" >> "$SUMMARY_CSV"
            fi
            echo "[DONE] ${ds} @ ${ckpt_name} → ${result_dir}"
        else
            printf '%s,%s,0,0,0,0,0,0,0,0,0,failed,%s\n' "$ckpt_name" "$ds" "$ckpt_out/$ds" >> "$SUMMARY_CSV"
            echo "[ERROR] ${ds} @ ${ckpt_name} 失败, 继续下一项" >&2
            failed=1
        fi
    done
done

echo ""
echo "=========================================================="
echo "批量测试结束。汇总: $SUMMARY_CSV"
echo "=========================================================="
exit "$failed"
