#!/bin/bash
# =============================================================================
# 对已经完成追踪的 GTOT/RGBT210/RGBT234/LasHeR 结果做官方重评（不重新推理）
# =============================================================================
# 用法：每组参数为 <dataset> <预测结果目录>，可连续传入多组。
#
#   bash recompute_official_rgbt_metrics.sh \
#     gtot    output/batch_testresults/checkpoint_epoch005/gtot/swintrack_b384_enc_promptlora_vtuav \
#     rgbt210 output/batch_testresults/checkpoint_epoch005/rgbt210/swintrack_b384_enc_promptlora_vtuav \
#     rgbt234 output/batch_testresults/checkpoint_epoch005/rgbt234/swintrack_b384_enc_promptlora_vtuav \
#     lasher  output/enc_promptlora_vtuav/lasher_testresults/checkpoint_epoch005/lasher/swintrack_b384_enc_promptlora_vtuav
#
# 注意：LasHeR 的官方重评要求结果目录内文件名与 RGBT toolkit 自带
#   lashertest.txt（245 条测试序列）完全一致。
#
# 环境变量：
#   RGBT_TOOLKIT_HOME  官方 rgbt-1.0.1 根目录。
#                      默认 /root（支持评测包源码直接放在根目录）。
#   SUMMARY_FILE       全部重评任务的状态汇总 CSV；默认写到项目 output/。
#
# 每个结果目录会更新/生成：
#   official_metrics.json, official_metrics.csv, summary.json
# =============================================================================
set -o pipefail

if [ $# -eq 0 ] || [ $(( $# % 2 )) -ne 0 ]; then
    echo "用法: bash $0 <gtot|rgbt210|rgbt234|lasher> <result_dir> [<dataset> <result_dir> ...]" >&2
    exit 2
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mambavision_rgbt

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_ROOT"
export RGBT_TOOLKIT_HOME="${RGBT_TOOLKIT_HOME:-/root}"
SUMMARY_FILE="${SUMMARY_FILE:-${PROJ_ROOT}/output/official_recompute_$(date +%Y%m%d_%H%M%S).csv}"
mkdir -p "$(dirname "$SUMMARY_FILE")"
printf 'dataset,result_dir,status,metrics_file\n' > "$SUMMARY_FILE"

failed=0
while [ $# -gt 0 ]; do
    dataset="$1"
    result_dir="$2"
    shift 2

    case "$dataset" in
        gtot|rgbt210|rgbt234|lasher) ;;
        *)
            echo "[ERROR] 不支持的数据集: $dataset" >&2
            printf '%s,%s,invalid_dataset,\n' "$dataset" "$result_dir" >> "$SUMMARY_FILE"
            failed=1
            continue
            ;;
    esac

    echo ""
    echo "────────────────────────────────────────"
    echo "  离线重评 ${dataset}"
    echo "  ${result_dir}"
    echo "────────────────────────────────────────"

    if python -u recompute_official_rgbt_metrics.py \
        --dataset "$dataset" \
        --result_dir "$result_dir" \
        --official_toolkit "$RGBT_TOOLKIT_HOME"; then
        abs_dir="$(cd "$result_dir" && pwd)"
        printf '%s,%s,success,%s\n' "$dataset" "$abs_dir" \
            "$abs_dir/official_metrics.csv" >> "$SUMMARY_FILE"
    else
        printf '%s,%s,failed,\n' "$dataset" "$result_dir" >> "$SUMMARY_FILE"
        failed=1
        echo "[ERROR] ${dataset} 重评失败，继续下一组。" >&2
    fi
done

echo ""
echo "全部离线重评结束。任务汇总: ${SUMMARY_FILE}"
exit "$failed"
