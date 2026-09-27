#!/usr/bin/env python3
"""无需重新推理，按 RGBT toolkit 1.0.1 重算已有 GTOT/RGBT210/RGBT234/LasHeR 结果。"""
from __future__ import annotations

import argparse
import json
import os
import sys

from core.official_rgbt_eval import evaluate_official_rgbt


def main() -> None:
    parser = argparse.ArgumentParser(
        description='对已有预测 txt 结果执行 RGBT toolkit 1.0.1 官方重评，不加载模型、不读取图像。')
    parser.add_argument('--dataset', required=True,
                        choices=['gtot', 'rgbt210', 'rgbt234', 'lasher'])
    parser.add_argument('--result_dir', required=True,
                        help='包含每条序列 <sequence>.txt 预测文件的目录')
    parser.add_argument('--official_toolkit',
                        default=os.environ.get(
                            'RGBT_TOOLKIT_HOME',
                            '/root'),
                        help='RGBT toolkit 1.0.1 根目录')
    parser.add_argument('--tracker_name', default='CIPTrack',
                        help='写入官方汇总文件的 tracker 标识')
    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    if not os.path.isdir(result_dir):
        parser.error(f'结果目录不存在: {result_dir}')

    print('=' * 78)
    print('  RGBT toolkit 1.0.1 离线官方重评')
    print('=' * 78)
    print(f'  数据集     : {args.dataset}')
    print(f'  预测目录   : {result_dir}')
    print(f'  官方评测包 : {args.official_toolkit}')
    print('  模型推理   : 跳过（直接读取已有 .txt 预测）')
    print('=' * 78)

    metrics = evaluate_official_rgbt(
        args.dataset, result_dir, args.official_toolkit, args.tracker_name)

    # 保留已有诊断汇总，仅追加官方评测结论。
    summary_path = os.path.join(result_dir, 'summary.json')
    summary = {}
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f'[WARN] 无法读取已有 summary.json，将新建官方汇总: {exc}')
    summary['official_toolkit'] = {
        'protocol': 'RGBT toolkit 1.0.1',
        'metrics': metrics,
        'metrics_file': 'official_metrics.json',
        'recomputed_without_inference': True,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    text = ' '.join(f'{name}={value:.4f}' for name, value in metrics.items())
    print(f'[OFFICIAL RESULT] RGBT toolkit 1.0.1: {text}')
    print(f'[INFO] 官方汇总 → {os.path.join(result_dir, "official_metrics.json")}')
    print(f'[INFO] 官方 CSV  → {os.path.join(result_dir, "official_metrics.csv")}')
    print(f'[INFO] 总结更新   → {summary_path}')


if __name__ == '__main__':
    main()
