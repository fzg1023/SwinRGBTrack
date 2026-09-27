"""调用 RGBT toolkit 1.0.1 计算 GTOT、RGBT210、RGBT234、LasHeR 的官方指标。"""
from __future__ import annotations

import csv
import importlib
import json
import os
import shutil
import sys
import tempfile
from typing import Dict, List

_SUPPORTED = {'gtot', 'rgbt210', 'rgbt234', 'lasher'}


def _toolkit_candidates(toolkit_root: str):
    """返回用户路径及服务器/本地常见的 toolkit 根目录候选项。"""
    candidates = [
        toolkit_root,
        os.environ.get('RGBT_TOOLKIT_HOME', ''),
        '/root',
        '/root/rgbt-1.0.1/rgbt-1.0.1',
        '/root/rgbt-1.0.1',
        '/home/fzg/rgbt-1.0.1/rgbt-1.0.1',
        '/home/fzg/rgbt-1.0.1',
    ]
    seen = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate) if candidate else ''
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate


def _load_toolkit(toolkit_root: str):
    """从指定源码目录加载官方 RGBT toolkit，而不依赖全局 pip 安装。"""
    resolved_root = next((candidate for candidate in _toolkit_candidates(toolkit_root)
                          if os.path.isfile(os.path.join(candidate, 'src', 'rgbt',
                                                         '__init__.py'))), None)
    if resolved_root is None:
        expected = os.path.join(os.path.abspath(toolkit_root), 'src', 'rgbt',
                                '__init__.py')
        raise FileNotFoundError(
            f'未找到 RGBT toolkit 1.0.1: {toolkit_root} '
            f'(期望存在 {expected}；也已检查 /root/rgbt-1.0.1 和 '
            '/home/fzg/rgbt-1.0.1 的常见目录)')
    src_dir = os.path.join(resolved_root, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return importlib.import_module('rgbt'), resolved_root


def _check_result_files(dataset_obj, result_dir: str, prefix: str = '') -> None:
    """在调用工具包前给出缺失结果文件的可读错误。"""
    missing = [
        f'{prefix}{seq_name}.txt' for seq_name in dataset_obj.seqs_name
        if not os.path.isfile(os.path.join(result_dir, f'{prefix}{seq_name}.txt'))
    ]
    if missing:
        preview = ', '.join(missing[:8])
        suffix = ' ...' if len(missing) > 8 else ''
        raise FileNotFoundError(
            f'官方评测需要 {len(dataset_obj.seqs_name)} 条序列结果，但缺少 '
            f'{len(missing)} 个文件: {preview}{suffix}')


def _read_pred_lines(path: str) -> List[str]:
    """读取预测文件所有行 (过滤空行)。"""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return [ln.strip() for ln in f if ln.strip()]


def _align_lasher_results(evaluator, result_dir: str, prefix: str = ''):
    """LasHeR 官方指标要求 预测行数 == 官方 GT 行数 (只截断不补齐)。

    若保存的预测短于官方 GT (常见于服务器数据缺失末尾帧), 官方 toolkit 会
    IndexError。此函数把每条序列预测对齐到官方 GT 帧数:
      - 超出 → 截断
      - 不足 → 用最后有效预测行补齐
    生成对齐后的临时目录并返回 (对齐目录, 诊断信息)。不改动原结果文件。
    """
    tmp_dir = tempfile.mkdtemp(prefix='lasher_align_')
    stats = {'short': [], 'long': [], 'ok': 0}
    for seq_name in evaluator.seqs_name:
        pred_path = os.path.join(result_dir, f'{prefix}{seq_name}.txt')
        if not os.path.isfile(pred_path):
            continue
        pred = _read_pred_lines(pred_path)
        gt = evaluator[seq_name]
        n_gt = len(gt)
        n_pred = len(pred)
        if n_pred == n_gt:
            stats['ok'] += 1
            aligned = pred
        elif n_pred < n_gt:
            # 用末帧补齐 (对应丢失帧沿用最后预测)
            last = pred[-1] if pred else '0,0,0,0'
            aligned = pred + [last] * (n_gt - n_pred)
            stats['short'].append((seq_name, n_pred, n_gt))
        else:
            aligned = pred[:n_gt]
            stats['long'].append((seq_name, n_pred, n_gt))
        out_path = os.path.join(tmp_dir, f'{prefix}{seq_name}.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(aligned) + '\n')
    return tmp_dir, stats


def evaluate_official_rgbt(dataset: str, result_dir: str, toolkit_root: str,
                           tracker_name: str = 'CIPTrack') -> Dict[str, float]:
    """按官方 toolkit 1.0.1 定义评测并保存机器可读结果。

    输入结果必须为每序列一个 ``x,y,w,h`` 文本文件。工具包会按照其历史协议
    对结果取整，并使用自身随包发布的可见光/红外标注，而非项目侧单模态标注。
    """
    dataset = dataset.lower()
    if dataset not in _SUPPORTED:
        raise ValueError(f'官方 RGBT toolkit 不支持数据集: {dataset}')
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f'结果目录不存在: {result_dir}')

    rgbt, resolved_toolkit_root = _load_toolkit(toolkit_root)
    dataset_module = importlib.import_module('rgbt.dataset')
    dataset_cls = {
        'gtot': dataset_module.GTOT,
        'rgbt210': dataset_module.RGBT210,
        'rgbt234': dataset_module.RGBT234,
        'lasher': dataset_module.LasHeR,
    }[dataset]

    if dataset == 'gtot':
        utils = importlib.import_module('rgbt.utils')
        utils.RGBT_start()
        try:
            evaluator = dataset_cls()
            _check_result_files(evaluator, result_dir)
            evaluator(tracker_name=tracker_name, result_path=result_dir,
                      bbox_type='ltwh')
            mpr, _ = evaluator.MPR(tracker_name)
            msr, _ = evaluator.MSR(tracker_name)
            metrics = {'MPR': float(mpr), 'MSR': float(msr)}
        finally:
            utils.RGBT_end()
    elif dataset == 'rgbt234':
        evaluator = dataset_cls()
        _check_result_files(evaluator, result_dir)
        evaluator(tracker_name=tracker_name, result_path=result_dir,
                  bbox_type='ltwh')
        mpr, _ = evaluator.MPR(tracker_name)
        msr, _ = evaluator.MSR(tracker_name)
        metrics = {'MPR': float(mpr), 'MSR': float(msr)}
    elif dataset == 'lasher':
        evaluator = dataset_cls()
        _check_result_files(evaluator, result_dir)
        # 官方 LasHeR 指标要求预测行数与官方 GT 一致, 否则越界报错
        # (服务器数据可能缺末尾帧导致预测短于官方 lasher_gt)
        eval_dir, align_stats = _align_lasher_results(evaluator, result_dir)
        try:
            evaluator(tracker_name=tracker_name, result_path=eval_dir,
                      bbox_type='ltwh')
            pr, _ = evaluator.PR(tracker_name)
            sr, _ = evaluator.SR(tracker_name)
            npr, _ = evaluator.NPR(tracker_name)
            metrics = {'PR': float(pr), 'SR': float(sr), 'NPR': float(npr)}
        finally:
            shutil.rmtree(eval_dir, ignore_errors=True)
        alignment_report = {
            'ok': int(align_stats['ok']),
            'padded_short': len(align_stats['short']),
            'truncated_long': len(align_stats['long']),
            'short_examples': [f'{s}: pred={p}, gt={g}' for s, p, g
                               in align_stats['short'][:10]],
            'long_examples': [f'{s}: pred={p}, gt={g}' for s, p, g
                              in align_stats['long'][:10]],
        }
        if align_stats['short']:
            print(f'[WARN] 有 {len(align_stats["short"])} 条序列预测帧数 < 官方 GT, '
                  f'已用末帧补齐后再评测。示例: '
                  f'{alignment_report["short_examples"][:3]}', flush=True)
        if align_stats['long']:
            print(f'[WARN] 有 {len(align_stats["long"])} 条序列预测帧数 > 官方 GT, '
                  f'已截断后再评测。示例: '
                  f'{alignment_report["long_examples"][:3]}', flush=True)
    else:
        evaluator = dataset_cls()
        _check_result_files(evaluator, result_dir)
        evaluator(tracker_name=tracker_name, result_path=result_dir,
                  bbox_type='ltwh')
        pr, _ = evaluator.PR(tracker_name)
        sr, _ = evaluator.SR(tracker_name)
        metrics = {'PR': float(pr), 'SR': float(sr)}

    payload = {
        'protocol': 'RGBT toolkit 1.0.1',
        'dataset': dataset,
        'tracker_name': tracker_name,
        'bbox_type': 'ltwh',
        'result_rounding': 'toolkit TrackerResult rounds input boxes to integers',
        'num_sequences': int(len(evaluator.seqs_name)),
        'toolkit_root': resolved_toolkit_root,
        'metrics': metrics,
    }
    if dataset == 'lasher':
        payload['frame_alignment'] = alignment_report
    json_path = os.path.join(result_dir, 'official_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(result_dir, 'official_metrics.csv')
    fields = ['protocol', 'dataset', 'tracker_name', 'bbox_type', 'num_sequences'] + list(metrics)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            'protocol': payload['protocol'],
            'dataset': dataset,
            'tracker_name': tracker_name,
            'bbox_type': 'ltwh',
            'num_sequences': payload['num_sequences'],
            **metrics,
        })
    return metrics
