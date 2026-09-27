#!/usr/bin/env python3
"""
RGBTSwinTrack-EncPromptLoRA — VTUAV / GTOT / RGBT210 / RGBT234 测试
====================================================================
参考 SGTrack RGBT_workspace/test.py 的多进程评估框架, 数据目录组织与
SGTrack 服务器一致 (可用环境变量覆盖):

  SGTEST_DATA_ROOT = /root/RGBTData
    vtuav_st : <root>/VTUAV/test_ST   (两级目录, 官方 13 组)
    vtuav_lt : <root>/VTUAV/test_LT   (两级目录, 官方 10 组)
    gtot     : <root>/GTOT            (v/ i/ groundTruth_v.txt)
    rgbt210  : <root>/RGBT210         (visible/ infrared/ visible.txt)
    rgbt234  : <root>/RGBT234         (visible/ infrared/ visible.txt)

指标: 实时诊断指标 AO / SS / SR50 / SR75 / PS / NPS；GTOT、RGBT210、
RGBT234 完成跟踪后额外调用 RGBT toolkit 1.0.1 写出官方汇总指标。

用法:
  python test_vtuav_promptlora.py --weight <ckpt.pth> \
      --dataset vtuav_st --data_root /root/RGBTData
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Dict, List

import cv2
import numpy as np
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# ── 项目根目录 ──
_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)

# ── 复用 EncFuse 测试工具 (curation/指标/图像读取) ──
from test_lasher_rgbt_enc_fuse import (
    read_frame_rgb, read_frame_tir, read_bboxes, compute_curation_params,
    crop_and_resize, map_bbox_to_original, clamp_bbox, iou, compute_metrics,
    save_preds,
)

from datasets.RGBT.vtuav_rgbt import collect_test_sequences
from core.official_rgbt_eval import evaluate_official_rgbt

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CONFIG = {
    'template_size': [192, 192],
    'search_size':   [384, 384],
    'template_area_factor': 2.0,
    'search_area_factor':   4.0,
    'min_object_size': [10, 10],
    'window_penalty': 0.49,
    'search_feat_shape': [24, 24],
}

# 与 train_vtuav_promptlora.py / train_rgbt_enc_fuse_v1.py 一致的归一化统计
_IMAGENET_MEAN = list(IMAGENET_DEFAULT_MEAN)
_IMAGENET_STD = list(IMAGENET_DEFAULT_STD)
_TIR_MEAN = [0.449, 0.449, 0.449]
_TIR_STD = [0.226, 0.226, 0.226]

_SUCCESS_THRESHOLDS = np.linspace(0, 1, 21)

# 指标顺序 (对齐 SGTrack RGBT_workspace/test.py 的 _METRICS)
_METRICS8 = ['AO', 'SS', 'SR50', 'SR75', 'PS', 'NPS', 'MSR', 'MPR']

# 以下 MSR/MPR 是逐序列诊断用的 SGTrack 风格指标；GTOT/RGBT210/RGBT234
# 的论文正式结果由 RGBT toolkit 1.0.1 在测试结束后重新计算。
_MPRMSR_DATASETS = {'gtot', 'rgbt234', 'vtuav_st', 'vtuav_lt'}
_MPRMSR_SHIFT_RANGE = range(-10, 11)

_MODEL_REGISTRY = {
    'enc_promptlora_vtuav': (
        'config/SwinRGBTrack/Base-384-enc-promptlora-vtuav',
        'models.methods.SwinRGBTrack.builder_enc_promptlora',
        'build_rgbt_enc_promptlora'),
    'enc_promptlora': (
        'config/SwinRGBTrack/Base-384-enc-promptlora',
        'models.methods.SwinRGBTrack.builder_enc_promptlora',
        'build_rgbt_enc_promptlora'),
}


def _build_model(model_type):
    from importlib import import_module
    from core.run.event_dispatcher.register import EventRegister
    from miscellanies.yaml_ops import load_yaml
    cfg_dir, module_name, func_name = _MODEL_REGISTRY[model_type]
    config = load_yaml(os.path.join(_PRJ_ROOT, cfg_dir, 'config.yaml'))
    builder = getattr(import_module(module_name), func_name)
    er = EventRegister('model/')
    return builder(config, False, 1, 1, er, False)


def load_model(weight_path, device, model_type='enc_promptlora_vtuav'):
    print(f'[INFO] Loading model (model_type={model_type})...', flush=True)
    model = _build_model(model_type)

    print(f'[INFO] 加载权重: {weight_path}', flush=True)
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)

    model_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped.append(k)
    missing = set(model_state.keys()) - set(filtered_state.keys())
    if missing:
        print(f'[WARN] 缺失 {len(missing)} 个键, 保留初始化值')
    if skipped:
        print(f'[WARN] 跳过 {len(skipped)} 个形状不匹配的键')
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 评估指标 (含 MSR/MPR)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(preds, gts, dataset: str = '') -> dict:
    """AO/SS/SR50/SR75/PS/NPS + MSR/MPR (GTOT/RGBT234/VTUAV)。"""
    valid = [(p, g) for p, g in zip(preds, gts)
             if len(p) >= 4 and len(g) >= 4 and g[2] > 0 and g[3] > 0]
    if not valid:
        return dict(AO=-1., SS=-1., SR50=-1., SR75=-1., PS=-1., NPS=-1.,
                    MSR=-1., MPR=-1., n_valid=0)

    use_mpr_msr = dataset in _MPRMSR_DATASETS

    ious, dists, nd = [], [], []
    msr_ious, mpr_dists = [], []
    for p, g in valid:
        ious.append(iou(p, g))
        cx_p = p[0] + p[2] / 2
        cy_p = p[1] + p[3] / 2
        cx_g = g[0] + g[2] / 2
        cy_g = g[1] + g[3] / 2
        d = float(np.sqrt((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2))
        dists.append(d)
        nd.append(d / float(np.sqrt(g[2] * g[3])) if g[2] * g[3] > 0 else 0.)

        if use_mpr_msr:
            best_iou, best_dist = ious[-1], d
            for dx in _MPRMSR_SHIFT_RANGE:
                for dy in _MPRMSR_SHIFT_RANGE:
                    sp = [p[0] + dx, p[1] + dy, p[2], p[3]]
                    si = iou(sp, g)
                    if si > best_iou:
                        best_iou = si
                    scx = sp[0] + sp[2] / 2
                    scy = sp[1] + sp[3] / 2
                    sd = float(np.sqrt((scx - cx_g) ** 2 + (scy - cy_g) ** 2))
                    if sd < best_dist:
                        best_dist = sd
            msr_ious.append(best_iou)
            mpr_dists.append(best_dist)
        else:
            msr_ious.append(ious[-1])
            mpr_dists.append(d)

    ia = np.array(ious, dtype=np.float64)
    da = np.array(dists, dtype=np.float64)
    na = np.array(nd, dtype=np.float64)
    mia = np.array(msr_ious, dtype=np.float64)
    mda = np.array(mpr_dists, dtype=np.float64)

    sr_curve = np.array([(ia >= t).mean() for t in _SUCCESS_THRESHOLDS])
    msr_curve = np.array([(mia >= t).mean() for t in _SUCCESS_THRESHOLDS])

    return dict(
        AO=float(ia.mean()),
        SS=float(np.trapz(sr_curve, _SUCCESS_THRESHOLDS)),
        SR50=float((ia >= 0.50).mean()),
        SR75=float((ia >= 0.75).mean()),
        PS=float((da <= 20.).mean()),
        NPS=float((na <= 0.5).mean()),
        MSR=float(np.trapz(msr_curve, _SUCCESS_THRESHOLDS)),
        MPR=float((mda <= 20.).mean()),
        n_valid=len(valid),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 子进程 Worker
# ══════════════════════════════════════════════════════════════════════════════

def _match_size(tir, rgb):
    """TIR 尺寸对齐 RGB (GTOT 等数据集两者分辨率可能不同)。"""
    if tir.shape[:2] != rgb.shape[:2]:
        tir = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    return tir


def _worker(worker_id, seqs, result_dir, weight_path, gpu_id, out_q,
            model_type, dataset):
    try:
        if gpu_id >= 0 and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
        model = load_model(weight_path, device, model_type)

        cfg = MODEL_CONFIG
        t_sz = cfg['template_size'][0]
        s_sz = cfg['search_size'][0]
        template_area_factor = cfg['template_area_factor']
        search_area_factor = cfg['search_area_factor']
        window_penalty = cfg['window_penalty']
        feat_h, feat_w = cfg['search_feat_shape']

        hann_window = torch.outer(
            torch.hann_window(feat_h, periodic=False),
            torch.hann_window(feat_w, periodic=False),
        ).flatten().to(device)

        rgb_norm = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        tir_norm = transforms.Normalize(mean=_TIR_MEAN, std=_TIR_STD)

        out_q.put(('ready', worker_id))
    except Exception:
        out_q.put(('init_error', worker_id, traceback.format_exc()))
        return

    for seq in seqs:
        seq_name = seq['seq_name']
        t0 = time.perf_counter()
        try:
            frame_paths_rgb = seq['rgb_paths']
            frame_paths_tir = seq['tir_paths']
            gt_all = seq['bbox']

            n_rgb = len(frame_paths_rgb)
            n_tir = len(frame_paths_tir)
            n_gt = len(gt_all)
            n = min(n_rgb, n_tir, n_gt)
            if n < 2:
                raise ValueError(f'序列 {seq_name}: 帧数不足 '
                                 f'(rgb={n_rgb}, tir={n_tir}, gt={n_gt})')

            preds = []
            init_bbox = [float(v) for v in gt_all[0]]
            img0_rgb = read_frame_rgb(frame_paths_rgb[0])
            img0_tir = _match_size(read_frame_tir(frame_paths_tir[0]), img0_rgb)
            init_bbox = clamp_bbox(init_bbox, img0_rgb.shape[1], img0_rgb.shape[0])

            tp = compute_curation_params(init_bbox, template_area_factor, (t_sz, t_sz))
            z_rgb_img, zm_rgb = crop_and_resize(img0_rgb, (t_sz, t_sz), tp)
            z_tir_img, zm_tir = crop_and_resize(img0_tir, (t_sz, t_sz), tp)

            z_rgb_tensor = torch.from_numpy(z_rgb_img / 255.0).permute(2, 0, 1).float()
            z_rgb_tensor = rgb_norm(z_rgb_tensor).unsqueeze(0).to(device)
            z_tir_tensor = torch.from_numpy(z_tir_img / 255.0).permute(2, 0, 1).float()
            z_tir_tensor = tir_norm(z_tir_tensor).unsqueeze(0).to(device)

            with torch.no_grad():
                cached = model.initialize(z_rgb_tensor, z_tir_tensor)

            preds.append(list(init_bbox))
            cached_search_bbox = list(init_bbox)
            t_track = time.perf_counter()

            for fi in range(1, n):
                img_f_rgb = read_frame_rgb(frame_paths_rgb[fi])
                img_f_tir = _match_size(read_frame_tir(frame_paths_tir[fi]),
                                        img_f_rgb)
                img_h, img_w = img_f_rgb.shape[:2]

                adj_bbox = cached_search_bbox.copy()
                adj_bbox[2] = max(adj_bbox[2], cfg['min_object_size'][0])
                adj_bbox[3] = max(adj_bbox[3], cfg['min_object_size'][1])

                sp = compute_curation_params(adj_bbox, search_area_factor, (s_sz, s_sz))
                x_rgb_img, _ = crop_and_resize(img_f_rgb, (s_sz, s_sz), sp,
                                               image_mean=zm_rgb)
                x_tir_img, _ = crop_and_resize(img_f_tir, (s_sz, s_sz), sp,
                                               image_mean=zm_tir)

                x_rgb_tensor = torch.from_numpy(x_rgb_img / 255.0).permute(2, 0, 1).float()
                x_rgb_tensor = rgb_norm(x_rgb_tensor).unsqueeze(0).to(device)
                x_tir_tensor = torch.from_numpy(x_tir_img / 255.0).permute(2, 0, 1).float()
                x_tir_tensor = tir_norm(x_tir_tensor).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model.track(cached, x_rgb_tensor, x_tir_tensor)

                class_score = output['class_score']
                bbox_pred = output['bbox']
                cls_map = class_score.view(1, feat_h * feat_w)
                cls_map = (cls_map * (1 - window_penalty)
                           + hann_window.view(1, feat_h * feat_w) * window_penalty)
                _, best_idx = torch.max(cls_map, dim=1)
                bbox_flat = bbox_pred.view(1, feat_h * feat_w, 4)
                best_bbox = bbox_flat[0, best_idx[0], :].cpu().numpy()

                rx, ry, rw, rh = best_bbox.tolist()
                cx, cy = rx * s_sz, ry * s_sz
                w, h = rw * s_sz, rh * s_sz
                pred_orig = map_bbox_to_original(
                    [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], sp)
                final_bbox = clamp_bbox(pred_orig, img_w, img_h)
                preds.append(final_bbox)
                cached_search_bbox = final_bbox

            elapsed = time.perf_counter() - t0
            fps = (n - 1) / max(time.perf_counter() - t_track, 1e-6)

            save_preds(preds, seq_name.replace('/', '_'), result_dir)
            m = _compute_metrics(preds, gt_all[:n], dataset)
            m.update(seq_name=seq_name, fps=float(fps), elapsed=float(elapsed))
            out_q.put(('result', worker_id, m))

        except Exception:
            elapsed = time.perf_counter() - t0
            out_q.put(('seq_error', worker_id, seq_name,
                       traceback.format_exc(), float(elapsed)))


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser('RGBTSwinTrack-EncPromptLoRA VTUAV/GTOT/RGBT 测试')
    p.add_argument('--weight', required=True, help='模型权重文件路径')
    p.add_argument('--model_type', default='enc_promptlora_vtuav',
                   choices=list(_MODEL_REGISTRY.keys()))
    p.add_argument('--dataset', default='vtuav_st',
                   choices=['vtuav_st', 'vtuav_lt', 'gtot', 'rgbt210', 'rgbt234'])
    p.add_argument('--data_root',
                   default=os.environ.get('SGTEST_DATA_ROOT', '/root/RGBTData'),
                   help='RGBT 数据根目录 (组织形式 <root>/<Dataset>)')
    p.add_argument('--save_dir',
                   default=os.path.join(_PRJ_ROOT, 'test_results'))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--sequence', default='', help='只跑单条序列 (调试)')
    p.add_argument('--official_toolkit',
                   default=os.environ.get(
                       'RGBT_TOOLKIT_HOME',
                       '/root'),
                   help='RGBT toolkit 1.0.1 根目录；可用 RGBT_TOOLKIT_HOME 覆盖')
    p.add_argument('--no_official_eval', action='store_true',
                   help='跳过 GTOT/RGBT210/RGBT234 官方 RGBT toolkit 汇总（仅调试）')
    p.add_argument('--skip_existing', action='store_true',
                   help='已有完整预测 .txt 的序列直接读取结果并统计, 不重新推理')
    args = p.parse_args()

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        print('[WARN] 未检测到 CUDA GPU，将使用 CPU 推理（较慢）')

    tag = {'enc_promptlora_vtuav': 'swintrack_b384_enc_promptlora_vtuav',
           'enc_promptlora': 'swintrack_b384_enc_promptlora'}[args.model_type]
    result_dir = os.path.join(args.save_dir, args.dataset, tag)
    os.makedirs(result_dir, exist_ok=True)

    print('=' * 78)
    print(f'  权重文件 : {args.weight}')
    print(f'  模型     : {args.model_type}')
    print(f'  数据集   : {args.dataset}  ({args.data_root})')
    print(f'  结果目录 : {result_dir}')
    print(f'  GPU 数量 : {num_gpus}')
    print(f'  Workers  : {args.workers}')
    if args.dataset in {'gtot', 'rgbt210', 'rgbt234'}:
        print(f'  官方评测 : {"关闭" if args.no_official_eval else args.official_toolkit}')
    print('=' * 78)

    # ── 收集序列 ──
    seqs = list(collect_test_sequences(args.dataset, args.data_root))
    if args.sequence:
        seqs = [s for s in seqs if s['seq_name'].split('/')[-1] == args.sequence
                or s['seq_name'] == args.sequence]

    # ── skip_existing: 复用已有完整预测 .txt 的序列 (不重新推理) ──
    # 一条序列视为"已完成"的条件: 预测 txt 存在且行数 == 该序列帧数 n。
    cached_metrics = []
    if args.skip_existing:
        kept = []
        for s in seqs:
            n_frames = min(len(s['rgb_paths']), len(s['tir_paths']),
                           len(s['bbox']))
            txt = os.path.join(result_dir,
                               s['seq_name'].replace('/', '_') + '.txt')
            rows = read_bboxes(txt) if os.path.isfile(txt) else []
            if n_frames >= 2 and len(rows) == n_frames:
                m = _compute_metrics(rows, s['bbox'][:n_frames], args.dataset)
                m.update(seq_name=s['seq_name'], fps=0., elapsed=0.,
                         cached=True)
                cached_metrics.append(m)
            else:
                kept.append(s)
        seqs = kept
        if cached_metrics:
            print(f'[INFO] skip_existing: {len(cached_metrics)} 条序列已有'
                  f'完整结果, 直接读取统计 (跳过推理)')

    n_seqs = len(seqs) + len(cached_metrics)
    if n_seqs == 0:
        print('[ERROR] 没有找到任何序列，请检查数据集路径和 --sequence 参数。')
        sys.exit(1)
    has_official_protocol = args.dataset in {'gtot', 'rgbt210', 'rgbt234'}
    print(f'[INFO] 共 {n_seqs} 条序列 (新推理 {len(seqs)}, '
          f'复用已有 {len(cached_metrics)})')

    # ── 按帧数升序后贪心均衡分配 (仅新推理的序列) ──
    seqs = sorted(seqs, key=lambda s: len(s['rgb_paths']))
    n_workers = min(args.workers, len(seqs))
    chunks: List[List] = [[] for _ in range(n_workers)]
    chunk_frames = [0] * n_workers
    for s in seqs:
        wid = min(range(n_workers), key=lambda i: chunk_frames[i])
        chunks[wid].append(s)
        chunk_frames[wid] += len(s['rgb_paths'])
    chunks = [c for c in chunks if c]
    n_workers = len(chunks)
    for i, (c, f) in enumerate(zip(chunks, chunk_frames)):
        print(f'       worker {i}: {len(c)} 条序列, 约 {f} 帧')

    # ── 启动子进程 (仅当存在需新推理的序列) ──
    out_q: mp.Queue = mp.Queue()
    procs = []
    if n_workers > 0:
        for wid, chunk in enumerate(chunks):
            gpu_id = 0 if num_gpus > 0 else -1
            proc = mp.Process(
                target=_worker,
                args=(wid, chunk, result_dir, args.weight, gpu_id, out_q,
                      args.model_type, args.dataset),
                daemon=True,
            )
            proc.start()
            procs.append(proc)

        print(f'\n[INFO] 等待 {n_workers} 个 worker 加载模型...', flush=True)
        ready = 0
        while ready < n_workers:
            msg = out_q.get()
            if msg[0] == 'ready':
                ready += 1
                print(f'[INFO]   worker {msg[1]} 就绪 ({ready}/{n_workers})',
                      flush=True)
            elif msg[0] == 'init_error':
                print(f'[ERROR]  worker {msg[1]} 加载失败:\n{msg[2]}', flush=True)
                ready += 1
        print(f'[INFO] 所有 worker 就绪，开始追踪...\n', flush=True)

    # ── 收集结果 (含 worker 崩溃看门狗, 避免永久挂起) ──
    all_recs: Dict[str, dict] = {}
    errors = []
    done = 0
    t_all = time.perf_counter()
    cum_msr_sum = cum_mpr_sum = 0.0
    cum_valid = 0

    # worker_id -> 尚未返回的序列名 (用于 worker 崩溃时补记)
    # chunk 顺序与 procs 下标一致: enumerate(chunks) 时 wid=i
    pending_seqs: Dict[int, set] = {wid: {s['seq_name'] for s in chunk}
                                    for wid, chunk in enumerate(chunks)}
    # 记录每条序列来自哪个 worker, 便于崩溃时定位
    seq_to_wid: Dict[str, int] = {}
    for wid, chunk in enumerate(chunks):
        for s in chunk:
            seq_to_wid[s['seq_name']] = wid

    def _mark_seq_error(name, wid, reason):
        """把一条序列记为错误, 并计数。"""
        nonlocal done
        done += 1
        errors.append((name, reason))
        err_line = str(reason).strip().splitlines()[-1][:80] if reason else 'unknown'
        print(f'[{done:03d}/{n_seqs:03d}] {name:<32s} [ERROR] {err_line} '
              f'(worker {wid} 异常/退出)', flush=True)
        all_recs[name] = dict(seq_name=name, AO=-1., SS=-1., SR50=-1.,
                              SR75=-1., PS=-1., NPS=-1., MSR=-1.,
                              MPR=-1., n_valid=0, fps=0., elapsed=0.)
        pending_seqs[wid].discard(name)

    HDR = (f"\n{'#':<7} {'序列名':<32} "
           f"{'AO':>6} {'MSR':>6} {'MPR':>6} "
           f"{'cMSR':>6} {'cMPR':>6} "
           f"{'SR50':>6} {'SR75':>6} "
           f"{'PS':>6} {'NPS':>6} {'FPS':>6} {'耗时s':>7}")
    SEP = '-' * (len(HDR) + 1)
    print(HDR)
    print(SEP)

    # ── 已复用序列 (skip_existing): 主进程已直接读结果, 立即计入 ──
    for m in cached_metrics:
        done += 1
        all_recs[m['seq_name']] = m
        if m.get('AO', -1) >= 0:
            cum_msr_sum += m['MSR']
            cum_mpr_sum += m['MPR']
            cum_valid += 1
            c_msr = cum_msr_sum / cum_valid
            c_mpr = cum_mpr_sum / cum_valid
            print(f'[{done:03d}/{n_seqs:03d}] {m["seq_name"]:<32s} '
                  f'{m["AO"]:6.3f} {m["MSR"]:6.3f} {m["MPR"]:6.3f} '
                  f'{c_msr:6.3f} {c_mpr:6.3f} '
                  f'{m["SR50"]:6.3f} {m["SR75"]:6.3f} '
                  f'{m["PS"]:6.3f} {m["NPS"]:6.3f} '
                  f'{"cached":>6} {m.get("elapsed", 0.):7.1f}', flush=True)
        else:
            print(f'[{done:03d}/{n_seqs:03d}] {m["seq_name"]:<32s} '
                  f'{"---":>6} {"---":>6} {"---":>6} {"---":>6} {"---":>6} '
                  f'{"---":>6} {"---":>6} {"---":>6} {"---":>6} '
                  f'{"cached":>6} {m.get("elapsed", 0.):7.1f}', flush=True)

    while done < n_seqs:
        try:
            msg = out_q.get(timeout=30)
        except Exception:
            # 队列超时: 检查是否有 worker 进程崩溃 (进程已退出但没发消息)
            msg = None
        if msg is not None:
            mtype = msg[0]
            if mtype == 'result':
                _, wid, m = msg
                done += 1
                all_recs[m['seq_name']] = m
                pending_seqs[wid].discard(m['seq_name'])
                if m.get('AO', -1) >= 0:
                    cum_msr_sum += m['MSR']
                    cum_mpr_sum += m['MPR']
                    cum_valid += 1
                    c_msr = cum_msr_sum / cum_valid
                    c_mpr = cum_mpr_sum / cum_valid
                    print(f'[{done:03d}/{n_seqs:03d}] {m["seq_name"]:<32s} '
                          f'{m["AO"]:6.3f} {m["MSR"]:6.3f} {m["MPR"]:6.3f} '
                          f'{c_msr:6.3f} {c_mpr:6.3f} '
                          f'{m["SR50"]:6.3f} {m["SR75"]:6.3f} '
                          f'{m["PS"]:6.3f} {m["NPS"]:6.3f} '
                          f'{m["fps"]:6.1f} {m["elapsed"]:7.1f}', flush=True)
                else:
                    print(f'[{done:03d}/{n_seqs:03d}] {m["seq_name"]:<32s} '
                          f'{"---":>6} {"---":>6} {"---":>6} {"---":>6} {"---":>6} '
                          f'{"---":>6} {"---":>6} {"---":>6} {"---":>6} '
                          f'{m.get("fps", 0.):6.1f} {m.get("elapsed", 0.):7.1f}',
                          flush=True)
            elif mtype == 'seq_error':
                _, wid, name, tb, elapsed = msg
                done += 1
                errors.append((name, tb))
                pending_seqs[wid].discard(name)
                err_line = tb.strip().splitlines()[-1][:80]
                print(f'[{done:03d}/{n_seqs:03d}] {name:<32s} [ERROR] {err_line} '
                      f'({elapsed:.1f}s)', flush=True)
                all_recs[name] = dict(seq_name=name, AO=-1., SS=-1., SR50=-1.,
                                      SR75=-1., PS=-1., NPS=-1., MSR=-1.,
                                      MPR=-1., n_valid=0, fps=0., elapsed=elapsed)
        else:
            # 看门狗: 找出已退出的 worker, 把其未完成序列补记为错误
            dead = [i for i, p in enumerate(procs) if not p.is_alive()]
            if dead:
                for wid in dead:
                    leftover = list(pending_seqs.get(wid, ()))
                    if leftover:
                        print(f'[WARN] worker {wid} 已退出, 剩余 '
                              f'{len(leftover)} 条序列标记为失败', flush=True)
                    for name in leftover:
                        _mark_seq_error(name, wid, 'worker process exited')
            else:
                # 所有 worker 都活着但长时间无消息 → 打印一次心跳, 继续等
                pass

    for proc in procs:
        proc.join(timeout=60)

    t_total = time.perf_counter() - t_all
    print(SEP)
    print(f'\n[INFO] 追踪完成，总耗时 {t_total / 60:.1f} 分钟', flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # 汇总统计 (对齐 SGTrack: 序列均值 + 帧加权)
    # ════════════════════════════════════════════════════════════════════════
    recs = list(all_recs.values())
    valid = [r for r in recs if r.get('AO', -1) >= 0]

    if valid:
        total_frames = sum(r['n_valid'] for r in valid)
        seq_means = {k: float(np.mean([r[k] for r in valid]))
                     for k in _METRICS8}
        frm_means = {k: (float(sum(r[k] * r['n_valid'] for r in valid) /
                               total_frames) if total_frames > 0 else -1.)
                     for k in _METRICS8}
        mfps = float(np.mean([r['fps'] for r in valid if r['fps'] > 0]))
    else:
        seq_means = {k: -1. for k in _METRICS8}
        frm_means = {k: -1. for k in _METRICS8}
        mfps = 0.
        total_frames = 0

    W = 9
    print('\n' + '=' * 78)
    print(f'[汇总] {len(valid)}/{len(recs)} 条有效序列  '
          f'总帧数={total_frames}  平均FPS={mfps:.1f}')
    print(f"{'':12}" + ''.join(f'{k:>{W}}' for k in _METRICS8))
    print(f"{'序列均值(%)':<12}" +
          ''.join(f'{seq_means[k] * 100:>{W}.2f}' for k in _METRICS8))
    print(f"{'帧加权(%)':<12}" +
          ''.join(f'{frm_means[k] * 100:>{W}.2f}' for k in _METRICS8))
    print('=' * 78)
    result_label = '[DIAGNOSTIC RESULT]' if has_official_protocol else '[RESULT]'
    print(result_label + ' ' +
          ' '.join(f'{k}={seq_means[k]:.4f}' for k in _METRICS8), flush=True)
    if has_official_protocol and not args.no_official_eval:
        print('[INFO] GTOT/RGBT210/RGBT234 的论文正式指标将在跟踪结束后由 '
              'RGBT toolkit 1.0.1 重新汇总。')

    # ── per_seq_metrics.csv (含 cMSR/cMPR 累计均值, 按完成顺序) ──
    csv_path = os.path.join(result_dir, 'per_seq_metrics.csv')
    csv_cum_msr, csv_cum_mpr, csv_cum_n = 0.0, 0.0, 0
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seq_name'] + _METRICS8 + ['cMSR', 'cMPR',
                                               'n_valid', 'fps', 'elapsed'])
        for r in recs:
            if r.get('AO', -1) >= 0:
                csv_cum_msr += r['MSR']
                csv_cum_mpr += r['MPR']
                csv_cum_n += 1
            c_msr = f'{csv_cum_msr / csv_cum_n:.4f}' if csv_cum_n > 0 else '-1'
            c_mpr = f'{csv_cum_mpr / csv_cum_n:.4f}' if csv_cum_n > 0 else '-1'
            w.writerow([r['seq_name']] + [r[k] for k in _METRICS8] +
                       [c_msr, c_mpr, r.get('n_valid', 0),
                        r.get('fps', 0.), r.get('elapsed', 0.)])
    print(f'[INFO] per_seq_metrics.csv → {csv_path}')

    # ── eval_history.csv (对齐 SGTrack: seq_*/frm_* 列) ──
    hist_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(hist_dir, exist_ok=True)
    hist_csv = os.path.join(hist_dir, 'eval_history.csv')
    hist_fields = (['ckpt_tag', 'checkpoint', 'dataset', 'n_valid', 'n_total',
                    'mean_fps'] +
                   [f'seq_{k}' for k in _METRICS8] +
                   [f'frm_{k}' for k in _METRICS8])
    hist_row = {'ckpt_tag': tag,
                'checkpoint': os.path.basename(args.weight),
                'dataset': args.dataset,
                'n_valid': len(valid),
                'n_total': len(recs),
                'mean_fps': f'{mfps:.2f}'}
    for k in _METRICS8:
        hist_row[f'seq_{k}'] = f'{seq_means[k]:.4f}'
        hist_row[f'frm_{k}'] = f'{frm_means[k]:.4f}'
    # 旧格式兼容: header 不一致时备份旧文件, 避免列错位
    is_new = True
    if os.path.isfile(hist_csv):
        with open(hist_csv, 'r') as f:
            first = f.readline().strip()
        if first == ','.join(hist_fields):
            is_new = False
        else:
            legacy = hist_csv.replace('.csv', f'_legacy_{int(time.time())}.csv')
            os.rename(hist_csv, legacy)
            print(f'[INFO] 旧版 eval_history.csv 已备份为 {os.path.basename(legacy)}')
    with open(hist_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(hist_fields)
        w.writerow([hist_row[k] for k in hist_fields])
    print(f'[INFO] eval_history.csv → {hist_csv}')

    # ── GTOT/RGBT210/RGBT234: 以官方 RGBT toolkit 1.0.1 为最终结果 ──
    # 工具包会使用自身发布的 visible/infrared 标注，按历史协议转换并取整
    # 预测框。因此 official_metrics.* 是这些数据集用于论文对比的唯一口径。
    official_metrics = None
    if args.dataset in {'gtot', 'rgbt210', 'rgbt234'} and not args.no_official_eval:
        if args.sequence:
            raise RuntimeError(
                '--sequence 仅适用于调试，无法生成官方全数据集结果；'
                '请移除此参数，或显式使用 --no_official_eval。')
        official_metrics = evaluate_official_rgbt(
            args.dataset, result_dir, args.official_toolkit, tracker_name=tag)
        metric_text = ' '.join(f'{key}={value:.4f}'
                               for key, value in official_metrics.items())
        print(f'[OFFICIAL RESULT] RGBT toolkit 1.0.1: {metric_text}', flush=True)
        print(f'[INFO] 官方结果 → {os.path.join(result_dir, "official_metrics.json")}')

    # ── summary.json (对齐 SGTrack: seq_means/frm_means 结构) ──
    summary = dict(
        checkpoint=args.weight, dataset=args.dataset,
        model_type=args.model_type,
        metric_protocol=('project diagnostic metrics; see official_toolkit for '
                         'the paper-ready GTOT/RGBT210/RGBT234 metrics'
                         if has_official_protocol else 'project evaluation metrics'),
        n_sequences=len(recs), n_valid=len(valid), mean_fps=mfps,
        seq_means={k: seq_means[k] for k in _METRICS8},
        frm_means={k: frm_means[k] for k in _METRICS8},
        total_time_min=f'{t_total / 60:.1f}',
        n_errors=len(errors),
    )
    if official_metrics is not None:
        summary['official_toolkit'] = dict(
            protocol='RGBT toolkit 1.0.1',
            metrics=official_metrics,
            metrics_file='official_metrics.json',
        )
    json_path = os.path.join(result_dir, 'summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[INFO] summary.json → {json_path}')

    if errors:
        print(f'[WARN] {len(errors)} 条序列失败: '
              f'{", ".join(e[0] for e in errors[:10])}')
        sys.exit(2)


if __name__ == '__main__':
    main()
