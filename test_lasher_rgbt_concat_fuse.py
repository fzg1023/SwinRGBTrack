#!/usr/bin/env python3
"""
RGBTSwinTrack-ConcatFuse — LasHeR RGBT 测试 (concat融合)
=========================================================================
与 EncFuse 的区别: 融合放在 Decoder 之后、Head 之前。

用法:
  python test_lasher_rgbt_dec_fuse.py --weight /path/to/weight.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

_LASHER_ROOT = os.environ.get('LASHER_ROOT', '/home/fzg/data/lasher')
_LASHER_TEST_ROOT = os.path.join(_LASHER_ROOT, 'testingset')

MODEL_CONFIG = {
    'template_size': [192, 192],
    'search_size':   [384, 384],
    'template_area_factor': 2.0,
    'search_area_factor':   4.0,
    'min_object_size': [10, 10],
    'window_penalty': 0.49,
    'interpolation_mode': 'bilinear',
    'backbone_name': 'swin_base_patch4_window12_384_in22k',
    'backbone_out_stage': 2,
    'transformer_dim': 512,
    'num_heads': 8,
    'mlp_ratio': 4,
    'encoder_num_layers': 8,
    'decoder_num_layers': 1,
    'template_feat_shape': [12, 12],
    'search_feat_shape':   [24, 24],
    'untied_abs_pos': True,
    'untied_rel_pos': True,
}

_SUCCESS_THRESHOLDS = np.linspace(0, 1, 21)
_METRICS = ['AO', 'SS', 'SR50', 'SR75', 'PS', 'NPS']

_IMAGENET_MEAN = torch.tensor(IMAGENET_DEFAULT_MEAN)
_IMAGENET_STD  = torch.tensor(IMAGENET_DEFAULT_STD)
_TIR_MEAN = torch.tensor(IMAGENET_DEFAULT_MEAN)
_TIR_STD  = torch.tensor(IMAGENET_DEFAULT_STD)

_DEFAULT_WEIGHT = os.path.join(_PRJ_ROOT, 'SwinTrack-B-384.pth')


# ══════════════════════════════════════════════════════════════════════════════
# 图像读取工具
# ══════════════════════════════════════════════════════════════════════════════

def list_frames(seq_dir: str, modal: str) -> List[str]:
    d = os.path.join(seq_dir, modal)
    if not os.path.isdir(d):
        return []
    files = sorted(
        f for f in os.listdir(d)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )
    return [os.path.join(d, f) for f in files]


def read_frame_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise IOError(f'cv2.imread failed: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_frame_tir(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'cv2.imread failed: {path}')
    if img.dtype == np.uint16:
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = ((img.astype(np.float32) - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)
    return img


def read_bboxes(path: str) -> List[List[float]]:
    if not os.path.isfile(path):
        return []
    bboxes = []
    with open(path, 'rb') as fb:
        raw = fb.read()
    if b'\x00' in raw:
        return []
    for line in raw.decode('utf-8', errors='replace').splitlines():
        line = line.strip().replace(',', ' ').replace('\t', ' ')
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            bboxes.append([float(v) for v in parts[:4]])
        except ValueError:
            continue
    return bboxes


def count_frames(seq_dir: str, modal: str = 'visible') -> int:
    return len(list_frames(seq_dir, modal))


def get_sequence_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f'数据集目录不存在: {root}')
    return sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )


# ══════════════════════════════════════════════════════════════════════════════
# AINet 风格 square crop + 坐标映射（统一训练和测试预处理）
# ══════════════════════════════════════════════════════════════════════════════

def sample_target(image, target_bb, area_factor, output_sz):
    """AINet square crop: 以 target_bb 为中心裁正方形, border constant 填充。"""
    x, y, w, h = target_bb
    crop_sz = math.ceil(math.sqrt(w * h) * area_factor)
    if crop_sz < 1: crop_sz = 1
    x1 = round(x + 0.5*w - crop_sz*0.5); x2 = x1 + crop_sz
    y1 = round(y + 0.5*h - crop_sz*0.5); y2 = y1 + crop_sz
    H, W = image.shape[:2]
    x1_pad = max(0, -x1); x2_pad = max(x2 - W + 1, 0)
    y1_pad = max(0, -y1); y2_pad = max(y2 - H + 1, 0)
    im_crop = image[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad, :]
    im_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
    resize_factor = output_sz / crop_sz
    im_padded = cv2.resize(im_padded, (output_sz, output_sz))
    return im_padded, resize_factor, crop_sz

def crop_bbox_to_original(bbox_crop_norm, bbox_extract, resize_factor, crop_sz):
    """将 crop 空间的归一化 CXCYWH bbox 映射回原始图像 XYWH。"""
    ecx = bbox_extract[0] + 0.5 * bbox_extract[2]
    ecy = bbox_extract[1] + 0.5 * bbox_extract[3]
    # 去归一化
    cx = bbox_crop_norm[0] * crop_sz
    cy = bbox_crop_norm[1] * crop_sz
    w  = bbox_crop_norm[2] * crop_sz
    h  = bbox_crop_norm[3] * crop_sz
    # 逆映射
    ocx = (cx - (crop_sz-1)/2) / resize_factor + ecx
    ocy = (cy - (crop_sz-1)/2) / resize_factor + ecy
    ow  = w / resize_factor
    oh  = h / resize_factor
    return [float(ocx - 0.5*ow), float(ocy - 0.5*oh), float(ow), float(oh)]


def clamp_bbox(bbox_xywh, img_w, img_h):
    x, y, w, h = bbox_xywh
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return [x, y, w, h]


# ══════════════════════════════════════════════════════════════════════════════
# RGBTSwinTrack-ConcatFuse 模型构建 (concat融合)
# ══════════════════════════════════════════════════════════════════════════════

def _load_config():
    from miscellanies.yaml_ops import load_yaml
    return load_yaml(os.path.join(_PRJ_ROOT, 'config', 'SwinRGBTrack', 'Base-384-concat-fuse', 'config.yaml'))


def _build_model_from_config(config):
    from core.run.event_dispatcher.register import EventRegister
    from models.methods.SwinRGBTrack.builder_concat_fuse import build_rgbt_concat_fuse
    er = EventRegister('model/')
    return build_rgbt_concat_fuse(config, False, 1, 1, er, False)


def load_model(weight_path, device):
    print(f'[INFO] Loading RGBTSwinTrack-ConcatFuse (concat融合)...', flush=True)
    config = _load_config()
    model = _build_model_from_config(config)

    print(f'[INFO] 加载权重: {weight_path}', flush=True)
    checkpoint = torch.load(weight_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

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
        print(f'[WARN] 缺失 {len(missing)} 个键 (ConcatFuse 新增或未初始化)，将保留初始化值')
    if skipped:
        print(f'[WARN] 跳过 {len(skipped)} 个形状不匹配的键')

    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 评估指标
# ══════════════════════════════════════════════════════════════════════════════

def iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.


def compute_metrics(preds, gts) -> dict:
    valid = [(p, g) for p, g in zip(preds, gts)
             if len(p) >= 4 and len(g) >= 4 and g[2] > 0 and g[3] > 0]
    if not valid:
        return dict(AO=-1., SS=-1., SR50=-1., SR75=-1.,
                    PS=-1., NPS=-1., n_valid=0)
    ious, dists, nd = [], [], []
    for p, g in valid:
        ious.append(iou(p, g))
        cx_p = p[0] + p[2]/2; cy_p = p[1] + p[3]/2
        cx_g = g[0] + g[2]/2; cy_g = g[1] + g[3]/2
        d = float(np.sqrt((cx_p-cx_g)**2 + (cy_p-cy_g)**2))
        dists.append(d)
        nd.append(d / float(np.sqrt(g[2]*g[3])) if g[2]*g[3] > 0 else 0.)
    ia = np.array(ious, dtype=np.float64)
    da = np.array(dists, dtype=np.float64)
    na = np.array(nd, dtype=np.float64)
    sr_curve = np.array([(ia >= t).mean() for t in _SUCCESS_THRESHOLDS])
    ss_auc = float(np.trapz(sr_curve, _SUCCESS_THRESHOLDS))
    return dict(
        AO=float(ia.mean()), SS=ss_auc,
        SR50=float((ia >= 0.50).mean()), SR75=float((ia >= 0.75).mean()),
        PS=float((da <= 20.).mean()), NPS=float((na <= 0.5).mean()),
        n_valid=len(valid),
    )


def save_preds(preds: list, seq_name: str, result_dir: str):
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{seq_name}.txt'), 'w') as f:
        for b in preds:
            f.write(','.join(f'{v:.4f}' for v in b) + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# 子进程 Worker
# ══════════════════════════════════════════════════════════════════════════════

def _worker(worker_id, seq_dirs, result_dir, weight_path, gpu_id, out_q):
    try:
        if gpu_id >= 0 and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
        model = load_model(weight_path, device)

        cfg = MODEL_CONFIG
        template_size = tuple(cfg['template_size'])
        search_size   = tuple(cfg['search_size'])
        t_sz = template_size[0]; s_sz = search_size[0]
        template_area_factor = cfg['template_area_factor']
        search_area_factor   = cfg['search_area_factor']
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

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        t0 = time.perf_counter()
        try:
            frame_paths_rgb = list_frames(seq_dir, 'visible')
            frame_paths_tir = list_frames(seq_dir, 'infrared')
            gt_all = read_bboxes(os.path.join(seq_dir, 'visible.txt'))
            if not gt_all:
                gt_all = read_bboxes(os.path.join(seq_dir, 'init.txt'))

            n_rgb = len(frame_paths_rgb)
            n_tir = len(frame_paths_tir)
            n_gt  = len(gt_all)
            n = min(n_rgb, n_tir, n_gt)
            if n < 2:
                raise ValueError(f'序列 {seq_name}: 帧数不足 (rgb={n_rgb}, tir={n_tir}, gt={n_gt})')

            preds = []

            init_bbox = gt_all[0]
            img0_rgb = read_frame_rgb(frame_paths_rgb[0])
            img0_tir = read_frame_tir(frame_paths_tir[0])

            # AINet: template crop (area_factor=2.0, output=192)
            z_rgb_img, _, _ = sample_target(img0_rgb, init_bbox, template_area_factor, t_sz)
            z_tir_img, _, _ = sample_target(img0_tir, init_bbox, template_area_factor, t_sz)

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
                img_f_tir = read_frame_tir(frame_paths_tir[fi])
                img_h, img_w = img_f_rgb.shape[:2]

                adj_bbox = cached_search_bbox.copy()
                adj_bbox[2] = max(adj_bbox[2], cfg['min_object_size'][0])
                adj_bbox[3] = max(adj_bbox[3], cfg['min_object_size'][1])

                # AINet: search crop (area_factor=4.0, output=384)
                x_rgb_img, x_rf, x_crop_sz = sample_target(img_f_rgb, adj_bbox, search_area_factor, s_sz)
                x_tir_img, _, _ = sample_target(img_f_tir, adj_bbox, search_area_factor, s_sz)

                x_rgb_tensor = torch.from_numpy(x_rgb_img / 255.0).permute(2, 0, 1).float()
                x_rgb_tensor = rgb_norm(x_rgb_tensor).unsqueeze(0).to(device)
                x_tir_tensor = torch.from_numpy(x_tir_img / 255.0).permute(2, 0, 1).float()
                x_tir_tensor = tir_norm(x_tir_tensor).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model.track(cached, x_rgb_tensor, x_tir_tensor)

                class_score = output['class_score']; bbox_pred = output['bbox']
                cls_map = class_score.view(1, feat_h * feat_w)
                cls_map = cls_map * (1 - window_penalty) + hann_window.view(1, feat_h * feat_w) * window_penalty
                _, best_idx = torch.max(cls_map, dim=1)
                bbox_flat = bbox_pred.view(1, feat_h * feat_w, 4)
                best_bbox = bbox_flat[0, best_idx[0], :].cpu().numpy()

                # AINet inverse mapping: crop coords → original image
                original_bbox = crop_bbox_to_original(best_bbox, adj_bbox, x_rf, x_crop_sz)
                final_bbox = clamp_bbox(original_bbox, img_w, img_h)
                preds.append(final_bbox)
                cached_search_bbox = final_bbox

            elapsed = time.perf_counter() - t0
            fps = (n - 1) / max(time.perf_counter() - t_track, 1e-6)
            save_preds(preds, seq_name, result_dir)
            m = compute_metrics(preds, gt_all[:n])
            m.update(seq_name=seq_name, fps=float(fps), elapsed=float(elapsed))
            out_q.put(('result', worker_id, m))

        except Exception:
            elapsed = time.perf_counter() - t0
            out_q.put(('seq_error', worker_id, seq_name, traceback.format_exc(), float(elapsed)))


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser('RGBTSwinTrack-ConcatFuse LasHeR 测试 (concat融合)')
    p.add_argument('--weight', default=_DEFAULT_WEIGHT, help='模型权重文件路径')
    p.add_argument('--dataset_root', default=_LASHER_TEST_ROOT, help='LasHeR 测试集根目录')
    p.add_argument('--save_dir', default=os.path.join(_PRJ_ROOT, 'test_results'), help='结果保存目录')
    p.add_argument('--workers', type=int, default=4, help='并行 worker 进程数')
    p.add_argument('--sequence', default='', help='只跑单条序列（调试用）')
    args = p.parse_args()

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        print('[WARN] 未检测到 CUDA GPU，将使用 CPU 推理（较慢）')

    n_workers = args.workers

    save_name = 'swintrack_b384_concatfuse'
    result_dir = os.path.join(args.save_dir, 'lasher', save_name)
    os.makedirs(result_dir, exist_ok=True)

    print('=' * 78)
    print(f'  权重文件 : {args.weight}')
    print(f'  模态     : RGBT ConcatFuse (concat融合, Head前)')
    print(f'  数据集   : {args.dataset_root}')
    print(f'  结果目录 : {result_dir}')
    print(f'  GPU 数量 : {num_gpus}')
    print(f'  Workers  : {n_workers}')
    print('=' * 78)

    seq_dirs = get_sequence_dirs(args.dataset_root)
    if args.sequence:
        seq_dirs = [d for d in seq_dirs if os.path.basename(d) == args.sequence]
    n_seqs = len(seq_dirs)
    if n_seqs == 0:
        print('[ERROR] 没有找到任何序列，请检查数据集路径。')
        sys.exit(1)

    # ── 按帧数升序后贪心分配：短序列先跑快速出结果，总帧数仍均衡 ──
    sorted_dirs = sorted(seq_dirs, key=lambda d: count_frames(d, 'visible'))
    n_workers = min(n_workers, n_seqs)
    chunks: List[List[str]] = [[] for _ in range(n_workers)]
    chunk_frames = [0] * n_workers
    for d in sorted_dirs:
        # 贪心：分配给当前总帧数最少的 worker
        wid = min(range(n_workers), key=lambda i: chunk_frames[i])
        chunks[wid].append(d)
        chunk_frames[wid] += count_frames(d, 'visible')
    chunks = [c for c in chunks if c]
    n_workers = len(chunks)

    print(f'[INFO] 共 {n_seqs} 条序列，启动 {n_workers} 个 worker')
    for i, (c, f) in enumerate(zip(chunks, chunk_frames)):
        print(f'       worker {i}: {len(c)} 条序列, 约 {f} 帧')

    HDR = (f"\n{'#':<7} {'序列名':<30} "
           f"{'AO':>6} {'SS':>6} {'SR50':>6} {'SR75':>6} "
           f"{'PS':>6} {'NPS':>6} {'FPS':>7} {'耗时s':>6} "
           f"{'avgSS':>7} {'avgPS':>7}")
    SEP = '-' * (len(HDR) + 1)
    print(HDR)
    print(SEP)

    out_q: mp.Queue = mp.Queue()
    procs = []
    for wid, chunk in enumerate(chunks):
        # 所有 worker 共享 GPU 0，或使用 CPU
        gpu_id = 0 if num_gpus > 0 else -1
        proc = mp.Process(target=_worker, args=(wid, chunk, result_dir, args.weight, gpu_id, out_q), daemon=True)
        proc.start()
        procs.append(proc)

    print(f'\n[INFO] 等待 {n_workers} 个 worker 加载模型...', flush=True)
    ready = 0
    while ready < n_workers:
        msg = out_q.get()
        if msg[0] == 'ready':
            ready += 1
            print(f'[INFO]   worker {msg[1]} 就绪 ({ready}/{n_workers})', flush=True)
        elif msg[0] == 'init_error':
            print(f'[ERROR]  worker {msg[1]} 加载失败:\n{msg[2]}', flush=True)
            ready += 1
    print(f'[INFO] 所有 worker 就绪，开始追踪...\n', flush=True)

    all_recs: Dict[str, dict] = {}
    done = 0
    t_all = time.perf_counter()
    ss_cum, ps_cum = 0., 0.
    n_valid_cum = 0

    while done < n_seqs:
        msg = out_q.get()
        if msg[0] == 'result':
            _, wid, m = msg
            done += 1
            all_recs[m['seq_name']] = m
            if m['AO'] >= 0:
                ss_cum += m['SS']; ps_cum += m['PS']; n_valid_cum += 1
                avg_ss = ss_cum / n_valid_cum; avg_ps = ps_cum / n_valid_cum
                print(f"[{done:03d}/{n_seqs:03d}] {m['seq_name']:<30s} "
                      f"{m['AO']:6.3f} {m['SS']:6.3f} {m['SR50']:6.3f} {m['SR75']:6.3f} "
                      f"{m['PS']:6.3f} {m['NPS']:6.3f} {m['fps']:7.1f} {m['elapsed']:6.1f} "
                      f"{avg_ss:7.3f} {avg_ps:7.3f}", flush=True)
            else:
                print(f"[{done:03d}/{n_seqs:03d}] {m['seq_name']:<30s} "
                      f"{'---':>6} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'---':>6} "
                      f"{m.get('fps',0.):7.1f} {m.get('elapsed',0.):6.1f} {'---':>7} {'---':>7}", flush=True)
        elif msg[0] == 'seq_error':
            _, wid, seq_name, tb, elapsed = msg
            done += 1
            err_line = tb.strip().splitlines()[-1][:80] if tb.strip().splitlines() else 'unknown'
            print(f"[{done:03d}/{n_seqs:03d}] {seq_name:<32s} [ERROR] {err_line}  ({elapsed:.1f}s) w={wid}", flush=True)
            print(f"  Full traceback:\n{tb}", flush=True)
            all_recs[seq_name] = dict(seq_name=seq_name, AO=-1., SS=-1., SR50=-1., SR75=-1.,
                                       PS=-1., NPS=-1., n_valid=0, fps=0., elapsed=elapsed)

    for proc in procs:
        proc.join(timeout=30)

    t_total = time.perf_counter() - t_all
    print(SEP)
    print(f'\n[INFO] 追踪完成，总耗时 {t_total/60:.1f} 分钟', flush=True)

    # ── 汇总 ──
    recs = list(all_recs.values())
    valid = [r for r in recs if r['AO'] >= 0]

    if valid:
        total_frames = sum(r['n_valid'] for r in valid)
        seq_means: Dict[str, float] = {}
        frm_means: Dict[str, float] = {}
        for k in _METRICS:
            seq_means[k] = float(np.mean([r[k] for r in valid]))
            frm_means[k] = float(sum(r[k]*r['n_valid'] for r in valid)/total_frames) if total_frames>0 else -1.
        mfps = float(np.mean([r['fps'] for r in valid if r['fps']>0]))
    else:
        seq_means = {k: -1. for k in _METRICS}
        frm_means = {k: -1. for k in _METRICS}
        mfps = 0.; total_frames = 0

    W = 9
    print(f'\n{"="*78}')
    print(f"[汇总] {len(valid)}/{len(recs)} 条有效序列  总帧数={total_frames}  平均FPS={mfps:.1f}")
    print(f"{'':12}" + ''.join(f"{k:>{W}}" for k in _METRICS))
    print(f"{'序列均值(%)':<12}" + ''.join(f"{seq_means[k]*100:>{W}.2f}" for k in _METRICS))
    print(f"{'帧加权(%)':<12}" + ''.join(f"{frm_means[k]*100:>{W}.2f}" for k in _METRICS))
    print(f'{"="*78}')
    print(f"[RESULT] " + " ".join(f"{k}={seq_means[k]:.4f}" for k in _METRICS), flush=True)

    # per_seq_metrics.csv
    per_seq_fields = ['seq_name'] + _METRICS + ['n_valid', 'fps', 'elapsed']
    csv_path = os.path.join(result_dir, 'per_seq_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=per_seq_fields)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, '') for k in per_seq_fields})
    print(f'[INFO] per_seq_metrics.csv  → {csv_path}')

    # eval_history.csv
    history_dir = os.path.join(args.save_dir, 'lasher')
    history_csv = os.path.join(history_dir, 'eval_history.csv')
    os.makedirs(history_dir, exist_ok=True)
    hist_fields = (['ckpt_tag','checkpoint','dataset','modality','n_valid','n_total','mean_fps'] +
                   [f'seq_{k}' for k in _METRICS] + [f'frm_{k}' for k in _METRICS])
    hist_row = {
        'ckpt_tag': save_name, 'checkpoint': os.path.basename(args.weight),
        'dataset': 'lasher', 'modality': 'rgbt_concatfuse',
        'n_valid': len(valid), 'n_total': len(recs), 'mean_fps': f'{mfps:.2f}',
    }
    for k in _METRICS:
        hist_row[f'seq_{k}'] = f'{seq_means[k]:.4f}'
        hist_row[f'frm_{k}'] = f'{frm_means[k]:.4f}'
    write_header = not os.path.isfile(history_csv)
    with open(history_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=hist_fields)
        if write_header: w.writeheader()
        w.writerow(hist_row)
    print(f'[INFO] eval_history.csv     → {history_csv}')

    # summary.json
    summary = dict(
        checkpoint=args.weight, dataset='lasher', modality='rgbt_concatfuse',
        n_sequences=len(recs), n_valid=len(valid), mean_fps=mfps,
        seq_means={k: seq_means[k] for k in _METRICS},
        frm_means={k: frm_means[k] for k in _METRICS},
        total_time_min=f'{t_total/60:.1f}',
    )
    summary_path = os.path.join(result_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[INFO] summary.json         → {summary_path}')


if __name__ == '__main__':
    main()
