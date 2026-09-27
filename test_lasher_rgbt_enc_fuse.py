#!/usr/bin/env python3
"""
RGBTSwinTrack — LasHeR RGBT (RGB + TIR) 双模态目标跟踪测试脚本
================================================================
功能:
  1. 同时读取 LasHeR RGB (visible) 和 TIR (infrared) 图像进行测试
  2. 共享骨干网络提取双模态特征，0.5 加权融合
  3. 4 个 spawn 子进程并行，每个独立加载模型
  4. 序列按帧数排序后 round-robin 分配，负载均衡
  5. 每个序列输出详细指标 (AO, SS, SR50, SR75, PS, NPS)
  6. 结果保存为 per_seq_metrics.csv / summary.json / eval_history.csv

用法:
  python test_lasher_rgbt.py --weight /path/to/weight.pth
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

# ── 确保项目根目录在 sys.path ────────────────────────────────────────────
_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

# LasHeR 数据集路径（可通过环境变量覆盖）
_LASHER_ROOT = os.environ.get('LASHER_ROOT', '/home/fzg/data/lasher')
_LASHER_TEST_ROOT = os.path.join(_LASHER_ROOT, 'testingset')

# RGBTSwinTrack Base-384 模型配置
MODEL_CONFIG = {
    'template_size': [192, 192],      # (W, H)
    'search_size':   [384, 384],      # (W, H)
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
    'template_feat_shape': [12, 12],   # (H, W)
    'search_feat_shape':   [24, 24],   # (H, W)
    'untied_abs_pos': True,
    'untied_rel_pos': True,
}

# 评估指标
_SUCCESS_THRESHOLDS = np.linspace(0, 1, 21)
_METRICS = ['AO', 'SS', 'SR50', 'SR75', 'PS', 'NPS']

# ImageNet 归一化
_IMAGENET_MEAN = torch.tensor(IMAGENET_DEFAULT_MEAN)
_IMAGENET_STD  = torch.tensor(IMAGENET_DEFAULT_STD)

# TIR 灰度图归一化 (所有通道统一, 基于 LasHeR 实际统计)
_TIR_MEAN = torch.tensor([0.449, 0.449, 0.449])
_TIR_STD  = torch.tensor([0.226, 0.226, 0.226])

_DEFAULT_WEIGHT = os.path.join(_PRJ_ROOT, 'SwinTrack-B-384.pth')


# ══════════════════════════════════════════════════════════════════════════════
# 图像读取工具
# ══════════════════════════════════════════════════════════════════════════════

def list_frames(seq_dir: str, modal: str) -> List[str]:
    """列出某模态下所有帧文件路径（已排序）。"""
    d = os.path.join(seq_dir, modal)
    if not os.path.isdir(d):
        return []
    files = sorted(
        f for f in os.listdir(d)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )
    return [os.path.join(d, f) for f in files]


def read_frame_rgb(path: str) -> np.ndarray:
    """读取 RGB 图像 (BGR→RGB, uint8, HxWx3)。"""
    img = cv2.imread(path)
    if img is None:
        raise IOError(f'cv2.imread failed: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_frame_tir(path: str) -> np.ndarray:
    """读取 TIR 图像，支持 uint16，统一返回 3 通道 RGB uint8。"""
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
    """读取 LasHeR 标注文件 (x,y,w,h 逗号分隔)。"""
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
    """获取所有序列目录（已排序）。"""
    if not os.path.isdir(root):
        raise FileNotFoundError(f'数据集目录不存在: {root}')
    return sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )


# ══════════════════════════════════════════════════════════════════════════════
# SiamFC Curation（图像裁剪+缩放）
# ══════════════════════════════════════════════════════════════════════════════

def compute_curation_params(bbox_xywh, area_factor, output_size):
    """计算 SiamFC curation 参数。"""
    x, y, w, h = bbox_xywh
    bg = (area_factor - 1) * ((w + h) * 0.5)
    w_z = w + bg
    h_z = h + bg
    scaling = math.sqrt((output_size[0] * output_size[1]) / (w_z * h_z))
    
    source_center = np.array([x + w / 2, y + h / 2], dtype=np.float64)
    target_center = np.array([output_size[0] / 2, output_size[1] / 2], dtype=np.float64)
    
    params = np.zeros((3, 2), dtype=np.float64)
    params[0] = [scaling, scaling]
    params[1] = source_center
    params[2] = target_center
    return params


def crop_and_resize(image: np.ndarray, output_size: Tuple[int, int],
                    curation_params: np.ndarray, image_mean: np.ndarray = None):
    """对图像进行 SiamFC curation（crop + resize）。"""
    H_in, W_in = image.shape[:2]
    W_out, H_out = output_size

    if image_mean is None:
        image_mean = image.astype(np.float32).mean(axis=(0, 1))
    else:
        image_mean = np.asarray(image_mean, dtype=np.float32)

    scaling = curation_params[0]
    source_center = curation_params[1]
    target_center = curation_params[2]

    ox1 = (0 - source_center[0]) * scaling[0] + target_center[0]
    oy1 = (0 - source_center[1]) * scaling[1] + target_center[1]
    ox2 = (W_in - source_center[0]) * scaling[0] + target_center[0]
    oy2 = (H_in - source_center[1]) * scaling[1] + target_center[1]

    ox1_c = max(0, int(np.floor(ox1)))
    oy1_c = max(0, int(np.floor(oy1)))
    ox2_c = min(W_out, int(np.ceil(ox2)))
    oy2_c = min(H_out, int(np.ceil(oy2)))

    if ox2_c > ox1_c and oy2_c > oy1_c:
        ix1_c = (ox1_c - target_center[0]) / scaling[0] + source_center[0]
        iy1_c = (oy1_c - target_center[1]) / scaling[1] + source_center[1]
        ix2_c = (ox2_c - target_center[0]) / scaling[0] + source_center[0]
        iy2_c = (oy2_c - target_center[1]) / scaling[1] + source_center[1]

        ix1_i = max(0, int(np.floor(ix1_c)))
        iy1_i = max(0, int(np.floor(iy1_c)))
        ix2_i = min(W_in, int(np.ceil(ix2_c)))
        iy2_i = min(H_in, int(np.ceil(iy2_c)))

        curated = np.full((H_out, W_out, 3), image_mean, dtype=np.float32)

        if ix2_i > ix1_i and iy2_i > iy1_i:
            input_crop = image[iy1_i:iy2_i, ix1_i:ix2_i].astype(np.float32)
            out_w = ox2_c - ox1_c
            out_h = oy2_c - oy1_c
            resized = cv2.resize(input_crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            curated[oy1_c:oy2_c, ox1_c:ox2_c] = resized
    else:
        curated = np.full((H_out, W_out, 3), image_mean, dtype=np.float32)

    return curated, image_mean


# ══════════════════════════════════════════════════════════════════════════════
# Bounding Box 后处理
# ══════════════════════════════════════════════════════════════════════════════

def map_bbox_to_original(bbox_search_coord, curation_params):
    """将搜索区域坐标系的 bbox 映射回原始图像坐标。"""
    scaling = curation_params[0]
    source_center = curation_params[1]
    target_center = curation_params[2]
    
    def _map_point(px, py):
        ox = (px - target_center[0]) / scaling[0] + source_center[0]
        oy = (py - target_center[1]) / scaling[1] + source_center[1]
        return ox, oy
    
    x1, y1 = _map_point(bbox_search_coord[0], bbox_search_coord[1])
    x2, y2 = _map_point(bbox_search_coord[2], bbox_search_coord[3])
    
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def clamp_bbox(bbox_xywh, img_w, img_h):
    """将 bbox 限制在图像范围内。"""
    x, y, w, h = bbox_xywh
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return [x, y, w, h]


# ══════════════════════════════════════════════════════════════════════════════
# RGBTSwinTrack 模型构建
# ══════════════════════════════════════════════════════════════════════════════

def _load_config(model_type='enc_fuse'):
    from miscellanies.yaml_ops import load_yaml
    cfg_dir = {'enc_fuse': 'Base-384-enc-fuse',
               'gate_sup': 'Base-384-enc-fuse-gate-sup',
               'enc_w_dec_gate_sup': 'Base-384-enc-w-dec-gate-sup',
               'enc_fusemlp': 'Base-384-enc-fusemlp',
               'enc_fusemamba': 'Base-384-enc-fusemamba',
               'enc_fusemamba_temp': 'Base-384-enc-fusemamba-temp',
               'enc_fusemamba_dyn': 'Base-384-enc-fusemamba-dyn',
               'enc_promptlora': 'Base-384-enc-promptlora',
               'enc_promptlora_b': 'Base-384-enc-promptlora-b',
               'enc_promptlora_vtuav': 'Base-384-enc-promptlora-vtuav'}[model_type]
    return load_yaml(os.path.join(_PRJ_ROOT, 'config', 'SwinRGBTrack', cfg_dir, 'config.yaml'))


def _build_model_from_config(config, model_type='enc_fuse'):
    from core.run.event_dispatcher.register import EventRegister
    er = EventRegister('model/')
    if model_type == 'gate_sup':
        from models.methods.SwinRGBTrack.builder_enc_fuse_gate_sup import build_rgbt_enc_fuse_gate_sup
        return build_rgbt_enc_fuse_gate_sup(config, False, 1, 1, er, False)
    if model_type == 'enc_w_dec_gate_sup':
        from models.methods.SwinRGBTrack.builder_enc_w_dec_gate_sup import build_rgbt_enc_w_dec_gate_sup
        return build_rgbt_enc_w_dec_gate_sup(config, False, 1, 1, er, False)
    if model_type == 'enc_fusemlp':
        from models.methods.SwinRGBTrack.builder_enc_fusemlp import build_rgbt_enc_fusemlp
        return build_rgbt_enc_fusemlp(config, False, 1, 1, er, False)
    if model_type == 'enc_fusemamba':
        from models.methods.SwinRGBTrack.builder_enc_fusemamba import build_rgbt_enc_fusemamba
        return build_rgbt_enc_fusemamba(config, False, 1, 1, er, False)
    if model_type == 'enc_fusemamba_temp':
        from models.methods.SwinRGBTrack.builder_enc_fusemamba_temp import build_rgbt_enc_fusemamba_temp
        return build_rgbt_enc_fusemamba_temp(config, False, 1, 1, er, False)
    if model_type == 'enc_fusemamba_dyn':
        from models.methods.SwinRGBTrack.builder_enc_fusemamba_dyn import build_rgbt_enc_fusemamba_dyn
        return build_rgbt_enc_fusemamba_dyn(config, False, 1, 1, er, False)
    if model_type == 'enc_promptlora':
        from models.methods.SwinRGBTrack.builder_enc_promptlora import build_rgbt_enc_promptlora
        return build_rgbt_enc_promptlora(config, False, 1, 1, er, False)
    if model_type == 'enc_promptlora_b':
        from models.methods.SwinRGBTrack.builder_enc_promptlora_b import build_rgbt_enc_promptlora_b
        return build_rgbt_enc_promptlora_b(config, False, 1, 1, er, False)
    if model_type in ('enc_promptlora_vtuav',):
        from models.methods.SwinRGBTrack.builder_enc_promptlora import build_rgbt_enc_promptlora
        return build_rgbt_enc_promptlora(config, False, 1, 1, er, False)
    from models.methods.SwinRGBTrack.builder_enc_fuse import build_rgbt_enc_fuse
    return build_rgbt_enc_fuse(config, False, 1, 1, er, False)


def load_model(weight_path, device, model_type='enc_fuse'):
    print(f'[INFO] Loading RGBTSwinTrack-EncFuse (model_type={model_type})...', flush=True)
    config = _load_config(model_type)
    model = _build_model_from_config(config, model_type)

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
        print(f'[WARN] 缺失 {len(missing)} 个键 (RGBTSwinTrack 新增或未初始化)，将保留初始化值')
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
    """返回单序列指标。"""
    valid = [(p, g) for p, g in zip(preds, gts)
             if len(p) >= 4 and len(g) >= 4 and g[2] > 0 and g[3] > 0]
    if not valid:
        return dict(AO=-1., SS=-1., SR50=-1., SR75=-1.,
                    PS=-1., NPS=-1., n_valid=0)

    ious, dists, nd = [], [], []
    for p, g in valid:
        ious.append(iou(p, g))
        cx_p = p[0] + p[2] / 2;  cy_p = p[1] + p[3] / 2
        cx_g = g[0] + g[2] / 2;  cy_g = g[1] + g[3] / 2
        d = float(np.sqrt((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2))
        dists.append(d)
        nd.append(d / float(np.sqrt(g[2] * g[3])) if g[2] * g[3] > 0 else 0.)

    ia = np.array(ious,  dtype=np.float64)
    da = np.array(dists, dtype=np.float64)
    na = np.array(nd,    dtype=np.float64)

    sr_curve = np.array([(ia >= t).mean() for t in _SUCCESS_THRESHOLDS])
    ss_auc   = float(np.trapz(sr_curve, _SUCCESS_THRESHOLDS))

    return dict(
        AO   = float(ia.mean()),
        SS   = ss_auc,
        SR50 = float((ia >= 0.50).mean()),
        SR75 = float((ia >= 0.75).mean()),
        PS   = float((da <= 20.).mean()),
        NPS  = float((na <= 0.5).mean()),
        n_valid = len(valid),
    )


def save_preds(preds: list, seq_name: str, result_dir: str):
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{seq_name}.txt'), 'w') as f:
        for b in preds:
            f.write(','.join(f'{v:.4f}' for v in b) + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# 子进程 Worker
# ══════════════════════════════════════════════════════════════════════════════

def _worker(worker_id: int,
            seq_dirs: List[str],
            result_dir: str,
            weight_path: str,
            gpu_id: int,
            out_q: mp.Queue,
            model_type: str = 'enc_fuse'):
    """spawn 子进程入口: 独立加载 RGBTSwinTrack 模型，处理分配的序列。"""
    try:
        if gpu_id >= 0 and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
        model = load_model(weight_path, device, model_type)

        cfg = MODEL_CONFIG
        template_size = tuple(cfg['template_size'])
        search_size   = tuple(cfg['search_size'])
        t_sz = template_size[0]; s_sz = search_size[0]
        template_area_factor = cfg['template_area_factor']
        search_area_factor   = cfg['search_area_factor']
        window_penalty = cfg['window_penalty']
        feat_h, feat_w = cfg['search_feat_shape']

        # Hann 窗惩罚
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
            # ── 读取 RGB 和 TIR 序列数据 ──
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
                raise ValueError(
                    f'序列 {seq_name}: 帧数不足 '
                    f'(rgb={n_rgb}, tir={n_tir}, gt={n_gt})')

            preds = []

            init_bbox = list(gt_all[0])
            img0_rgb = read_frame_rgb(frame_paths_rgb[0])
            img0_tir = read_frame_tir(frame_paths_tir[0])
            init_bbox = clamp_bbox(init_bbox, img0_rgb.shape[1], img0_rgb.shape[0])

            # SiamFC curation (与训练预处理一致)
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
            feat_buf = None  # 时序缓冲 (enc_fusemamba_temp 专用)

            for fi in range(1, n):
                img_f_rgb = read_frame_rgb(frame_paths_rgb[fi])
                img_f_tir = read_frame_tir(frame_paths_tir[fi])
                img_h, img_w = img_f_rgb.shape[:2]

                adj_bbox = cached_search_bbox.copy()
                adj_bbox[2] = max(adj_bbox[2], cfg['min_object_size'][0])
                adj_bbox[3] = max(adj_bbox[3], cfg['min_object_size'][1])

                sp = compute_curation_params(adj_bbox, search_area_factor, (s_sz, s_sz))
                x_rgb_img, _ = crop_and_resize(img_f_rgb, (s_sz, s_sz), sp, image_mean=zm_rgb)
                x_tir_img, _ = crop_and_resize(img_f_tir, (s_sz, s_sz), sp, image_mean=zm_tir)

                x_rgb_tensor = torch.from_numpy(x_rgb_img / 255.0).permute(2, 0, 1).float()
                x_rgb_tensor = rgb_norm(x_rgb_tensor).unsqueeze(0).to(device)
                x_tir_tensor = torch.from_numpy(x_tir_img / 255.0).permute(2, 0, 1).float()
                x_tir_tensor = tir_norm(x_tir_tensor).unsqueeze(0).to(device)

                with torch.no_grad():
                    if model_type == 'enc_fusemamba_temp':
                        output, feat_buf = model.track_temp(
                            cached, x_rgb_tensor, x_tir_tensor, feat_buf)
                    else:
                        output = model.track(cached, x_rgb_tensor, x_tir_tensor)

                class_score = output['class_score']; bbox_pred = output['bbox']
                cls_map = class_score.view(1, feat_h * feat_w)
                cls_map = cls_map * (1 - window_penalty) + hann_window.view(1, feat_h * feat_w) * window_penalty
                _, best_idx = torch.max(cls_map, dim=1)
                bbox_flat = bbox_pred.view(1, feat_h * feat_w, 4)
                best_bbox = bbox_flat[0, best_idx[0], :].cpu().numpy()

                # reg 输出为 CXCYWH 归一化, 映射回原图 XYWH
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

            save_preds(preds, seq_name, result_dir)
            m = compute_metrics(preds, gt_all[:n])
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

    p = argparse.ArgumentParser('RGBTSwinTrack-EncFuse LasHeR 测试')
    p.add_argument('--weight', default=_DEFAULT_WEIGHT,
                   help='模型权重文件路径')
    p.add_argument('--dataset_root', default=_LASHER_TEST_ROOT,
                   help='LasHeR 测试集根目录')
    p.add_argument('--save_dir', default=os.path.join(_PRJ_ROOT, 'test_results'),
                   help='结果保存目录')
    p.add_argument('--workers', type=int, default=4,
                   help='并行 worker 进程数')
    p.add_argument('--model_type', type=str, default='enc_fuse',
                   choices=['enc_fuse', 'gate_sup', 'enc_w_dec_gate_sup',
                            'enc_fusemlp', 'enc_fusemamba', 'enc_fusemamba_temp',
                            'enc_fusemamba_dyn', 'enc_promptlora', 'enc_promptlora_b',
                            'enc_promptlora_vtuav'],
                   help='模型变体')
    p.add_argument('--sequence', default='',
                   help='只跑单条序列（调试用）')
    args = p.parse_args()

    # ── GPU 检测 ──
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        print('[WARN] 未检测到 CUDA GPU，将使用 CPU 推理（较慢）')

    n_workers = args.workers

    # ── 结果目录 ──
    save_name = {
        'enc_fuse': 'swintrack_b384_encfuse',
        'gate_sup': 'swintrack_b384_encfuse_gate_sup',
        'enc_w_dec_gate_sup': 'swintrack_b384_encw_dec_gate_sup',
        'enc_fusemlp': 'swintrack_b384_enc_fusemlp',
        'enc_fusemamba': 'swintrack_b384_enc_fusemamba',
        'enc_fusemamba_temp': 'swintrack_b384_enc_fusemamba_temp',
        'enc_fusemamba_dyn': 'swintrack_b384_enc_fusemamba_dyn',
        'enc_promptlora': 'swintrack_b384_enc_promptlora',
        'enc_promptlora_b': 'swintrack_b384_enc_promptlora_b',
        'enc_promptlora_vtuav': 'swintrack_b384_enc_promptlora_vtuav',
    }[args.model_type]
    result_dir = os.path.join(args.save_dir, 'lasher', save_name)
    os.makedirs(result_dir, exist_ok=True)

    print('=' * 78)
    print(f'  权重文件 : {args.weight}')
    print(f'  模态     : RGBT EncFuse (Encoder后融合, model_type={args.model_type})')
    print(f'  数据集   : {args.dataset_root}')
    print(f'  结果目录 : {result_dir}')
    print(f'  GPU 数量 : {num_gpus}')
    print(f'  Workers  : {n_workers}')
    print('=' * 78)

    # ── 收集并过滤序列 ──
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
        wid = min(range(n_workers), key=lambda i: chunk_frames[i])
        chunks[wid].append(d)
        chunk_frames[wid] += count_frames(d, 'visible')
    chunks = [c for c in chunks if c]
    n_workers = len(chunks)

    print(f'[INFO] 共 {n_seqs} 条序列，启动 {n_workers} 个 worker')
    for i, (c, f) in enumerate(zip(chunks, chunk_frames)):
        print(f'       worker {i}: {len(c)} 条序列, 约 {f} 帧')

    # ── 表头 ──
    HDR = (f"\n{'#':<7} {'序列名':<28} "
           f"{'AO':>6} {'SS':>6} {'SR50':>6} {'SR75':>6} "
           f"{'PS':>6} {'NPS':>6} {'FPS':>7} {'耗时s':>6} "
           f"{'avgAO':>7} {'avgSR50':>7} {'avgSS':>7} {'avgPS':>7}")
    SEP = '-' * (len(HDR) + 1)
    print(HDR)
    print(SEP)

    # ── 启动子进程 ──
    out_q: mp.Queue = mp.Queue()
    procs = []
    for wid, chunk in enumerate(chunks):
        gpu_id = 0 if num_gpus > 0 else -1
        proc = mp.Process(
            target=_worker,
            args=(wid, chunk, result_dir, args.weight, gpu_id, out_q, args.model_type),
            daemon=True,
        )
        proc.start()
        procs.append(proc)

    # ── 等待所有 worker 完成模型加载 ──
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

    # ── 收集结果 ──
    all_recs: Dict[str, dict] = {}
    done = 0
    t_all = time.perf_counter()
    ao_cum, ss_cum, ps_cum, sr50_cum = 0., 0., 0., 0.   # 实时累计（运行均值）
    n_valid_cum = 0

    while done < n_seqs:
        msg = out_q.get()
        if msg[0] == 'result':
            _, wid, m = msg
            done += 1
            all_recs[m['seq_name']] = m
            if m['AO'] >= 0:
                ao_cum += m['AO']
                ss_cum += m['SS']
                ps_cum += m['PS']
                sr50_cum += m['SR50']
                n_valid_cum += 1
                avg_ao = ao_cum / n_valid_cum
                avg_ss = ss_cum / n_valid_cum
                avg_ps = ps_cum / n_valid_cum
                avg_sr50 = sr50_cum / n_valid_cum
                print(
                    f"[{done:03d}/{n_seqs:03d}] {m['seq_name']:<28s} "
                    f"{m['AO']:6.3f} {m['SS']:6.3f} "
                    f"{m['SR50']:6.3f} {m['SR75']:6.3f} "
                    f"{m['PS']:6.3f} {m['NPS']:6.3f} "
                    f"{m['fps']:7.1f} {m['elapsed']:6.1f} "
                    f"{avg_ao:7.3f} {avg_sr50:7.3f} "
                    f"{avg_ss:7.3f} {avg_ps:7.3f}",
                    flush=True)
            else:
                print(
                    f"[{done:03d}/{n_seqs:03d}] {m['seq_name']:<28s} "
                    f"{'---':>6} {'---':>6} {'---':>6} {'---':>6} "
                    f"{'---':>6} {'---':>6} "
                    f"{m.get('fps', 0.):7.1f} {m.get('elapsed', 0.):6.1f} "
                    f"{'---':>7} {'---':>7} {'---':>7} {'---':>7}",
                    flush=True)
        elif msg[0] == 'seq_error':
            _, wid, seq_name, tb, elapsed = msg
            done += 1
            err_line = tb.strip().splitlines()[-1][:80] if tb.strip().splitlines() else 'unknown'
            print(
                f"[{done:03d}/{n_seqs:03d}] {seq_name:<32s} "
                f"[ERROR] {err_line}  ({elapsed:.1f}s) w={wid}",
                flush=True)
            print(f"  Full traceback:\n{tb}", flush=True)
            all_recs[seq_name] = dict(
                seq_name=seq_name, AO=-1., SS=-1., SR50=-1., SR75=-1.,
                PS=-1., NPS=-1., n_valid=0, fps=0., elapsed=elapsed)

    for proc in procs:
        proc.join(timeout=30)

    t_total = time.perf_counter() - t_all
    print(SEP)
    print(f'\n[INFO] 追踪完成，总耗时 {t_total / 60:.1f} 分钟', flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 汇总统计
    # ══════════════════════════════════════════════════════════════════════════
    recs = list(all_recs.values())
    valid = [r for r in recs if r['AO'] >= 0]

    if valid:
        total_frames = sum(r['n_valid'] for r in valid)
        seq_means: Dict[str, float] = {}
        frm_means: Dict[str, float] = {}
        for k in _METRICS:
            seq_means[k] = float(np.mean([r[k] for r in valid]))
            frm_means[k] = (
                float(sum(r[k] * r['n_valid'] for r in valid) / total_frames)
                if total_frames > 0 else -1.)
        mfps = float(np.mean([r['fps'] for r in valid if r['fps'] > 0]))
    else:
        seq_means = {k: -1. for k in _METRICS}
        frm_means = {k: -1. for k in _METRICS}
        mfps = 0.
        total_frames = 0

    W = 9
    print(f'\n{"=" * 78}')
    print(f"[汇总] {len(valid)}/{len(recs)} 条有效序列  "
          f"总帧数={total_frames}  平均FPS={mfps:.1f}")
    print(f"{'':12}" + ''.join(f"{k:>{W}}" for k in _METRICS))
    print(f"{'序列均值(%)':<12}" +
          ''.join(f"{seq_means[k]*100:>{W}.2f}" for k in _METRICS))
    print(f"{'帧加权(%)' :<12}" +
          ''.join(f"{frm_means[k]*100:>{W}.2f}" for k in _METRICS))
    print(f'{"=" * 78}')
    print(f"[RESULT] " +
          " ".join(f"{k}={seq_means[k]:.4f}" for k in _METRICS), flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # per_seq_metrics.csv
    # ══════════════════════════════════════════════════════════════════════════
    per_seq_fields = ['seq_name'] + _METRICS + ['n_valid', 'fps', 'elapsed']
    csv_path = os.path.join(result_dir, 'per_seq_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=per_seq_fields)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, '') for k in per_seq_fields})
    print(f'[INFO] per_seq_metrics.csv  → {csv_path}')

    # ══════════════════════════════════════════════════════════════════════════
    # eval_history.csv（追加）
    # ══════════════════════════════════════════════════════════════════════════
    history_dir = os.path.join(args.save_dir, 'lasher')
    history_csv = os.path.join(history_dir, 'eval_history.csv')
    os.makedirs(history_dir, exist_ok=True)

    hist_fields = (
        ['ckpt_tag', 'checkpoint', 'dataset', 'modality', 'n_valid', 'n_total', 'mean_fps'] +
        [f'seq_{k}' for k in _METRICS] +
        [f'frm_{k}' for k in _METRICS]
    )
    hist_row: dict = {
        'ckpt_tag':   save_name,
        'checkpoint': os.path.basename(args.weight),
        'dataset':    'lasher',
        'modality':   {'enc_fuse': 'rgbt_encfuse',
                       'gate_sup': 'rgbt_encfuse_gate_sup',
                       'enc_w_dec_gate_sup': 'rgbt_encw_dec_gate_sup',
                       'enc_fusemlp': 'rgbt_enc_fusemlp',
                       'enc_fusemamba': 'rgbt_enc_fusemamba',
                       'enc_fusemamba_temp': 'rgbt_enc_fusemamba_temp',
                       'enc_fusemamba_dyn': 'rgbt_enc_fusemamba_dyn',
                       'enc_promptlora': 'rgbt_enc_promptlora',
                       'enc_promptlora_b': 'rgbt_enc_promptlora_b',
                       'enc_promptlora_vtuav': 'rgbt_enc_promptlora_vtuav'}[args.model_type],
        'n_valid':    len(valid),
        'n_total':    len(recs),
        'mean_fps':   f'{mfps:.2f}',
    }
    for k in _METRICS:
        hist_row[f'seq_{k}'] = f'{seq_means[k]:.4f}'
        hist_row[f'frm_{k}'] = f'{frm_means[k]:.4f}'

    write_header = not os.path.isfile(history_csv)
    with open(history_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=hist_fields)
        if write_header:
            w.writeheader()
        w.writerow(hist_row)
    print(f'[INFO] eval_history.csv     → {history_csv}')

    # ══════════════════════════════════════════════════════════════════════════
    # summary.json
    # ══════════════════════════════════════════════════════════════════════════
    summary = dict(
        checkpoint=args.weight,
        dataset='lasher',
        modality={'enc_fuse': 'rgbt_encfuse',
                  'gate_sup': 'rgbt_encfuse_gate_sup',
                  'enc_w_dec_gate_sup': 'rgbt_encw_dec_gate_sup',
                  'enc_fusemlp': 'rgbt_enc_fusemlp',
                  'enc_fusemamba': 'rgbt_enc_fusemamba',
                  'enc_fusemamba_temp': 'rgbt_enc_fusemamba_temp',
                  'enc_fusemamba_dyn': 'rgbt_enc_fusemamba_dyn',
                  'enc_promptlora': 'rgbt_enc_promptlora',
                  'enc_promptlora_b': 'rgbt_enc_promptlora_b',
                  'enc_promptlora_vtuav': 'rgbt_enc_promptlora_vtuav'}[args.model_type],
        n_sequences=len(recs),
        n_valid=len(valid),
        mean_fps=mfps,
        seq_means={k: seq_means[k] for k in _METRICS},
        frm_means={k: frm_means[k] for k in _METRICS},
        total_time_min=f'{t_total / 60:.1f}',
    )
    summary_path = os.path.join(result_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[INFO] summary.json         → {summary_path}')


if __name__ == '__main__':
    main()
