#!/usr/bin/env python3
"""
RGBTSwinTrack-EncFuse-V1 — LasHeR 训练 (Encoder 后融合)
=========================================================
Encoder 后融合: Backbone → Encoder(每模态) → 0.5 Fuse → Decoder → Head

相对 train_rgbt_enc_fuse.py 的修复:
  1. TIR 归一化统计对齐测试脚本 ([0.449]/[0.226] 而非 ImageNet)
  2. 学习率调度修复: 原版 cosine 调度器从未 step(), LR 恒为初始值
     → 改为显式 warmup + cosine LambdaLR, 每 epoch 正确衰减
  3. drop_path: has_training_run=False (原版注册的 warmup hook 从未触发,
     导致 drop path 全程为 0)
  4. 新增 --resume 支持
  5. backbone_lr=0 时 backbone 全程冻结 (原版会静默不更新)
  6. 新增 AMP 混合精度
  7. 自动测试改用 sys.executable + subprocess (原版 os.system 脆弱)

用法:
  python train_rgbt_enc_fuse_v1.py --weight /path/to/SwinTrack-B-384.pth \
      --output_dir /path/to/output
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
import random
import numpy as np
import cv2 as cv
import torch
import core.amp_compat  # AMP 兼容层 (服务器旧版 torch 无 torch.amp.GradScaler)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ── 项目根目录 ──
_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)

# ── 归一化统计量 (与 test_lasher_rgbt_enc_fuse.py 严格一致) ──
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_TIR_MEAN = [0.449, 0.449, 0.449]      # LasHeR TIR 实际统计
_TIR_STD = [0.226, 0.226, 0.226]

_MODEL_REGISTRY = {
    'enc_fuse': ('config/SwinRGBTrack/Base-384-enc-fuse-v1',
                 'models.methods.SwinRGBTrack.builder_enc_fuse', 'build_rgbt_enc_fuse'),
    'enc_fusemlp': ('config/SwinRGBTrack/Base-384-enc-fusemlp',
                    'models.methods.SwinRGBTrack.builder_enc_fusemlp', 'build_rgbt_enc_fusemlp'),
    'enc_fusemamba': ('config/SwinRGBTrack/Base-384-enc-fusemamba',
                      'models.methods.SwinRGBTrack.builder_enc_fusemamba', 'build_rgbt_enc_fusemamba'),
    'enc_fusemamba_dyn': ('config/SwinRGBTrack/Base-384-enc-fusemamba-dyn',
                          'models.methods.SwinRGBTrack.builder_enc_fusemamba_dyn', 'build_rgbt_enc_fusemamba_dyn'),
    'enc_promptlora': ('config/SwinRGBTrack/Base-384-enc-promptlora',
                       'models.methods.SwinRGBTrack.builder_enc_promptlora', 'build_rgbt_enc_promptlora'),
    'enc_promptlora_b': ('config/SwinRGBTrack/Base-384-enc-promptlora-b',
                         'models.methods.SwinRGBTrack.builder_enc_promptlora_b', 'build_rgbt_enc_promptlora_b'),
    # VTUAV 训练的 EncPromptLoRA (结构同 enc_promptlora, 仅 config 名不同)
    'enc_promptlora_vtuav': ('config/SwinRGBTrack/Base-384-enc-promptlora-vtuav',
                             'models.methods.SwinRGBTrack.builder_enc_promptlora', 'build_rgbt_enc_promptlora'),
}


# ══════════════════════════════════════════════════════════════════════════════
# LasHeR Siamese 采样 Dataset (SiamFC curation)
# ══════════════════════════════════════════════════════════════════════════════

class LasHeRSiameseDataset(Dataset):
    """SiamFC curation: crop + resize with image mean fill."""

    def __init__(self, root, split, template_size=(192, 192), search_size=(384, 384),
                 template_area_factor=2.0, search_area_factor=4.0,
                 max_frame_gap=200, samples_per_epoch=60000, seed=0):
        from datasets.RGBT.lasher_rgbt import LasHeRRGBTDataset
        self.dataset = LasHeRRGBTDataset(root, split)
        self.split = split
        self.template_size = template_size; self.search_size = search_size
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        self.max_frame_gap = max_frame_gap
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.RandomState(seed)
        self._seq_frame_counts = []
        for i in range(self.dataset.num_sequences):
            info = self.dataset.get_sequence_info(i)
            n = info['valid'].sum().item()
            self._seq_frame_counts.append(max(n, 1))
        self._seq_p = np.array(self._seq_frame_counts, dtype=np.float64)
        self._seq_p /= self._seq_p.sum()
        print(f"LasHeRSiameseDataset({split}): {self.dataset.num_sequences} seqs, "
              f"{samples_per_epoch} samples/epoch")

    def __len__(self):
        return self.samples_per_epoch

    def _read_rgb(self, path):
        img = cv.imread(path)
        if img is None:
            raise IOError(f"Cannot read {path}")
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)

    def _read_tir(self, path):
        img = cv.imread(path, cv.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read {path}")
        if img.dtype == np.uint16:
            mn, mx = img.min(), img.max()
            img = ((img.astype(np.float32) - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8) if mx > mn else np.zeros_like(img, dtype=np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        return img

    def _compute_curation_params(self, bbox_xywh, area_factor, output_size):
        x, y, w, h = bbox_xywh
        bg = (area_factor - 1) * ((w + h) * 0.5)
        w_z = w + bg; h_z = h + bg
        scaling = math.sqrt((output_size[0] * output_size[1]) / max(w_z * h_z, 1e-6))
        source_center = np.array([x + w / 2, y + h / 2], dtype=np.float64)
        target_center = np.array([output_size[0] / 2, output_size[1] / 2], dtype=np.float64)
        params = np.zeros((3, 2), dtype=np.float64)
        params[0] = [scaling, scaling]; params[1] = source_center; params[2] = target_center
        return params

    def _crop_and_resize(self, image, output_size, curation_params, image_mean=None):
        H_in, W_in = image.shape[:2]; W_out, H_out = output_size
        if image_mean is None:
            image_mean = image.astype(np.float32).mean(axis=(0, 1))
        else:
            image_mean = np.asarray(image_mean, dtype=np.float32)
        scaling = curation_params[0]; sc = curation_params[1]; tc = curation_params[2]
        ox1 = (0 - sc[0]) * scaling[0] + tc[0]; oy1 = (0 - sc[1]) * scaling[1] + tc[1]
        ox2 = (W_in - sc[0]) * scaling[0] + tc[0]; oy2 = (H_in - sc[1]) * scaling[1] + tc[1]
        ox1_c = max(0, int(np.floor(ox1))); oy1_c = max(0, int(np.floor(oy1)))
        ox2_c = min(W_out, int(np.ceil(ox2))); oy2_c = min(H_out, int(np.ceil(oy2)))
        curated = np.full((H_out, W_out, 3), image_mean, dtype=np.float32)
        if ox2_c > ox1_c and oy2_c > oy1_c:
            ix1_c = (ox1_c - tc[0]) / scaling[0] + sc[0]
            iy1_c = (oy1_c - tc[1]) / scaling[1] + sc[1]
            ix2_c = (ox2_c - tc[0]) / scaling[0] + sc[0]
            iy2_c = (oy2_c - tc[1]) / scaling[1] + sc[1]
            ix1_i = max(0, int(np.floor(ix1_c))); iy1_i = max(0, int(np.floor(iy1_c)))
            ix2_i = min(W_in, int(np.ceil(ix2_c))); iy2_i = min(H_in, int(np.ceil(iy2_c)))
            if ix2_i > ix1_i and iy2_i > iy1_i:
                input_crop = image[iy1_i:iy2_i, ix1_i:ix2_i].astype(np.float32)
                out_w = ox2_c - ox1_c; out_h = oy2_c - oy1_c
                resized = cv.resize(input_crop, (out_w, out_h), interpolation=cv.INTER_LINEAR)
                curated[oy1_c:oy2_c, ox1_c:ox2_c] = resized
        return curated, image_mean

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self._getitem_impl()
            except (OSError, IOError, ValueError):
                continue
        raise RuntimeError("Failed after 10 retries")

    def _getitem_impl(self):
        for _retry in range(50):
            seq_id = int(self.rng.choice(self.dataset.num_sequences, p=self._seq_p))
            info = self.dataset.get_sequence_info(seq_id)
            bboxes = info['bbox']; valid = info['valid'].numpy(); n_frames = len(bboxes)
            valid_ids = np.where(valid)[0]
            if n_frames < 20 or len(valid_ids) < 2:
                continue

            # causal sampling
            search_frame_ids = None; gap_increase = 0
            while search_frame_ids is None and gap_increase <= 500:
                min_base = 1; max_base = n_frames - 1
                bc = valid_ids[(valid_ids >= min_base) & (valid_ids <= max_base)]
                if len(bc) == 0:
                    gap_increase += 5; continue
                base_id = int(self.rng.choice(bc))
                pmin = max(0, base_id - self.max_frame_gap - gap_increase)
                pc = valid_ids[(valid_ids >= pmin) & (valid_ids < base_id)]
                if len(pc) == 0:
                    gap_increase += 5; continue
                template_id = int(self.rng.choice(pc))
                smax = min(n_frames, template_id + self.max_frame_gap + gap_increase + 1)
                sc = valid_ids[(valid_ids > template_id) & (valid_ids < smax)]
                if len(sc) == 0:
                    gap_increase += 5; continue
                search_frame_ids = [int(self.rng.choice(sc))]
            if search_frame_ids is None:
                continue
            search_id = search_frame_ids[0]

            try:
                tp1, tp2 = self.dataset.get_frame_paths(seq_id, template_id)
                sp1, sp2 = self.dataset.get_frame_paths(seq_id, search_id)
                t_rgb = self._read_rgb(tp1); t_tir = self._read_tir(tp2)
                s_rgb = self._read_rgb(sp1); s_tir = self._read_tir(sp2)
            except (OSError, IOError):
                continue

            t_bbox = bboxes[template_id].numpy().tolist()
            s_bbox = bboxes[search_id].numpy().tolist()

            try:
                # Template (no jitter)
                tp = self._compute_curation_params(t_bbox, self.template_area_factor, self.template_size)
                z_rgb, zm_rgb = self._crop_and_resize(t_rgb, self.template_size, tp)
                z_tir, zm_tir = self._crop_and_resize(t_tir, self.template_size, tp)

                # Search jitter: DIF-Net 式 scale=0.35, center=3.0
                # (对齐 train_rgbt_dec_fuse.py, 目标可出现在搜索区任意位置)
                sj = s_bbox.copy()
                js = math.exp(self.rng.randn() * 0.35)
                sj[2] = max(s_bbox[2] * js, 10); sj[3] = max(s_bbox[3] * js, 10)
                mo = math.sqrt(sj[2] * sj[3]) * 3.0
                sj[0] = s_bbox[0] + s_bbox[2] / 2 + mo * (self.rng.rand() - 0.5) - sj[2] / 2
                sj[1] = s_bbox[1] + s_bbox[3] / 2 + mo * (self.rng.rand() - 0.5) - sj[3] / 2
                sp = self._compute_curation_params(sj, self.search_area_factor, self.search_size)
                x_rgb, _ = self._crop_and_resize(s_rgb, self.search_size, sp, image_mean=zm_rgb)
                x_tir, _ = self._crop_and_resize(s_tir, self.search_size, sp, image_mean=zm_tir)

                # Map GT bbox to crop (SiamFC style)
                scaling_s = sp[0]; sc_s = sp[1]; tc_s = sp[2]
                cx = s_bbox[0] + s_bbox[2] / 2; cy = s_bbox[1] + s_bbox[3] / 2
                ocx = (cx - sc_s[0]) * scaling_s[0] + tc_s[0]
                ocy = (cy - sc_s[1]) * scaling_s[1] + tc_s[1]
                ow = s_bbox[2] * scaling_s[0]; oh = s_bbox[3] * scaling_s[1]
                W_out, H_out = self.search_size
                bbox_gt = [ocx / W_out, ocy / H_out, ow / W_out, oh / H_out]
                bbox_gt = [max(0.0, min(1.0, v)) for v in bbox_gt]
            except (ValueError, Exception):
                continue

            z_rgb_t = torch.from_numpy(z_rgb / 255.0).permute(2, 0, 1).float()
            z_tir_t = torch.from_numpy(z_tir / 255.0).permute(2, 0, 1).float()
            x_rgb_t = torch.from_numpy(x_rgb / 255.0).permute(2, 0, 1).float()
            x_tir_t = torch.from_numpy(x_tir / 255.0).permute(2, 0, 1).float()

            rgb_norm = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
            tir_norm = transforms.Normalize(mean=_TIR_MEAN, std=_TIR_STD)

            if self.split == 'train':
                js_state = self.rng.randint(0, 2 ** 31)
                flip_rand = self.rng.rand(); gray_rand = self.rng.rand()
                cj = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
                torch.manual_seed(js_state)
                if gray_rand < 0.05:
                    z_rgb_t = z_rgb_t.mean(dim=0, keepdim=True).expand(3, -1, -1)
                else:
                    z_rgb_t = cj(z_rgb_t)
                torch.manual_seed(js_state)
                if gray_rand < 0.05:
                    x_rgb_t = x_rgb_t.mean(dim=0, keepdim=True).expand(3, -1, -1)
                else:
                    x_rgb_t = cj(x_rgb_t)
                if flip_rand < 0.5:
                    z_rgb_t = torch.flip(z_rgb_t, [-1]); z_tir_t = torch.flip(z_tir_t, [-1])
                    x_rgb_t = torch.flip(x_rgb_t, [-1]); x_tir_t = torch.flip(x_tir_t, [-1])
                    bbox_gt[0] = 1.0 - bbox_gt[0]

            z_rgb_t = rgb_norm(z_rgb_t); z_tir_t = tir_norm(z_tir_t)
            x_rgb_t = rgb_norm(x_rgb_t); x_tir_t = tir_norm(x_tir_t)
            bbox_gt_t = torch.tensor(bbox_gt, dtype=torch.float32)
            return z_rgb_t, z_tir_t, x_rgb_t, x_tir_t, bbox_gt_t

        raise RuntimeError("Failed to sample after 50 retries")


# ══════════════════════════════════════════════════════════════════════════════
# 损失函数
# ══════════════════════════════════════════════════════════════════════════════

def giou_loss(pred, target, reduction='mean'):
    px1 = pred[:, 0] - pred[:, 2] / 2; py1 = pred[:, 1] - pred[:, 3] / 2
    px2 = pred[:, 0] + pred[:, 2] / 2; py2 = pred[:, 1] + pred[:, 3] / 2
    tx1 = target[:, 0] - target[:, 2] / 2; ty1 = target[:, 1] - target[:, 3] / 2
    tx2 = target[:, 0] + target[:, 2] / 2; ty2 = target[:, 1] + target[:, 3] / 2
    ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    pa = (px2 - px1) * (py2 - py1); ta = (tx2 - tx1) * (ty2 - ty1)
    union = pa + ta - inter
    iou = inter / (union + 1e-6)
    ex1 = torch.min(px1, tx1); ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2); ey2 = torch.max(py2, ty2)
    ea = (ex2 - ex1) * (ey2 - ey1)
    giou = iou - (ea - union) / (ea + 1e-6)
    loss = 1 - giou
    return loss.mean() if reduction == 'mean' else loss


def varifocal_loss(pred, target, alpha=0.8, gamma=3.0):
    pred = pred.float()
    target = target.float()
    pred_sigmoid = pred.clamp(1e-6, 1 - 1e-6)
    weight = target + alpha * pred_sigmoid.pow(gamma) * (1 - target)
    loss = F.binary_cross_entropy(pred_sigmoid, target, reduction='none')
    loss = weight * loss
    n_pos = max(target.sum(), 1.0)
    return loss.sum() / n_pos


def compute_iou(bbox_pred, bbox_gt):
    px1 = bbox_pred[:, 0] - bbox_pred[:, 2] / 2; py1 = bbox_pred[:, 1] - bbox_pred[:, 3] / 2
    px2 = bbox_pred[:, 0] + bbox_pred[:, 2] / 2; py2 = bbox_pred[:, 1] + bbox_pred[:, 3] / 2
    gx1 = bbox_gt[:, 0] - bbox_gt[:, 2] / 2; gy1 = bbox_gt[:, 1] - bbox_gt[:, 3] / 2
    gx2 = bbox_gt[:, 0] + bbox_gt[:, 2] / 2; gy2 = bbox_gt[:, 1] + bbox_gt[:, 3] / 2
    ix1 = torch.max(px1, gx1); iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2); iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_p = (px2 - px1) * (py2 - py1)
    area_g = (gx2 - gx1) * (gy2 - gy1)
    union = area_p + area_g - inter + 1e-6
    return inter / union


def compute_targets(bbox_gt, feat_size, search_size):
    B = bbox_gt.shape[0]; H, W = feat_size; SW, SH = search_size
    gcx = bbox_gt[:, 0] * SW; gcy = bbox_gt[:, 1] * SH
    gw = bbox_gt[:, 2] * SW; gh = bbox_gt[:, 3] * SH
    y = torch.arange(H, device=bbox_gt.device).float()
    x = torch.arange(W, device=bbox_gt.device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    px = (xx + 0.5) * (SW / W); py = (yy + 0.5) * (SH / H)
    gx1 = gcx - gw / 2; gy1 = gcy - gh / 2
    gx2 = gcx + gw / 2; gy2 = gcy + gh / 2
    px = px.unsqueeze(0); py = py.unsqueeze(0)
    pos_mask = (px >= gx1.view(B, 1, 1)) & (px < gx2.view(B, 1, 1)) & \
               (py >= gy1.view(B, 1, 1)) & (py < gy2.view(B, 1, 1))
    cls_target = pos_mask.float()
    reg_target = bbox_gt.view(B, 1, 1, 4).expand(-1, H, W, -1).contiguous()
    return cls_target, reg_target


# ══════════════════════════════════════════════════════════════════════════════
# 模型构建
# ══════════════════════════════════════════════════════════════════════════════

def build_model_enc_fuse(model_type='enc_fuse'):
    from importlib import import_module
    from core.run.event_dispatcher.register import EventRegister
    from miscellanies.yaml_ops import load_yaml
    cfg_dir, module_name, func_name = _MODEL_REGISTRY[model_type]
    config = load_yaml(os.path.join(_PRJ_ROOT, cfg_dir, 'config.yaml'))
    builder = getattr(import_module(module_name), func_name)
    er = EventRegister('model/')
    # has_training_run=False: 独立训练循环不触发 drop_path warmup hook,
    # 保持 drop_path 静态速率 (linspace 0→0.1)
    return builder(config, False, 1, 1, er, False)


def load_state_dict_filtered(model, state_dict):
    """按 key+shape 过滤加载权重, 返回 (missing, skipped) 统计。"""
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        elif k in model_state:
            skipped.append(k)
    model.load_state_dict(filtered, strict=False)
    missing = set(model_state.keys()) - set(filtered.keys())
    return missing, skipped


# ══════════════════════════════════════════════════════════════════════════════
# 训练主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser('RGBTSwinTrack-EncFuse-V1 LasHeR 训练')
    parser.add_argument('--weight', type=str, required=True, help='预训练权重路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--lasher_root', type=str,
                        default=os.environ.get('LASHER_ROOT', '/home/fzg/data/lasher'))
    parser.add_argument('--lasher_test_root', type=str, default='',
                        help='LasHeR 测试集根目录 (auto_test 用); 留空则自动在 '
                             'lasher_root 下寻找 testingset 或 test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--samples_per_epoch', type=int, default=60000)
    parser.add_argument('--lr', type=float, default=5e-5, help='encoder/decoder/head 学习率')
    parser.add_argument('--backbone_lr', type=float, default=1e-5, help='backbone 学习率 (0=全程冻结)')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=3)
    parser.add_argument('--freeze_stem_epochs', type=int, default=0,
                        help='前N轮冻结 backbone+encoder+decoder, 只训融合层+head (热启动专用, 0=关闭)')
    parser.add_argument('--freeze_stem_exclude_decoder', action='store_true',
                        help='freeze_stem 期间保持 decoder 可训练 (decoder+head 微调专用)')
    parser.add_argument('--freeze_fusion_epochs', type=int, default=0,
                        help='前N轮冻结融合模块 (mamba/mlp/gate), 0=关闭')
    parser.add_argument('--freeze_head', action='store_true',
                        help='全程冻结 head (只训新增/融合模块, 消融实验专用)')
    parser.add_argument('--const_lr', action='store_true',
                        help='恒定学习率 (关闭 warmup+cosine, 微调专用)')
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default='', help='从 checkpoint 恢复训练')
    parser.add_argument('--reinit_spatial', action='store_true',
                        help='热启动后重初始化空间融合 in_proj (修复历史死链, out_proj 归零)')
    parser.add_argument('--revive_context', action='store_true',
                        help='只重初始化 in_proj 保留 out_proj (复活上下文且输出不变, 死链 checkpoint 专用)')
    parser.add_argument('--kill_context', action='store_true',
                        help='复刻历史死链: in_proj/out_proj.weight 全零, 仅 out_proj.bias 可学习')
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='amp')
    parser.add_argument('--auto_test', action='store_true', default=True,
                        help='每5轮自动运行 LasHeR 全量测试')
    parser.add_argument('--no_auto_test', action='store_false', dest='auto_test')
    parser.add_argument('--test_interval', type=int, default=5, help='自动测试间隔 (epoch)')
    parser.add_argument('--test_timeout', type=int, default=1800,
                        help='单次自动全量测试超时秒数 (默认1800=30min; 每轮测试建议调大)')
    parser.add_argument('--test_workers', type=int, default=4,
                        help='自动全量测试的并行 worker 数')
    parser.add_argument('--model_type', type=str, default='enc_fuse',
                        choices=list(_MODEL_REGISTRY.keys()),
                        help='enc_fuse=0.5融合; enc_fusemlp=Concat-Linear; enc_fusemamba=Mamba残差')
    args = parser.parse_args()

    # 解析 LasHeR 测试集根目录 (auto_test 用): 显式指定优先, 否则在 lasher_root
    # 下自动找 testingset 或 test (兼容本地与服务器官方 train/test 布局)
    from datasets.RGBT.lasher_rgbt import _resolve_split_root
    if args.lasher_test_root and os.path.isdir(args.lasher_test_root):
        test_root = args.lasher_test_root
    else:
        test_root = _resolve_split_root(args.lasher_root, 'val')
    print(f"LasHeR test root: {test_root}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Output dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # ── 构建模型 ──
    print(f"Building RGBTSwinTrack model (model_type={args.model_type})...")
    model = build_model_enc_fuse(args.model_type)

    start_epoch = 1
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        load_state_dict_filtered(model, ckpt['model'])
        start_epoch = ckpt.get('epoch', 1) + 1
    else:
        print(f"Loading pretrained weights: {args.weight}")
        checkpoint = torch.load(args.weight, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        missing, skipped = load_state_dict_filtered(model, state_dict)
        if missing:
            print(f"[WARN] 缺失 {len(missing)} 个键, 保留初始化值")
        if skipped:
            print(f"[WARN] 跳过 {len(skipped)} 个形状不匹配的键")
    model.to(device)

    if args.reinit_spatial:
        if hasattr(model, 'reinit_spatial'):
            model.reinit_spatial()
            print('[INFO] 空间融合已重初始化 (in_proj 随机, out_proj 归零, 死链已修复)')
        else:
            print(f'[WARN] --reinit_spatial 对 model_type={args.model_type} 无效 (无 reinit_spatial 方法)')

    if args.revive_context:
        if hasattr(model, 'revive_context'):
            model.revive_context()
            print('[INFO] mamba 上下文已复活 (in_proj 随机, out_proj 保留, 输出与热启动一致)')
        else:
            print(f'[WARN] --revive_context 对 model_type={args.model_type} 无效 (无 revive_context 方法)')

    if args.kill_context:
        if hasattr(model, 'kill_context'):
            model.kill_context()
            print('[INFO] mamba 上下文已杀死 (in_proj/out_proj.weight=0, 仅 out_proj.bias 可学, 复刻历史死链机制)')
        else:
            print(f'[WARN] --kill_context 对 model_type={args.model_type} 无效 (无 kill_context 方法)')

    # ── 数据集 ──
    train_dataset = LasHeRSiameseDataset(
        root=args.lasher_root, split='train',
        samples_per_epoch=args.samples_per_epoch, seed=args.seed)
    val_dataset = LasHeRSiameseDataset(
        root=args.lasher_root, split='val',
        samples_per_epoch=4000, seed=args.seed + 1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=False,  # WSL2 下 cudaHostAlloc 易 OOM
                              drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=False,  # WSL2 下禁用
                            drop_last=False,
                            persistent_workers=args.workers > 0)

    # ── 优化器 ──
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [{'params': other_params, 'lr': args.lr}], weight_decay=1e-4)
    if args.backbone_lr > 0 and len(backbone_params) > 0:
        optimizer.add_param_group({'params': backbone_params, 'lr': args.backbone_lr})
        backbone_trainable = True
    else:
        # backbone_lr=0 → 全程冻结 backbone
        backbone_trainable = False
        for param in backbone_params:
            param.requires_grad = False
        args.freeze_backbone_epochs = args.epochs + 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                print("[INFO] Optimizer state restored")
            except Exception as e:
                print(f"[WARN] Optimizer state restore failed: {e}")

    # ── 学习率调度: linear warmup + cosine decay ──
    warmup = max(0, args.warmup_epochs)
    total = max(1, args.epochs - warmup)
    lr_min_frac = 1e-3

    def lr_factor(epoch):
        # epoch 从 1 开始
        if args.const_lr:
            return 1.0
        if warmup > 0 and epoch <= warmup:
            return epoch / max(warmup, 1)
        progress = min(1.0, (epoch - warmup) / total)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return lr_min_frac + (1 - lr_min_frac) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: lr_factor(ep + 1))  # LambdaLR epoch 从 0 开始

    # resume 续训: checkpoint 未存 scheduler 状态, 需把 LR 调度器快进到
    # start_epoch, 否则 LR 会从 warmup 重新开始 (过高破坏微调稳定性)
    if start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"[INFO] LR scheduler synced to epoch {start_epoch} "
              f"(LR={optimizer.param_groups[0]['lr']:.2e})")

    # init_scale=2^10: 空间 mamba 通路在默认 2^16 下 fp16 反向梯度溢出产生 NaN
    # (mamba 段已在网络内部固定 fp32 计算, 但其余部分仍走 fp16)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda',
                                  init_scale=2 ** 10)

    feat_h, feat_w = 24, 24
    search_w, search_h = 384, 384

    print(f"Freeze backbone: {args.freeze_backbone_epochs} epochs"
          f" (trainable={'yes' if backbone_trainable else 'no'})")
    print(f"Warmup: {warmup}, Cosine over {total} epochs")
    print(f"Backbone LR: {args.backbone_lr if backbone_trainable else 0}, Other LR: {args.lr}")
    print(f"Training {start_epoch}-{args.epochs} epochs, "
          f"{len(train_loader)} iters/epoch, AMP={args.amp}")

    # ════════════════════════════════════════════════════════════════════════
    # 训练循环
    # ════════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, args.epochs + 1):
        # ── 冻结策略: stem 冻结优先, 期间 backbone 保持冻结 ──
        stem_freeze = args.freeze_stem_epochs > 0 and epoch <= args.freeze_stem_epochs
        bb_freeze = (stem_freeze
                     or (backbone_trainable
                         and args.freeze_backbone_epochs > 0
                         and epoch <= args.freeze_backbone_epochs))

        # ── Stem 冻结: backbone+encoder (+decoder 默认) 冻结 ──
        if args.freeze_stem_epochs > 0:
            for name, param in model.named_parameters():
                # enc_promptlora 新增的 prompt/LoRA 参数虽然挂在 encoder 命名空间下,
                # 但不属于 stem, 需要排除在外, 否则会被 stem 冻结误伤。
                if 'lora_A' in name or 'lora_B' in name or name.startswith('prompt_t2r') or name.startswith('prompt_r2t'):
                    continue
                if 'backbone' in name or 'encoder' in name:
                    param.requires_grad = not stem_freeze
                if 'decoder' in name and not args.freeze_stem_exclude_decoder:
                    param.requires_grad = not stem_freeze
            if epoch == start_epoch:
                if args.freeze_stem_exclude_decoder:
                    print("Stem frozen (backbone+encoder), decoder stays trainable")
                else:
                    print(f"Stem frozen for first {args.freeze_stem_epochs} epochs "
                          f"(fusion+head only)")
            if epoch == args.freeze_stem_epochs + 1:
                print(f"Stem UNFROZEN at epoch {epoch}")

        # ── 融合模块冻结: 只训 decoder+head 以适配融合输出分布 (逃生门) ──
        if args.freeze_fusion_epochs > 0:
            fus_freeze = epoch <= args.freeze_fusion_epochs
            fusion_keys = ('in_proj', 'out_proj', 'mamba_fwd', 'mamba_bwd',
                           'fuse_proj', 'gate_mlp')
            for name, param in model.named_parameters():
                if any(k in name for k in fusion_keys):
                    param.requires_grad = not fus_freeze
            if epoch == start_epoch:
                n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Fusion frozen for first {args.freeze_fusion_epochs} epochs, "
                      f"trainable params: {n_train:,}")

        # ── Head 冻结 (消融实验: 隔离新增模块的贡献) ──
        if args.freeze_head:
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    param.requires_grad = False
            if epoch == start_epoch:
                n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Head frozen (--freeze_head), trainable params: {n_train:,}")

        # ── Backbone 冻结/解冻 (stem 冻结期间强制冻结) ──
        if backbone_trainable and args.freeze_backbone_epochs > 0:
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = not bb_freeze
            if epoch == start_epoch:
                if stem_freeze:
                    print(f"Backbone frozen (stem freeze active, "
                          f"{args.freeze_stem_epochs} epochs)")
                else:
                    print(f"Backbone frozen for first {args.freeze_backbone_epochs} epochs")
            if not stem_freeze and epoch == args.freeze_backbone_epochs + 1:
                print(f"Backbone UNFROZEN at epoch {epoch} (lr={args.backbone_lr})")

        model.train()
        et = ecl = egl = eio = epr = 0.

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, (z_rgb, z_tir, x_rgb, x_tir, bbox_gt) in enumerate(pbar):
            z_rgb = z_rgb.to(device, non_blocking=True)
            z_tir = z_tir.to(device, non_blocking=True)
            x_rgb = x_rgb.to(device, non_blocking=True)
            x_tir = x_tir.to(device, non_blocking=True)
            bbox_gt = bbox_gt.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.amp and device.type == 'cuda'):
                output = model(z_rgb, z_tir, x_rgb, x_tir)

            # ── loss 在 autocast 之外以 fp32 计算 (BCE 对 autocast 不安全) ──
            cls_pred = output['class_score'].float()
            reg_pred = output['bbox'].float()
            B = cls_pred.shape[0]

            cls_target, reg_target = compute_targets(bbox_gt, (feat_h, feat_w), (search_w, search_h))
            pos_mask = cls_target > 0.5
            n_pos = pos_mask.sum().item()
            pos_ratio = n_pos / (B * feat_h * feat_w)

            cls_pred_f = cls_pred.view(B, -1)
            cls_target_f = cls_target.view(B, -1)
            if n_pos > 0:
                reg_pred_flat = reg_pred.view(B, feat_h * feat_w, 4)
                reg_target_flat = reg_target.view(B, feat_h * feat_w, 4)
                pos_mask_f = pos_mask.view(B, -1)
                iou_values = compute_iou(reg_pred_flat[pos_mask_f], reg_target_flat[pos_mask_f])
                cls_target_f = cls_target_f.clone()
                cls_target_f[pos_mask_f] = iou_values.detach()
                cls_pred_prob = cls_pred_f.clamp(1e-6, 1 - 1e-6)
                cls_loss = 2.0 * varifocal_loss(cls_pred_prob, cls_target_f)
                gl = 2.0 * giou_loss(reg_pred_flat[pos_mask_f], reg_target_flat[pos_mask_f])
                io = 1.0 - gl.item() / 2.0
            else:
                cls_pred_prob = cls_pred_f.clamp(1e-6, 1 - 1e-6)
                cls_loss = 2.0 * varifocal_loss(cls_pred_prob, cls_target_f)
                gl = torch.tensor(0.0, device=device); io = 0.

            loss = cls_loss + gl

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            et += loss.item(); ecl += cls_loss.item(); egl += gl.item()
            eio += io; epr += pos_ratio
            if (batch_idx + 1) % 50 == 0:
                n = batch_idx + 1
                pbar.set_description(
                    f'Epoch {epoch}/{args.epochs} '
                    f'loss={et/n:.4f} cls={ecl/n:.4f} GIoU={egl/n:.4f} '
                    f'IoU={eio/n:.3f} Pos%={epr/n*100:.1f}')

        scheduler.step()
        n_batches = len(train_loader)
        print(f'Epoch {epoch}/{args.epochs} | '
              f'Loss:{et/n_batches:.4f} Cls:{ecl/n_batches:.4f} '
              f'GIoU:{egl/n_batches:.4f} IoU:{eio/n_batches:.3f} '
              f'Pos%:{epr/n_batches*100:.1f} LR:{optimizer.param_groups[0]["lr"]:.2e}')

        # ── 验证 ──
        model.eval()
        vt = vcl = vgl = vio = vpr = 0.
        with torch.no_grad():
            for z_rgb, z_tir, x_rgb, x_tir, bbox_gt in val_loader:
                z_rgb = z_rgb.to(device); z_tir = z_tir.to(device)
                x_rgb = x_rgb.to(device); x_tir = x_tir.to(device)
                bbox_gt = bbox_gt.to(device)
                with torch.amp.autocast('cuda', enabled=args.amp and device.type == 'cuda'):
                    output = model(z_rgb, z_tir, x_rgb, x_tir)
                # loss 计算在 autocast 之外, 显式 fp32
                cls_pred = output['class_score'].float()
                reg_pred = output['bbox'].float()
                B = cls_pred.shape[0]
                cls_target, reg_target = compute_targets(bbox_gt, (feat_h, feat_w), (search_w, search_h))
                cls_pred_f = cls_pred.view(B, -1); cls_target_f = cls_target.view(B, -1)
                pm = cls_target > 0.5; np_ = pm.sum().item()
                rp = np_ / (B * feat_h * feat_w)
                if np_ > 0:
                    reg_pred_f = reg_pred.view(B, feat_h * feat_w, 4)
                    reg_target_f = reg_target.view(B, feat_h * feat_w, 4)
                    pm_f = pm.view(B, -1)
                    iou_vals = compute_iou(reg_pred_f[pm_f], reg_target_f[pm_f])
                    cls_target_f = cls_target_f.clone()
                    cls_target_f[pm_f] = iou_vals.detach()
                    cls_pred_prob = cls_pred_f.clamp(1e-6, 1 - 1e-6)
                    cl = 2.0 * varifocal_loss(cls_pred_prob, cls_target_f)
                    gl = 2.0 * giou_loss(reg_pred_f[pm_f], reg_target_f[pm_f])
                    io = 1.0 - gl.item() / 2.0
                else:
                    cls_pred_prob = cls_pred_f.clamp(1e-6, 1 - 1e-6)
                    cl = 2.0 * varifocal_loss(cls_pred_prob, cls_target_f)
                    gl = torch.tensor(0.0, device=device); io = 0.
                vt += (cl + gl).item(); vcl += cl.item(); vgl += gl.item()
                vio += io; vpr += rp
        n_val = len(val_loader)
        print(f'  Val | Loss:{vt/n_val:.4f} Cls:{vcl/n_val:.4f} '
              f'GIoU:{vgl/n_val:.4f} IoU:{vio/n_val:.3f} Pos%:{vpr/n_val*100:.1f}')

        # ── 保存 checkpoint ──
        ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:03d}.pth')
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, ckpt_path)
        print(f'  Checkpoint saved: {ckpt_path}')

        # ── 自动测试 (实时透传子进程输出, 可看到逐序列进度) ──
        if args.auto_test and epoch % args.test_interval == 0:
            test_result_dir = os.path.join(args.output_dir, 'test_results', f'epoch{epoch:03d}')
            test_py = os.path.join(_PRJ_ROOT, 'test_lasher_rgbt_enc_fuse.py')
            print(f'[AUTO_TEST] epoch{epoch:03d} 启动测试...', flush=True)
            t0 = time.time()
            cmd = [sys.executable, '-u', test_py,
                   '--weight', ckpt_path,
                   '--model_type', args.model_type,
                   '--dataset_root', test_root,
                   '--save_dir', test_result_dir,
                   '--workers', str(args.test_workers)]
            ret_code = None
            try:
                # Popen + 逐行实时打印: 训练日志能看到每条序列结果, 且不丢缓冲
                proc = subprocess.Popen(
                    cmd, cwd=_PRJ_ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(f'  {line}', end='', flush=True)
                proc.wait(timeout=args.test_timeout)
                ret_code = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f'[AUTO_TEST] epoch{epoch:03d} 超时 '
                      f'(>{args.test_timeout}s), 已终止', flush=True)
                ret_code = -1
            print(f'[AUTO_TEST] epoch{epoch:03d} 完成, '
                  f'耗时 {(time.time()-t0)/60:.1f}min, 退出码={ret_code}', flush=True)
            if ret_code not in (0, None):
                print(f'[AUTO_TEST] epoch{epoch:03d} 返回非零退出码 '
                      f'{ret_code}, 详见上方测试日志', flush=True)

    # ── 最终保存 ──
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({'epoch': args.epochs, 'model': model.state_dict()}, final_path)
    print(f'Final model saved: {final_path}')
    print('Training complete!')


if __name__ == '__main__':
    main()
