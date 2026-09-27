#!/usr/bin/env python3
"""
RGBTSwinTrack — LasHeR 训练集微调脚本
======================================
参考 OmniAdapt 的数据采样 / 读取 / 预处理 / 训练管线：
  - Mmap 元数据缓存 (序列名 / 帧路径 / bbox 缓冲)
  - SiamFC curation (crop + resize)
  - 0.5 融合共享骨干微调
  - Varifocal Loss + GIoU Loss + L1 Loss

用法:
  python train_rgbt_lasher.py --weight /path/to/pretrained.pth --output_dir /path/to/output
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import random
import numpy as np
import cv2 as cv
import torch

_imread_orig = cv.imread
def _imread_silent(path, *args, **kwargs):
    fd = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    os.dup2(fd, 2); os.close(fd)
    try:
        return _imread_orig(path, *args, **kwargs)
    finally:
        os.dup2(old, 2); os.close(old)
cv.imread = _imread_silent
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm

# ── 项目根目录 ──
_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)

# ── 归一化统计量 ──
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_TIR_MEAN = [0.485, 0.456, 0.406]
_TIR_STD  = [0.229, 0.224, 0.225]



# ══════════════════════════════════════════════════════════════════════════════
# LasHeR Siamese 采样 Dataset (SiamFC curation, 对齐预训练模型)
# ══════════════════════════════════════════════════════════════════════════════

class LasHeRSiameseDataset(Dataset):
    """SiamFC curation: crop + resize with image mean fill."""

    def __init__(self, root, split, template_size=(192,192), search_size=(384,384),
                 template_area_factor=2.0, search_area_factor=4.0,
                 max_frame_gap=100, samples_per_epoch=60000, seed=0):
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
        print(f"LasHeRSiameseDataset({split}): {self.dataset.num_sequences} seqs, {samples_per_epoch} samples/epoch")

    def __len__(self): return self.samples_per_epoch

    def _read_rgb(self, path):
        img = cv.imread(path)
        if img is None: raise IOError(f"Cannot read {path}")
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)

    def _read_tir(self, path):
        img = cv.imread(path, cv.IMREAD_UNCHANGED)
        if img is None: raise IOError(f"Cannot read {path}")
        if img.dtype == np.uint16:
            mn, mx = img.min(), img.max()
            img = ((img.astype(np.float32)-mn)/(mx-mn+1e-6)*255).astype(np.uint8) if mx>mn else np.zeros_like(img, dtype=np.uint8)
        elif img.dtype != np.uint8: img = img.astype(np.uint8)
        if img.ndim == 2: img = np.stack([img,img,img], axis=2)
        elif img.shape[2] == 1: img = np.concatenate([img,img,img], axis=2)
        return img

    def _compute_curation_params(self, bbox_xywh, area_factor, output_size):
        x, y, w, h = bbox_xywh
        bg = (area_factor-1)*((w+h)*0.5); w_z = w+bg; h_z = h+bg
        scaling = math.sqrt((output_size[0]*output_size[1])/max(w_z*h_z,1e-6))
        source_center = np.array([x+w/2, y+h/2], dtype=np.float64)
        target_center = np.array([output_size[0]/2, output_size[1]/2], dtype=np.float64)
        params = np.zeros((3,2), dtype=np.float64)
        params[0]=[scaling,scaling]; params[1]=source_center; params[2]=target_center
        return params

    def _crop_and_resize(self, image, output_size, curation_params, image_mean=None):
        H_in,W_in=image.shape[:2]; W_out,H_out=output_size
        if image_mean is None: image_mean=image.astype(np.float32).mean(axis=(0,1))
        else: image_mean=np.asarray(image_mean, dtype=np.float32)
        scaling=curation_params[0]; sc=curation_params[1]; tc=curation_params[2]
        ox1=(0-sc[0])*scaling[0]+tc[0]; oy1=(0-sc[1])*scaling[1]+tc[1]
        ox2=(W_in-sc[0])*scaling[0]+tc[0]; oy2=(H_in-sc[1])*scaling[1]+tc[1]
        ox1_c=max(0,int(np.floor(ox1))); oy1_c=max(0,int(np.floor(oy1)))
        ox2_c=min(W_out,int(np.ceil(ox2))); oy2_c=min(H_out,int(np.ceil(oy2)))
        curated=np.full((H_out,W_out,3),image_mean,dtype=np.float32)
        if ox2_c>ox1_c and oy2_c>oy1_c:
            ix1_c=(ox1_c-tc[0])/scaling[0]+sc[0]; iy1_c=(oy1_c-tc[1])/scaling[1]+sc[1]
            ix2_c=(ox2_c-tc[0])/scaling[0]+sc[0]; iy2_c=(oy2_c-tc[1])/scaling[1]+sc[1]
            ix1_i=max(0,int(np.floor(ix1_c))); iy1_i=max(0,int(np.floor(iy1_c)))
            ix2_i=min(W_in,int(np.ceil(ix2_c))); iy2_i=min(H_in,int(np.ceil(iy2_c)))
            if ix2_i>ix1_i and iy2_i>iy1_i:
                input_crop=image[iy1_i:iy2_i,ix1_i:ix2_i].astype(np.float32)
                out_w=ox2_c-ox1_c; out_h=oy2_c-oy1_c
                resized=cv.resize(input_crop,(out_w,out_h),interpolation=cv.INTER_LINEAR)
                curated[oy1_c:oy2_c,ox1_c:ox2_c]=resized
        return curated,image_mean

    def __getitem__(self, index):
        max_retries=10
        for _ in range(max_retries):
            try: return self._getitem_impl()
            except (OSError,IOError,Exception): continue
        raise RuntimeError(f"Failed after {max_retries} retries")

    def _getitem_impl(self):
        for _retry in range(50):
            seq_id=int(self.rng.choice(self.dataset.num_sequences,p=self._seq_p))
            info=self.dataset.get_sequence_info(seq_id)
            bboxes=info['bbox']; valid=info['valid'].numpy(); n_frames=len(bboxes)
            valid_ids=np.where(valid)[0]
            if n_frames<20 or len(valid_ids)<2: continue

            # causal sampling
            search_frame_ids=None; gap_increase=0
            while search_frame_ids is None and gap_increase<=500:
                min_base=1; max_base=n_frames-1
                bc=valid_ids[(valid_ids>=min_base)&(valid_ids<=max_base)]
                if len(bc)==0: gap_increase+=5; continue
                base_id=int(self.rng.choice(bc))
                pmin=max(0,base_id-self.max_frame_gap-gap_increase)
                pc=valid_ids[(valid_ids>=pmin)&(valid_ids<base_id)]
                if len(pc)==0: gap_increase+=5; continue
                template_id=int(self.rng.choice(pc))
                smax=min(n_frames,template_id+self.max_frame_gap+gap_increase+1)
                sc=valid_ids[(valid_ids>template_id)&(valid_ids<smax)]
                if len(sc)==0: gap_increase+=5; continue
                search_frame_ids=[int(self.rng.choice(sc))]
            if search_frame_ids is None: continue
            search_id=search_frame_ids[0]

            try:
                tp1,tp2=self.dataset.get_frame_paths(seq_id,template_id)
                sp1,sp2=self.dataset.get_frame_paths(seq_id,search_id)
                t_rgb=self._read_rgb(tp1); t_tir=self._read_tir(tp2)
                s_rgb=self._read_rgb(sp1); s_tir=self._read_tir(sp2)
            except (OSError,IOError): continue

            t_bbox=bboxes[template_id].numpy().tolist()
            s_bbox=bboxes[search_id].numpy().tolist()

            try:
                # Template (no jitter)
                tp=self._compute_curation_params(t_bbox,self.template_area_factor,self.template_size)
                z_rgb,zm_rgb=self._crop_and_resize(t_rgb,self.template_size,tp)
                z_tir,zm_tir=self._crop_and_resize(t_tir,self.template_size,tp)

                # Search jitter (温和)
                sj=s_bbox.copy()
                js=math.exp(self.rng.randn()*0.1)
                sj[0]=s_bbox[0]+self.rng.uniform(-0.1,0.1)*s_bbox[2]
                sj[1]=s_bbox[1]+self.rng.uniform(-0.1,0.1)*s_bbox[3]
                sj[2]=max(s_bbox[2]*js,10); sj[3]=max(s_bbox[3]*js,10)
                sp=self._compute_curation_params(sj,self.search_area_factor,self.search_size)
                x_rgb,_=self._crop_and_resize(s_rgb,self.search_size,sp,image_mean=zm_rgb)
                x_tir,_=self._crop_and_resize(s_tir,self.search_size,sp,image_mean=zm_tir)

                # Map GT bbox to crop (SiamFC style)
                scaling_s=sp[0]; sc_s=sp[1]; tc_s=sp[2]
                cx=s_bbox[0]+s_bbox[2]/2; cy=s_bbox[1]+s_bbox[3]/2
                ocx=(cx-sc_s[0])*scaling_s[0]+tc_s[0]; ocy=(cy-sc_s[1])*scaling_s[1]+tc_s[1]
                ow=s_bbox[2]*scaling_s[0]; oh=s_bbox[3]*scaling_s[1]
                W_out,H_out=self.search_size
                bbox_gt=[ocx/W_out,ocy/H_out,ow/W_out,oh/H_out]
                bbox_gt=[max(0.0,min(1.0,v)) for v in bbox_gt]
            except (ValueError,Exception): continue

            z_rgb_t=torch.from_numpy(z_rgb/255.0).permute(2,0,1).float()
            z_tir_t=torch.from_numpy(z_tir/255.0).permute(2,0,1).float()
            x_rgb_t=torch.from_numpy(x_rgb/255.0).permute(2,0,1).float()
            x_tir_t=torch.from_numpy(x_tir/255.0).permute(2,0,1).float()

            rgb_norm=transforms.Normalize(mean=_IMAGENET_MEAN,std=_IMAGENET_STD)
            tir_norm=transforms.Normalize(mean=_TIR_MEAN,std=_TIR_STD)

            if self.split=='train':
                js_state=self.rng.randint(0,2**31)
                flip_rand=self.rng.rand(); gray_rand=self.rng.rand()
                cj=transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.0)
                torch.manual_seed(js_state)
                if gray_rand<0.05: z_rgb_t=z_rgb_t.mean(dim=0,keepdim=True).expand(3,-1,-1)
                else: z_rgb_t=cj(z_rgb_t)
                torch.manual_seed(js_state)
                if gray_rand<0.05: x_rgb_t=x_rgb_t.mean(dim=0,keepdim=True).expand(3,-1,-1)
                else: x_rgb_t=cj(x_rgb_t)
                if flip_rand<0.5:
                    z_rgb_t=torch.flip(z_rgb_t,[-1]); z_tir_t=torch.flip(z_tir_t,[-1])
                    x_rgb_t=torch.flip(x_rgb_t,[-1]); x_tir_t=torch.flip(x_tir_t,[-1])
                    bbox_gt[0]=1.0-bbox_gt[0]

            z_rgb_t=rgb_norm(z_rgb_t); z_tir_t=tir_norm(z_tir_t)
            x_rgb_t=rgb_norm(x_rgb_t); x_tir_t=tir_norm(x_tir_t)
            bbox_gt_t=torch.tensor(bbox_gt,dtype=torch.float32)
            return z_rgb_t,z_tir_t,x_rgb_t,x_tir_t,bbox_gt_t

        raise RuntimeError("Failed to sample after 50 retries")

# ══════════════════════════════════════════════════════════════════════════════
# 损失函数
# ══════════════════════════════════════════════════════════════════════════════

def giou_loss(pred, target, reduction='mean'):
    """GIoU Loss (简化实现)。"""
    # pred/target: (B, 4) CXCYWH [0,1)
    px1 = pred[:, 0] - pred[:, 2] / 2
    py1 = pred[:, 1] - pred[:, 3] / 2
    px2 = pred[:, 0] + pred[:, 2] / 2
    py2 = pred[:, 1] + pred[:, 3] / 2

    tx1 = target[:, 0] - target[:, 2] / 2
    ty1 = target[:, 1] - target[:, 3] / 2
    tx2 = target[:, 0] + target[:, 2] / 2
    ty2 = target[:, 1] + target[:, 3] / 2

    # Intersection
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    # Union
    pa = (px2 - px1) * (py2 - py1)
    ta = (tx2 - tx1) * (ty2 - ty1)
    union = pa + ta - inter

    iou = inter / (union + 1e-6)

    # Enclosing box
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2)
    ey2 = torch.max(py2, ty2)
    ea = (ex2 - ex1) * (ey2 - ey1)
    giou = iou - (ea - union) / (ea + 1e-6)

    loss = 1 - giou
    if reduction == 'mean':
        return loss.mean()
    return loss


def varifocal_loss(pred, target, alpha=0.75, gamma=2.0):
    """Varifocal Loss — 对齐原始 SwinTrack 实现。"""
    pred_sigmoid = pred.clamp(1e-6, 1 - 1e-6)
    target = target.float()
    weight = target + alpha * pred_sigmoid.pow(gamma) * (1 - target)
    loss = F.binary_cross_entropy(pred_sigmoid, target, reduction='none')
    loss = weight * loss
    n_pos = max(target.sum(), 1.0)
    return loss.sum() / n_pos


def compute_iou(bbox_pred, bbox_gt):
    """计算两组 CXCYWH bbox 的 IoU。"""
    px1 = bbox_pred[:, 0] - bbox_pred[:, 2] / 2
    py1 = bbox_pred[:, 1] - bbox_pred[:, 3] / 2
    px2 = bbox_pred[:, 0] + bbox_pred[:, 2] / 2
    py2 = bbox_pred[:, 1] + bbox_pred[:, 3] / 2
    gx1 = bbox_gt[:, 0] - bbox_gt[:, 2] / 2
    gy1 = bbox_gt[:, 1] - bbox_gt[:, 3] / 2
    gx2 = bbox_gt[:, 0] + bbox_gt[:, 2] / 2
    gy2 = bbox_gt[:, 1] + bbox_gt[:, 3] / 2
    ix1 = torch.max(px1, gx1); iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2); iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_p = (px2 - px1) * (py2 - py1)
    area_g = (gx2 - gx1) * (gy2 - gy1)
    union = area_p + area_g - inter + 1e-6
    return inter / union


def compute_targets(bbox_gt, feat_size, search_size):
    """计算 ResponseMap 训练目标。

    Args:
        bbox_gt: (B, 4) CXCYWH 归一化 [0,1)
        feat_size: (H, W) 特征图尺寸
        search_size: (W, H) 搜索图像尺寸
    Returns:
        cls_target: (B, H, W)
        reg_target: (B, H, W, 4) CXCYWH 归一化
    """
    B = bbox_gt.shape[0]
    H, W = feat_size
    SW, SH = search_size

    # GT bbox 在搜索图像中的像素坐标
    gcx = bbox_gt[:, 0] * SW   # (B,)
    gcy = bbox_gt[:, 1] * SH
    gw  = bbox_gt[:, 2] * SW
    gh  = bbox_gt[:, 3] * SH

    # 特征图网格
    y = torch.arange(H, device=bbox_gt.device).float()
    x = torch.arange(W, device=bbox_gt.device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # (H, W)
    # 特征图位置对应的搜索图像坐标
    px = (xx + 0.5) * (SW / W)  # (H, W)
    py = (yy + 0.5) * (SH / H)

    # 正样本: 落在 GT bbox 内的网格点
    gx1 = gcx - gw / 2  # (B,)
    gy1 = gcy - gh / 2
    gx2 = gcx + gw / 2
    gy2 = gcy + gh / 2

    px = px.unsqueeze(0)   # (1, H, W)
    py = py.unsqueeze(0)
    pos_mask = (px >= gx1.view(B,1,1)) & (px < gx2.view(B,1,1)) & \
               (py >= gy1.view(B,1,1)) & (py < gy2.view(B,1,1))  # (B, H, W)

    cls_target = pos_mask.float()

    # 回归目标: 每个位置 → GT center + size 的偏移
    # 归一化到 [0,1)
    reg_target = bbox_gt.view(B, 1, 1, 4).expand(-1, H, W, -1).contiguous()

    return cls_target, reg_target


# ══════════════════════════════════════════════════════════════════════════════
# 模型构建
# ══════════════════════════════════════════════════════════════════════════════

def build_model_enc_fuse(device):
    from core.run.event_dispatcher.register import EventRegister
    from models.methods.SwinRGBTrack.builder_enc_fuse import build_rgbt_enc_fuse
    from miscellanies.yaml_ops import load_yaml
    config = load_yaml(os.path.join(_PRJ_ROOT, 'config', 'SwinRGBTrack', 'Base-384-enc-fuse', 'config.yaml'))
    er = EventRegister('model/')
    return build_rgbt_enc_fuse(config, False, 1, 1, er, True), config


# ══════════════════════════════════════════════════════════════════════════════
# 训练主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser('RGBTSwinTrack LasHeR 微调训练')
    parser.add_argument('--weight', type=str, required=True, help='预训练权重路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--lasher_root', type=str,
                        default=os.environ.get('LASHER_ROOT', '/home/fzg/data/lasher'),
                        help='LasHeR 数据集根目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率 (微调建议低 lr)')
    parser.add_argument('--backbone_lr', type=float, default=1e-7, help='骨干网络学习率 (建议 0 或 1e-7)')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=10,
                        help='前 N 轮冻结 backbone (0=不冻结)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率 warmup 轮数 (0=不 warmup)')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=-1, help='分布式训练的 local rank')
    args = parser.parse_args()

    # ── 分布式初始化 ──
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{args.local_rank}')
        is_main = (args.local_rank == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main = True

    if is_main:
        print(f"Device: {device}")
        print(f"Output dir: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

    # ── 随机种子 ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ── 构建模型 ──
    if is_main:
        print("Building RGBTSwinTrack-EncFuse model...")
    model, config = build_model_enc_fuse(device)

    # ── 加载预训练权重 ──
    if is_main:
        print(f"Loading pretrained weights: {args.weight}")
    checkpoint = torch.load(args.weight, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model_state = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    model.to(device)

    # ── 分布式包装 ──
    if args.local_rank != -1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      find_unused_parameters=True)

    # ── 数据集 ──
    train_dataset = LasHeRSiameseDataset(
        root=args.lasher_root, split='train',
        seed=args.seed,
    )
    val_dataset = LasHeRSiameseDataset(
        root=args.lasher_root, split='val',
        samples_per_epoch=4000,
        seed=args.seed + 1,
    )

    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=args.workers,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, sampler=val_sampler,
                            num_workers=args.workers,
                            pin_memory=False, drop_last=False)

    # ── 优化器 ──
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    # Backbone 单独参数组（可能冻结）
    if args.backbone_lr > 0 and len(backbone_params) > 0:
        optimizer.add_param_group({'params': backbone_params, 'lr': args.backbone_lr})

    # ── 学习率调度 (CosineAnnealing + 可选 Warmup) ──
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs)

    def get_lr(epoch):
        if epoch < args.warmup_epochs:
            # Linear warmup
            return (epoch + 1) / args.warmup_epochs
        else:
            return base_scheduler.get_last_lr()[0] / args.lr  # normalized

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[get_lr] * len(optimizer.param_groups))

    if is_main:
        print(f"Freeze backbone: {args.freeze_backbone_epochs} epochs")
        print(f"Warmup: {args.warmup_epochs} epochs")
        print(f"Backbone LR: {args.backbone_lr}, Other LR: {args.lr}")

    # ── 特征图尺寸 ──
    feat_h, feat_w = 24, 24
    search_w, search_h = 384, 384

    if is_main:
        print(f"Training for {args.epochs} epochs, {len(train_loader)} iters/epoch")

    # ════════════════════════════════════════════════════════════════════════
    # 训练循环
    # ════════════════════════════════════════════════════════════════════════
    for epoch in range(1, args.epochs + 1):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)

        # ── Backbone 冻结/解冻 ──
        if args.freeze_backbone_epochs > 0:
            freeze = epoch <= args.freeze_backbone_epochs
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = not freeze
            if is_main and epoch == 1:
                print(f"Backbone frozen for first {args.freeze_backbone_epochs} epochs")
            if is_main and epoch == args.freeze_backbone_epochs + 1:
                print(f"Backbone UNFROZEN at epoch {epoch} (lr={args.backbone_lr})")

        model.train()
        et, ecl, egl, eio, el1, epr = 0., 0., 0., 0., 0., 0.

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', disable=not is_main)
        for batch_idx, (z_rgb, z_tir, x_rgb, x_tir, bbox_gt) in enumerate(pbar):
            z_rgb = z_rgb.to(device, non_blocking=True)
            z_tir = z_tir.to(device, non_blocking=True)
            x_rgb = x_rgb.to(device, non_blocking=True)
            x_tir = x_tir.to(device, non_blocking=True)
            bbox_gt = bbox_gt.to(device, non_blocking=True)

            output = model(z_rgb, z_tir, x_rgb, x_tir)
            cls_pred = output['class_score']; reg_pred = output['bbox']; B = cls_pred.shape[0]

            cls_target, reg_target = compute_targets(bbox_gt, (feat_h, feat_w), (search_w, search_h))
            pos_mask = cls_target > 0.5; n_pos = pos_mask.sum().item()
            pos_ratio = n_pos / (B * feat_h * feat_w)

            # ── IoU-aware Varifocal target（对齐原始 SwinTrack）──
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
                cls_loss = 1.5 * varifocal_loss(cls_pred_prob, cls_target_f)
                gl = giou_loss(reg_pred_flat[pos_mask_f], reg_target_flat[pos_mask_f])
                io = 1.0 - gl.item()
                ll = torch.tensor(0.0, device=device)
            else:
                cls_pred_prob = cls_pred_f.clamp(1e-6, 1 - 1e-6)
                cls_loss = 1.5 * varifocal_loss(cls_pred_prob, cls_target_f)
                gl = torch.tensor(0.0, device=device); io = 0.; ll = torch.tensor(0.0, device=device)

            loss = cls_loss + 1.5 * gl
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1); optimizer.step()

            et+=loss.item(); ecl+=cls_loss.item(); egl+=gl.item()
            eio+=io; el1+=ll.item(); epr+=pos_ratio
            if is_main and (batch_idx + 1) % 50 == 0:
                n = batch_idx + 1
                pbar.set_description(f'Epoch {epoch}/{args.epochs} [{batch_idx+1}/{len(train_loader)}]')
                print(f'  Step {batch_idx+1:5d} | loss={et/n:.4f} cls={ecl/n:.4f} GIoU={egl/n:.4f} IoU={eio/n:.3f} L1={el1/n:.4f} Pos%={epr/n*100:.1f}', flush=True)

        scheduler.step()
        n_batches = len(train_loader)
        if is_main:
            print(f'Epoch {epoch}/{args.epochs} | '
                  f'Loss:{et/n_batches:.4f} Cls:{ecl/n_batches:.4f} GIoU:{egl/n_batches:.4f} IoU:{eio/n_batches:.3f} L1:{el1/n_batches:.4f} Pos%:{epr/n_batches*100:.1f} '
f'LR:{optimizer.param_groups[0]["lr"]:.2e}')

        # ── 验证 (每轮) ──
        if is_main:
            model.eval(); vt, vcl, vgl, vio, vl1, vpr = 0., 0., 0., 0., 0., 0.
            with torch.no_grad():
                for z_rgb, z_tir, x_rgb, x_tir, bbox_gt in val_loader:
                    z_rgb = z_rgb.to(device); z_tir = z_tir.to(device)
                    x_rgb = x_rgb.to(device); x_tir = x_tir.to(device)
                    bbox_gt = bbox_gt.to(device)
                    output = model(z_rgb, z_tir, x_rgb, x_tir)
                    cls_pred = output['class_score']; reg_pred = output['bbox']; B = cls_pred.shape[0]
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
                        cl = 1.5 * varifocal_loss(cls_pred_prob, cls_target_f)
                        gl = giou_loss(reg_pred_f[pm_f], reg_target_f[pm_f]); io = 1.0 - gl.item()
                        ll = torch.tensor(0.0, device=device)
                    else:
                        cls_pred_prob = cls_pred_f.clamp(1e-6, 1 - 1e-6)
                        cl = 1.5 * varifocal_loss(cls_pred_prob, cls_target_f)
                        gl = torch.tensor(0.0, device=device); io = 0.; ll = torch.tensor(0.0, device=device)
                    vt += (cl + 1.5 * gl).item(); vcl += cl.item()
                    vgl += gl.item(); vio += io; vl1 += ll.item(); vpr += rp
            n_val = len(val_loader)
            print(f'  Val | Loss:{vt/n_val:.4f} Cls:{vcl/n_val:.4f} GIoU:{vgl/n_val:.4f} IoU:{vio/n_val:.3f} L1:{vl1/n_val:.4f} Pos%:{vpr/n_val*100:.1f}')

        # ── 保存 checkpoint (每轮) ──
        if is_main:
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:03d}.pth')
            state = model.module.state_dict() if args.local_rank != -1 else model.state_dict()
            torch.save({'epoch': epoch, 'model': state, 'optimizer': optimizer.state_dict()}, ckpt_path)
            print(f'  Checkpoint saved: {ckpt_path}')

        # ── 每 5 轮完整测试 ──
        if is_main and epoch % 5 == 0:
            test_result_dir = os.path.join(args.output_dir, 'test_results', f'epoch{epoch:03d}')
            print(f'  Running full RGBT test for epoch {epoch} ...')
            ret = os.system(f'python {os.path.join(_PRJ_ROOT,"test_lasher_rgbt_enc_fuse.py")} --weight {ckpt_path} --save_dir {test_result_dir} --workers 4')
            if ret != 0: print(f'  [WARN] Full test exited with code {ret}')
            else: print(f'  Full test results → {test_result_dir}')

    if is_main:
        # 保存最终模型
        final_path = os.path.join(args.output_dir, 'final_model.pth')
        if args.local_rank != -1:
            state = model.module.state_dict()
        else:
            state = model.state_dict()
        torch.save({'model': state}, final_path)
        print(f'Final model saved: {final_path}')
        print('Training complete!')


if __name__ == '__main__':
    main()
