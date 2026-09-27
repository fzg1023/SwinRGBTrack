#!/usr/bin/env python3
"""
RGBTSwinTrack-LayerFuse — LasHeR 微调训练脚本
==============================================
参考 DropFuseRGBT 的逐层融合 + 跨层聚合, 在 Encoder 每层后做
Concat→Linear→LN→GELU 跨模态融合, 然后跨层求和聚合 → Decoder → Head。

推荐训练策略:
  --init_from dec_fuse  : 从 DecFuse epoch5 最优权重初始化 (推荐, 收敛快)
  --init_from scratch    : 从 SwinTrack 预训练权重从头训练
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import OrderedDict

import cv2 as cv
import numpy as np
import torch
import core.amp_compat  # AMP 兼容层 (服务器旧版 torch 无 torch.amp.GradScaler)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)

# ══════════════════════════════════════════════════════════════════════════════
# Dataset & Loss (复用 DecFuse 的实现)
# ══════════════════════════════════════════════════════════════════════════════

from train_rgbt_dec_fuse import (
    LasHeRSiameseDataset,
    giou_loss, varifocal_loss, compute_iou, compute_targets,
)

# ══════════════════════════════════════════════════════════════════════════════
# 模型构建
# ══════════════════════════════════════════════════════════════════════════════

def build_model_layer_fuse(device):
    from core.run.event_dispatcher.register import EventRegister
    from models.methods.SwinRGBTrack.builder_layer_fuse import build_rgbt_layer_fuse
    from miscellanies.yaml_ops import load_yaml
    config = load_yaml(os.path.join(_PRJ_ROOT, 'config', 'SwinRGBTrack',
                                     'Base-384-layer-fuse', 'config.yaml'))
    er = EventRegister('model/')
    return build_rgbt_layer_fuse(config, False, 1, 1, er, True), config


def load_weights(model, weight_path, device):
    """灵活的权重加载: 支持 strict=False, 自动映射 DecFuse→LayerFuse 键名。

    键名映射:
      encoder.layers.*  → layer_fuse_encoder.layers.*   (自注意力层)
      encoder.*         → layer_fuse_encoder.*           (位置编码等)
      decoder.*         → decoder.*                      (不变)
      backbone.*        → backbone.*                     (不变)
      head.*            → head.*                         (不变)
      新增的 cross_fuse.* 和 cross_layer_agg.* → 随机初始化
    """
    print(f"Loading weights: {weight_path}")
    ckpt = torch.load(weight_path, map_location='cpu')

    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # 键名映射: DecFuse → LayerFuse
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        # encoder.* → layer_fuse_encoder.*
        if k.startswith('encoder.'):
            k = 'layer_fuse_encoder.' + k[8:]
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  [INFO] Missing keys (random init): {len(missing)}")
        # 只显示 fusion 模块的 missing keys (其余是预期的)
        fusion_missing = [m for m in missing if 'cross_fuse' in m or 'cross_layer_agg' in m]
        other_missing = [m for m in missing if 'cross_fuse' not in m and 'cross_layer_agg' not in m]
        if fusion_missing:
            print(f"    Fusion modules (expected): {len(fusion_missing)}")
        if other_missing:
            print(f"    Other (unexpected): {len(other_missing)}")
            for m in other_missing[:3]:
                print(f"      - {m}")
    if unexpected:
        print(f"  [INFO] Unexpected keys (ignored): {len(unexpected)}")


# ══════════════════════════════════════════════════════════════════════════════
# 训练主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser('RGBTSwinTrack-LayerFuse LasHeR 微调训练')
    parser.add_argument('--weight', type=str, required=True,
                        help='初始化权重路径 (推荐 DecFuse epoch5 checkpoint)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--lasher_root', type=str,
                        default=os.environ.get('LASHER_ROOT', '/home/fzg/data/lasher'),
                        help='LasHeR 数据集根目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--backbone_lr', type=float, default=1e-5, help='骨干网络学习率')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=2,
                        help='前 N 轮冻结 backbone + encoder SA 层 (0=不冻结)')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='学习率 warmup 轮数')
    parser.add_argument('--grad_accum', type=int, default=2,
                        help='梯度累积步数')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='使用 AMP 混合精度训练')
    parser.add_argument('--no_amp', action='store_false', dest='amp',
                        help='禁用 AMP')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--resume', type=str, default='',
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=-1)
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
        print("=" * 60)
        print("  RGBTSwinTrack-LayerFuse LasHeR 微调训练")
        print("  (Encoder逐层融合 + 跨层聚合 + Decoder)")
        print("=" * 60)
        print(f"  初始化权重: {args.weight}")
        print(f"  输出目录:   {args.output_dir}")
        print(f"  LasHeR:     {args.lasher_root}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Epochs:     {args.epochs}")
        print(f"  LR:         {args.lr}")
        print(f"  Backbone LR:{args.backbone_lr}")
        print(f"  Freeze:     {args.freeze_backbone_epochs} epochs")
        print(f"  Warmup:     {args.warmup_epochs} epochs")
        print(f"  Workers:    {args.workers}")
        print(f"  GPUs:       {max(1, torch.cuda.device_count())}")
        print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    device_name = str(device)
    print(f"Device: {device_name}")
    print(f"Output dir: {args.output_dir}")

    # ── 模型 ──
    if is_main:
        print("Building RGBTSwinTrack-LayerFuse model...")
    model, config = build_model_layer_fuse(device)
    load_weights(model, args.weight, device)
    model = model.to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"  Total params: {n_params:.1f}M, Trainable: {n_trainable:.1f}M")

    # ── 数据集 ──
    train_ds = LasHeRSiameseDataset(root=args.lasher_root, split='train',
                                     template_size=(192, 192), search_size=(384, 384),
                                     template_area_factor=2.0, search_area_factor=4.0,
                                     max_frame_gap=200, samples_per_epoch=60000)
    val_ds = LasHeRSiameseDataset(root=args.lasher_root, split='val',
                                   template_size=(192, 192), search_size=(384, 384),
                                   template_area_factor=2.0, search_area_factor=4.0,
                                   max_frame_gap=200, samples_per_epoch=4000)

    train_sampler = None
    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=(train_sampler is None),
                               sampler=train_sampler,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers,
                             pin_memory=True, drop_last=False)

    # ── 优化器 ──
    backbone_params = []; fusion_params = []; other_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'cross_fuse' in name or 'cross_layer_agg' in name:
            fusion_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': fusion_params, 'lr': args.lr},  # 新融合模块用主学习率
    ], weight_decay=1e-4)

    if args.backbone_lr > 0 and len(backbone_params) > 0:
        optimizer.add_param_group({'params': backbone_params, 'lr': args.backbone_lr})

    # ── Scheduler ──
    try:
        from timm.scheduler import CosineLRScheduler
        scheduler = CosineLRScheduler(
            optimizer, t_initial=args.epochs,
            lr_min=args.lr * 1e-3,
            warmup_t=args.warmup_epochs,
            warmup_lr_init=args.lr * 1e-2,
            warmup_prefix=True,
        )
        use_timm_scheduler = True
    except ImportError:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        use_timm_scheduler = False

    # ── AMP ──
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if args.amp else None

    # ── Resume ──
    start_epoch = 1
    if args.resume:
        if is_main: print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 1) + 1
        if is_main: print(f"Resumed at epoch {start_epoch}")

    if is_main:
        print(f"Freeze backbone+encoder SA: {args.freeze_backbone_epochs} epochs")
        print(f"Warmup: {args.warmup_epochs} epochs")
        print(f"Fusion LR: {args.lr}, Backbone LR: {args.backbone_lr}")
        print(f"AMP: {args.amp}, Grad Accum: {args.grad_accum}")

    feat_h, feat_w = 24, 24
    search_w, search_h = 384, 384

    if is_main:
        print(f"Training for {args.epochs} epochs, {len(train_loader)} iters/epoch")
        if args.grad_accum > 1:
            opt_steps = len(train_loader) // args.grad_accum
            print(f"  Effective optimizer steps/epoch: {opt_steps} (grad_accum={args.grad_accum})")

    # ════════════════════════════════════════════════════════════════════════
    # 训练循环
    # ════════════════════════════════════════════════════════════════════════
    optimizer_step_count = 0
    total_opt_steps = len(train_loader) // args.grad_accum
    for epoch in range(start_epoch, args.epochs + 1):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)

        # ── 冻结/解冻 backbone + encoder SA 层 ──
        if args.freeze_backbone_epochs > 0:
            freeze = epoch <= args.freeze_backbone_epochs
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = not freeze
                # 冻结 encoder 自注意力层 (layer_fuse_encoder.layers.*)
                elif 'layer_fuse_encoder.layers' in name:
                    param.requires_grad = not freeze
            if is_main and epoch == 1:
                print(f"Backbone + Encoder SA frozen for first {args.freeze_backbone_epochs} epochs")
            if is_main and epoch == args.freeze_backbone_epochs + 1:
                print(f"Backbone + Encoder SA UNFROZEN at epoch {epoch} (backbone_lr={args.backbone_lr})")

        model.train()
        et, ecl, egl, eio, el1, epr = 0., 0., 0., 0., 0., 0.
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', disable=not is_main)
        for batch_idx, (z_rgb, z_tir, x_rgb, x_tir, bbox_gt) in enumerate(pbar):
            z_rgb = z_rgb.to(device, non_blocking=True)
            z_tir = z_tir.to(device, non_blocking=True)
            x_rgb = x_rgb.to(device, non_blocking=True)
            x_tir = x_tir.to(device, non_blocking=True)
            bbox_gt = bbox_gt.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=args.amp):
                output = model(z_rgb, z_tir, x_rgb, x_tir)

            cls_pred = output['class_score'].float()
            reg_pred = output['bbox'].float()
            B = cls_pred.shape[0]

            cls_target, reg_target = compute_targets(bbox_gt, (feat_h, feat_w), (search_w, search_h))
            pos_mask = cls_target > 0.5; n_pos = pos_mask.sum().item()
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
                gl = giou_loss(reg_pred_flat[pos_mask_f], reg_target_flat[pos_mask_f])
                io = 1.0 - gl.item()
                ll = torch.tensor(0.0, device=device)
            else:
                cls_pred_prob = cls_pred_f.clamp(1e-6, 1 - 1e-6)
                cls_loss = 2.0 * varifocal_loss(cls_pred_prob, cls_target_f)
                gl = torch.tensor(0.0, device=device); io = 0.; ll = torch.tensor(0.0, device=device)

            loss = (cls_loss + 2.0 * gl) / args.grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                optimizer_step_count += 1
                if use_timm_scheduler:
                    effective_step = epoch - 1 + optimizer_step_count / total_opt_steps
                    scheduler.step(effective_step)

            et += loss.item(); ecl += cls_loss.item(); egl += gl.item()
            eio += io; el1 += ll.item(); epr += pos_ratio
            if is_main and (batch_idx + 1) % 50 == 0:
                n = batch_idx + 1
                pbar.set_description(f'Epoch {epoch}/{args.epochs} [{batch_idx+1}/{len(train_loader)}]')
                print(f'  Step {batch_idx+1:5d} | loss={et/n:.4f} cls={ecl/n:.4f} GIoU={egl/n:.4f} IoU={eio/n:.3f} L1={el1/n:.4f} Pos%={epr/n*100:.1f}', flush=True)

        if not use_timm_scheduler:
            scheduler.step()
        n_batches = len(train_loader)
        if is_main:
            lr_to_print = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}/{args.epochs} | '
                  f'Loss:{et/n_batches:.4f} Cls:{ecl/n_batches:.4f} GIoU:{egl/n_batches:.4f} IoU:{eio/n_batches:.3f} L1:{el1/n_batches:.4f} Pos%:{epr/n_batches*100:.1f} '
                  f'LR:{lr_to_print:.2e}')

        # ── 验证 ──
        if is_main:
            model.eval(); vt, vcl, vgl, vio, vl1, vpr = 0., 0., 0., 0., 0., 0.
            with torch.no_grad():
                for z_rgb, z_tir, x_rgb, x_tir, bbox_gt in val_loader:
                    z_rgb = z_rgb.to(device); z_tir = z_tir.to(device)
                    x_rgb = x_rgb.to(device); x_tir = x_tir.to(device)
                    bbox_gt = bbox_gt.to(device)
                    with torch.amp.autocast('cuda', enabled=args.amp):
                        output = model(z_rgb, z_tir, x_rgb, x_tir)
                    cls_pred = output['class_score'].float()
                    reg_pred = output['bbox'].float()
                    B = cls_pred.shape[0]
                    cls_target, reg_target = compute_targets(bbox_gt, (feat_h, feat_w), (search_w, search_h))
                    pos_mask = cls_target > 0.5; n_pos = pos_mask.sum().item()
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
                        cl = 2.0 * varifocal_loss(cls_pred_prob, cls_target_f)
                        gl = giou_loss(reg_pred_flat[pos_mask_f], reg_target_flat[pos_mask_f])
                        io = 1.0 - gl.item(); ll2 = torch.tensor(0.0, device=device)
                    else:
                        cl = 2.0 * varifocal_loss(cls_pred_f.clamp(1e-6, 1-1e-6), cls_target_f)
                        gl = torch.tensor(0.0, device=device); io = 0.; ll2 = torch.tensor(0.0, device=device)
                    vl = cl.item() + 2.0 * gl.item()
                    vt += vl; vcl += cl.item(); vgl += gl.item()
                    vio += io; vl1 += ll2.item(); vpr += n_pos/(B*feat_h*feat_w)
            n_val = len(val_loader)
            print(f'  Val | Loss:{vt/n_val:.4f} Cls:{vcl/n_val:.4f} GIoU:{vgl/n_val:.4f} IoU:{vio/n_val:.3f} L1:{vl1/n_val:.4f} Pos%:{vpr/n_val*100:.1f}')

        # ── 保存 ──
        if is_main:
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:03d}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            print(f'Checkpoint saved: {ckpt_path}')


if __name__ == '__main__':
    main()
