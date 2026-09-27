#!/usr/bin/env python3
"""
RGBTSwinTrack-EncPromptLoRA — VTUAV 从头训练
===============================================================
在 Base-384-enc-promptlora-vtuav 配置上从头训练 (backbone 用
SwinTrack-B-384.pth ImageNet 预训练初始化, encoder/decoder/head
与 prompt/LoRA 模块随机初始化)。

数据 (与 SGTrack 服务器目录组织一致):
  VTUAV_HOME=/root/RGBTData/VTUAV
    train : <VTUAV_HOME>/train    (训练)
    val_st: <VTUAV_HOME>/test_ST  (每轮验证, 短时测试集)
  每 5 轮自动对 vtuav_st 全量测试。

用法:
  python train_vtuav_promptlora.py \
      --weight SwinTrack-B-384.pth \
      --output_dir output/enc_promptlora_vtuav \
      --vtuav_root /root/RGBTData/VTUAV
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
import torch
import core.amp_compat  # AMP 兼容层 (服务器旧版 torch 无 torch.amp.GradScaler)
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── 项目根目录 ──
_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)

# ── 复用 EncFuse-V1 训练管线组件 (Siamese 采样/损失/目标/权重加载) ──
from train_rgbt_enc_fuse_v1 import (
    LasHeRSiameseDataset, giou_loss, varifocal_loss, compute_iou,
    compute_targets, load_state_dict_filtered,
)

_MODEL_REGISTRY = {
    'enc_promptlora_vtuav': (
        'config/SwinRGBTrack/Base-384-enc-promptlora-vtuav',
        'models.methods.SwinRGBTrack.builder_enc_promptlora',
        'build_rgbt_enc_promptlora'),
}


def build_model(model_type='enc_promptlora_vtuav'):
    from importlib import import_module
    from core.run.event_dispatcher.register import EventRegister
    from miscellanies.yaml_ops import load_yaml
    cfg_dir, module_name, func_name = _MODEL_REGISTRY[model_type]
    config = load_yaml(os.path.join(_PRJ_ROOT, cfg_dir, 'config.yaml'))
    builder = getattr(import_module(module_name), func_name)
    er = EventRegister('model/')
    # has_training_run=False: 独立训练循环不触发 drop_path warmup hook
    return builder(config, False, 1, 1, er, False)


# ══════════════════════════════════════════════════════════════════════════════
# VTUAV Siamese 采样 Dataset — 复用 LasHeRSiameseDataset 采样逻辑,
# 仅替换底层 loader 为 VtuavRGBTDataset
# ══════════════════════════════════════════════════════════════════════════════

class VtuavSiameseDataset(LasHeRSiameseDataset):
    """VTUAV 版 Siamese 采样数据集 (接口/采样逻辑继承 LasHeRSiameseDataset)。"""

    def __init__(self, root, split, template_size=(192, 192), search_size=(384, 384),
                 template_area_factor=2.0, search_area_factor=4.0,
                 max_frame_gap=200, samples_per_epoch=20000, seed=0):
        from datasets.RGBT.vtuav_rgbt import VtuavRGBTDataset
        self.dataset = VtuavRGBTDataset(root, split)
        self.split = split
        self.template_size = template_size
        self.search_size = search_size
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        self.max_frame_gap = max_frame_gap
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.RandomState(seed)
        self._seq_frame_counts = []
        for i in range(self.dataset.num_sequences):
            info = self.dataset.get_sequence_info(i)
            n = int(info['valid'].sum().item())
            self._seq_frame_counts.append(max(n, 1))
        self._seq_p = np.array(self._seq_frame_counts, dtype=np.float64)
        self._seq_p /= self._seq_p.sum()
        print(f'VtuavSiameseDataset({split}): {self.dataset.num_sequences} seqs, '
              f'{samples_per_epoch} samples/epoch')


# ══════════════════════════════════════════════════════════════════════════════
# 训练主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser('RGBTSwinTrack-EncPromptLoRA VTUAV 从头训练')
    parser.add_argument('--weight', type=str,
                        default=os.path.join(_PRJ_ROOT, 'SwinTrack-B-384.pth'),
                        help='backbone ImageNet 预训练权重')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--vtuav_root', type=str,
                        default=os.environ.get('VTUAV_HOME', '/root/RGBTData/VTUAV'))
    parser.add_argument('--data_root', type=str,
                        default=os.environ.get('SGTEST_DATA_ROOT', '/root/RGBTData'),
                        help='测试数据集根目录 (auto_test 用)')
    parser.add_argument('--model_type', type=str, default='enc_promptlora_vtuav',
                        choices=list(_MODEL_REGISTRY.keys()))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--samples_per_epoch', type=int, default=20000)
    parser.add_argument('--val_samples', type=int, default=4000,
                        help='每轮验证样本数 (VTUAV test_ST)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr', type=float, default=1e-5,
                        help='backbone 学习率 (0=全程冻结)')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='amp')
    parser.add_argument('--auto_test', action='store_true', default=True,
                        help='每 test_interval 轮自动测试 vtuav_st')
    parser.add_argument('--no_auto_test', action='store_false', dest='auto_test')
    parser.add_argument('--test_interval', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Output dir: {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ── 构建模型 (从头: 仅 backbone 加载 ImageNet 预训练) ──
    print(f'Building RGBTSwinTrack-EncPromptLoRA (model_type={args.model_type})...')
    model = build_model(args.model_type)

    start_epoch = 1
    resume_ckpt = None
    if args.resume:
        print(f'Resuming from: {args.resume}')
        resume_ckpt = torch.load(args.resume, map_location='cpu')
        load_state_dict_filtered(model, resume_ckpt.get('model', resume_ckpt))
        start_epoch = resume_ckpt.get('epoch', 1) + 1
        print(f'[INFO] 将从 epoch {start_epoch} 继续训练')
    else:
        if not os.path.isfile(args.weight):
            raise FileNotFoundError(f'预训练权重不存在: {args.weight}')
        print(f'Loading backbone ImageNet weights: {args.weight}')
        checkpoint = torch.load(args.weight, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        missing, skipped = load_state_dict_filtered(model, state_dict)
        if missing:
            print(f'[INFO] {len(missing)} 个键未匹配 (encoder/decoder/head/prompt '
                  f'随机初始化)')
        if skipped:
            print(f'[WARN] 跳过 {len(skipped)} 个形状不匹配的键')
    model.to(device)

    # ── 数据集 ──
    train_dataset = VtuavSiameseDataset(
        root=args.vtuav_root, split='train',
        samples_per_epoch=args.samples_per_epoch, seed=args.seed)
    val_dataset = VtuavSiameseDataset(
        root=args.vtuav_root, split='val_st',
        samples_per_epoch=args.val_samples, seed=args.seed + 1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=False, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=False, drop_last=False,
                            persistent_workers=args.workers > 0)

    # ── 优化器 ──
    backbone_params, other_params = [], []
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
        backbone_trainable = False
        for param in backbone_params:
            param.requires_grad = False
        args.freeze_backbone_epochs = args.epochs + 1

    if resume_ckpt is not None:
        if 'optimizer' in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt['optimizer'])
                print('[INFO] Optimizer state restored')
            except Exception as e:
                print(f'[WARN] Optimizer state restore failed: {e}')

    # ── 学习率: linear warmup + cosine decay ──
    warmup = max(0, args.warmup_epochs)
    total = max(1, args.epochs - warmup)
    lr_min_frac = 1e-3

    def lr_factor(epoch):  # epoch 从 1 开始
        if warmup > 0 and epoch <= warmup:
            return epoch / max(warmup, 1)
        progress = min(1.0, (epoch - warmup) / total)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return lr_min_frac + (1 - lr_min_frac) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: lr_factor(ep + 1))

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda',
                                  init_scale=2 ** 10)

    # ── resume: 同步 LR 调度器与 GradScaler 状态 ──
    if start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f'[INFO] LR scheduler synced to epoch {start_epoch} '
              f'(LR={optimizer.param_groups[0]["lr"]:.2e})')
    if resume_ckpt is not None and 'scaler' in resume_ckpt \
            and scaler.is_enabled():
        try:
            scaler.load_state_dict(resume_ckpt['scaler'])
            print('[INFO] GradScaler state restored')
        except Exception as e:
            print(f'[WARN] GradScaler state restore failed: {e}')

    feat_h, feat_w = 24, 24
    search_w, search_h = 384, 384

    print(f'Freeze backbone: {args.freeze_backbone_epochs} epochs '
          f'(trainable={"yes" if backbone_trainable else "no"})')
    print(f'Warmup: {warmup}, Cosine over {total} epochs')
    print(f'Backbone LR: {args.backbone_lr if backbone_trainable else 0}, '
          f'Other LR: {args.lr}')
    print(f'Training {start_epoch}-{args.epochs} epochs, '
          f'{len(train_loader)} iters/epoch, AMP={args.amp}')

    # ════════════════════════════════════════════════════════════════════════
    # 训练循环
    # ════════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, args.epochs + 1):
        bb_freeze = (backbone_trainable and args.freeze_backbone_epochs > 0
                     and epoch <= args.freeze_backbone_epochs)
        if backbone_trainable and args.freeze_backbone_epochs > 0:
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = not bb_freeze
            if epoch == start_epoch:
                print(f'Backbone frozen for first {args.freeze_backbone_epochs} epochs')
            if epoch == args.freeze_backbone_epochs + 1:
                print(f'Backbone UNFROZEN at epoch {epoch} (lr={args.backbone_lr})')

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

            cls_pred = output['class_score'].float()
            reg_pred = output['bbox'].float()
            B = cls_pred.shape[0]

            cls_target, reg_target = compute_targets(
                bbox_gt, (feat_h, feat_w), (search_w, search_h))
            pos_mask = cls_target > 0.5
            n_pos = pos_mask.sum().item()
            pos_ratio = n_pos / (B * feat_h * feat_w)

            cls_pred_f = cls_pred.view(B, -1)
            cls_target_f = cls_target.view(B, -1)
            if n_pos > 0:
                reg_pred_flat = reg_pred.view(B, feat_h * feat_w, 4)
                reg_target_flat = reg_target.view(B, feat_h * feat_w, 4)
                pos_mask_f = pos_mask.view(B, -1)
                iou_values = compute_iou(reg_pred_flat[pos_mask_f],
                                         reg_target_flat[pos_mask_f])
                cls_target_f = cls_target_f.clone()
                cls_target_f[pos_mask_f] = iou_values.detach()
                cls_pred_prob = cls_pred_f.clamp(1e-6, 1 - 1e-6)
                cls_loss = 2.0 * varifocal_loss(cls_pred_prob, cls_target_f)
                gl = 2.0 * giou_loss(reg_pred_flat[pos_mask_f],
                                     reg_target_flat[pos_mask_f])
                io = 1.0 - gl.item() / 2.0
            else:
                cls_pred_prob = cls_pred_f.clamp(1e-6, 1 - 1e-6)
                cls_loss = 2.0 * varifocal_loss(cls_pred_prob, cls_target_f)
                gl = torch.tensor(0.0, device=device)
                io = 0.

            loss = cls_loss + gl

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            et += loss.item()
            ecl += cls_loss.item()
            egl += gl.item()
            eio += io
            epr += pos_ratio
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
              f'Pos%:{epr/n_batches*100:.1f} '
              f'LR:{optimizer.param_groups[0]["lr"]:.2e}')

        # ── 验证 (VTUAV test_ST) ──
        model.eval()
        vt = vcl = vgl = vio = vpr = 0.
        with torch.no_grad():
            for z_rgb, z_tir, x_rgb, x_tir, bbox_gt in val_loader:
                z_rgb = z_rgb.to(device)
                z_tir = z_tir.to(device)
                x_rgb = x_rgb.to(device)
                x_tir = x_tir.to(device)
                bbox_gt = bbox_gt.to(device)
                with torch.amp.autocast('cuda', enabled=args.amp and device.type == 'cuda'):
                    output = model(z_rgb, z_tir, x_rgb, x_tir)
                cls_pred = output['class_score'].float()
                reg_pred = output['bbox'].float()
                B = cls_pred.shape[0]
                cls_target, reg_target = compute_targets(
                    bbox_gt, (feat_h, feat_w), (search_w, search_h))
                cls_pred_f = cls_pred.view(B, -1)
                cls_target_f = cls_target.view(B, -1)
                pm = cls_target > 0.5
                np_ = pm.sum().item()
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
                    gl = torch.tensor(0.0, device=device)
                    io = 0.
                vt += (cl + gl).item()
                vcl += cl.item()
                vgl += gl.item()
                vio += io
                vpr += rp
        n_val = len(val_loader)
        print(f'  Val(ST) | Loss:{vt/n_val:.4f} Cls:{vcl/n_val:.4f} '
              f'GIoU:{vgl/n_val:.4f} IoU:{vio/n_val:.3f} Pos%:{vpr/n_val*100:.1f}')

        # ── 保存 checkpoint (含 optimizer 与 GradScaler 状态, 支持精确续训) ──
        ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:03d}.pth')
        save_dict = {'epoch': epoch, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        if scaler.is_enabled():
            save_dict['scaler'] = scaler.state_dict()
        torch.save(save_dict, ckpt_path)
        print(f'  Checkpoint saved: {ckpt_path}')

        # ── 自动测试 vtuav_st ──
        if args.auto_test and epoch % args.test_interval == 0:
            test_result_dir = os.path.join(args.output_dir, 'testresults',
                                           f'epoch{epoch:03d}')
            test_py = os.path.join(_PRJ_ROOT, 'test_vtuav_promptlora.py')
            print(f'[AUTO_TEST] epoch{epoch:03d} 启动 vtuav_st 测试...', flush=True)
            t0 = time.time()
            try:
                ret = subprocess.run(
                    [sys.executable, '-u', test_py,
                     '--weight', ckpt_path,
                     '--model_type', args.model_type,
                     '--dataset', 'vtuav_st',
                     '--data_root', args.data_root,
                     '--save_dir', test_result_dir,
                     '--workers', '4'],
                    cwd=_PRJ_ROOT, capture_output=True, text=True, timeout=3600)
            except subprocess.TimeoutExpired:
                print(f'[AUTO_TEST] epoch{epoch:03d} 超时 (>60min), 跳过')
            else:
                print(f'[AUTO_TEST] epoch{epoch:03d} 完成, '
                      f'耗时 {(time.time()-t0)/60:.1f}min, 退出码={ret.returncode}',
                      flush=True)
                for line in ret.stdout.splitlines():
                    ls = line.strip()
                    if ls.startswith('[RESULT]') or ls.startswith('序列均值') \
                            or ls.startswith('帧加权'):
                        print(f'  {ls}', flush=True)
                if ret.returncode != 0:
                    for line in ret.stderr.splitlines()[-15:]:
                        ls = line.strip()
                        if ls and 'Warning' not in ls:
                            print(f'  [STDERR] {ls[:200]}', flush=True)

    # ── 最终保存 ──
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({'epoch': args.epochs, 'model': model.state_dict()}, final_path)
    print(f'Final model saved: {final_path}')
    print('Training complete!')


if __name__ == '__main__':
    main()
