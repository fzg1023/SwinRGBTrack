#!/usr/bin/env python3
"""
RGBTSwinTrack-DecFuse-Adaptive — LasHeR 训练 (Step 2: 场景自适应门控)
========================================================================
从 DecFuse ep5 初始化, gate_mlp 最后层零初始化 → σ(0)=0.5 等价 DecFuse
"""
from __future__ import annotations
import argparse, math, os, sys, time
from collections import OrderedDict
import cv2 as cv, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core.amp_compat  # AMP 兼容层 (服务器旧版 torch 无 torch.amp.GradScaler)
from torch.utils.data import DataLoader
from tqdm import tqdm

_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _PRJ_ROOT not in sys.path: sys.path.insert(0, _PRJ_ROOT)

from train_rgbt_dec_fuse import (
    LasHeRSiameseDataset, giou_loss, varifocal_loss, compute_iou, compute_targets,
)


def build_model(device):
    from core.run.event_dispatcher.register import EventRegister
    from models.methods.SwinRGBTrack.builder_dec_fuse_adaptive import build_rgbt_dec_fuse_adaptive
    from miscellanies.yaml_ops import load_yaml
    config = load_yaml(os.path.join(_PRJ_ROOT, 'config', 'SwinRGBTrack', 'Base-384-dec-fuse-adaptive', 'config.yaml'))
    return build_rgbt_dec_fuse_adaptive(config, False, 1, 1, EventRegister('model/'), True), config


def load_weights(model, weight_path):
    print(f"Loading weights: {weight_path}")
    ckpt = torch.load(weight_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'): k = k[7:]
        new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        gate_missing = [m for m in missing if 'gate_mlp' in m]
        other = [m for m in missing if 'gate_mlp' not in m]
        if gate_missing and not other:
            print(f"  [INFO] gate_mlp zero-init (expected), all other weights loaded")
        elif other:
            print(f"  [WARN] {len(other)} unexpected missing: {other[:3]}")
    if unexpected: print(f"  [INFO] {len(unexpected)} unexpected keys ignored")


def main():
    parser = argparse.ArgumentParser('DecFuse-Adaptive LasHeR 训练')
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--lasher_root', type=str, default=os.environ.get('LASHER_ROOT', '/home/fzg/data/lasher'))
    parser.add_argument('--batch_size', type=int, default=16); parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4); parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--freeze_epochs', type=int, default=2, help='前N轮只训练gate+head')
    parser.add_argument('--warmup_epochs', type=int, default=1); parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--amp', action='store_true', default=True); parser.add_argument('--no_amp', action='store_false', dest='amp')
    parser.add_argument('--workers', type=int, default=4); parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("  DecFuse-Adaptive (Step 2: 场景自适应门控)")
    print(f"  Weight: {args.weight}  Output: {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    print("Building model...")
    model, _ = build_model(device)
    load_weights(model, args.weight)
    model = model.to(device)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    gate_n = sum(p.numel() for n, p in model.named_parameters() if 'gate_mlp' in n) / 1e6
    print(f"  Params: {n:.1f}M  Gate: {gate_n:.2f}M")

    train_ds = LasHeRSiameseDataset(root=args.lasher_root, split='train',
                                     template_size=(192,192), search_size=(384,384),
                                     template_area_factor=2.0, search_area_factor=4.0,
                                     max_frame_gap=200, samples_per_epoch=60000)
    val_ds = LasHeRSiameseDataset(root=args.lasher_root, split='val',
                                   template_size=(192,192), search_size=(384,384),
                                   template_area_factor=2.0, search_area_factor=4.0,
                                   max_frame_gap=200, samples_per_epoch=4000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n]
    gate_params = [p for n, p in model.named_parameters() if 'gate_mlp' in n]
    other_params = [p for n, p in model.named_parameters() if 'backbone' not in n and 'gate_mlp' not in n]

    optimizer = torch.optim.AdamW([
        {'params': gate_params, 'lr': args.lr},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=1e-4)
    if backbone_params:
        optimizer.add_param_group({'params': backbone_params, 'lr': args.backbone_lr})

    try:
        from timm.scheduler import CosineLRScheduler
        scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=args.lr*1e-3,
                                       warmup_t=args.warmup_epochs, warmup_lr_init=args.lr*1e-2, warmup_prefix=True)
        use_timm_scheduler = True
    except ImportError:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        use_timm_scheduler = False

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if args.amp else None
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False); optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 1) + 1

    feat_h, feat_w = 24, 24; total_opt_steps = len(train_loader) // args.grad_accum
    print(f"Training {args.epochs} epochs, {len(train_loader)} iters/epoch")

    for epoch in range(start_epoch, args.epochs + 1):
        if args.freeze_epochs > 0:
            freeze = epoch <= args.freeze_epochs
            for name, param in model.named_parameters():
                if 'gate_mlp' in name or 'head' in name:
                    param.requires_grad = True
                elif 'backbone' in name:
                    param.requires_grad = not freeze
                elif 'encoder' in name or 'decoder' in name:
                    param.requires_grad = not freeze
            if epoch == 1:
                print(f"Epoch 1-{args.freeze_epochs}: gate+head trainable, others frozen")
            if epoch == args.freeze_epochs + 1:
                print(f"Epoch {epoch}: ALL unfrozen")

        model.train(); et = ecl = egl = eio = el1 = epr = 0.
        optimizer.zero_grad(); opt_step = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, (z_rgb, z_tir, x_rgb, x_tir, bbox_gt) in enumerate(pbar):
            z_rgb, z_tir = z_rgb.to(device, non_blocking=True), z_tir.to(device, non_blocking=True)
            x_rgb, x_tir = x_rgb.to(device, non_blocking=True), x_tir.to(device, non_blocking=True)
            bbox_gt = bbox_gt.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                output = model(z_rgb, z_tir, x_rgb, x_tir)
            cp = output['class_score'].float(); rp = output['bbox'].float(); B = cp.shape[0]
            ct, rt = compute_targets(bbox_gt, (feat_h, feat_w), (384, 384))
            pm = ct > 0.5; np_ = pm.sum().item(); pr = np_/(B*feat_h*feat_w)
            cpf, ctf = cp.view(B,-1), ct.view(B,-1)
            if np_ > 0:
                rpf2, rtf2 = rp.view(B,feat_h*feat_w,4), rt.view(B,feat_h*feat_w,4)
                pmf2 = pm.view(B,-1); iou_vals = compute_iou(rpf2[pmf2], rtf2[pmf2])
                ctf = ctf.clone(); ctf[pmf2] = iou_vals.detach()
                cl = 2.0*varifocal_loss(cpf.clamp(1e-6,1-1e-6), ctf)
                gl = giou_loss(rpf2[pmf2], rtf2[pmf2]); io = 1.0-gl.item()
                ll = torch.tensor(0.0, device=device)
            else:
                cl = 2.0*varifocal_loss(cpf.clamp(1e-6,1-1e-6), ctf)
                gl = torch.tensor(0.0,device=device); io=0.; ll=torch.tensor(0.0,device=device)
            loss = (cl+2.0*gl)/args.grad_accum
            if scaler is not None: scaler.scale(loss).backward()
            else: loss.backward()
            if (batch_idx+1)%args.grad_accum==0:
                if scaler is not None:
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                optimizer.zero_grad(); opt_step+=1
                if use_timm_scheduler: scheduler.step(epoch-1+opt_step/total_opt_steps)
            et+=loss.item(); ecl+=cl.item(); egl+=gl.item(); eio+=io; el1+=ll.item(); epr+=pr
            if (batch_idx+1)%50==0:
                n=batch_idx+1
                print(f'  Step {batch_idx+1:5d} | loss={et/n:.4f} cls={ecl/n:.4f} GIoU={egl/n:.4f} IoU={eio/n:.3f} L1={el1/n:.4f} Pos%={epr/n*100:.1f}', flush=True)

        if not use_timm_scheduler: scheduler.step()
        nb=len(train_loader)
        print(f'Epoch {epoch}/{args.epochs} | Loss:{et/nb:.4f} Cls:{ecl/nb:.4f} GIoU:{egl/nb:.4f} IoU:{eio/nb:.3f} L1:{el1/nb:.4f} Pos%:{epr/nb*100:.1f} LR:{optimizer.param_groups[0]["lr"]:.2e}')

        model.eval(); vt=vcl=vgl=vio=vl1=vpr=0.
        with torch.no_grad():
            for z_rgb,z_tir,x_rgb,x_tir,bbox_gt in val_loader:
                z_rgb,z_tir=z_rgb.to(device),z_tir.to(device)
                x_rgb,x_tir=x_rgb.to(device),x_tir.to(device); bbox_gt=bbox_gt.to(device)
                with torch.amp.autocast('cuda',enabled=args.amp):
                    output=model(z_rgb,z_tir,x_rgb,x_tir)
                cp=output['class_score'].float(); rp=output['bbox'].float(); B=cp.shape[0]
                ct,rt=compute_targets(bbox_gt,(feat_h,feat_w),(384,384))
                pm=ct>0.5; np_=pm.sum().item()
                cpf,ctf=cp.view(B,-1),ct.view(B,-1)
                if np_>0:
                    rpf2,rtf2=rp.view(B,feat_h*feat_w,4),rt.view(B,feat_h*feat_w,4)
                    pmf2=pm.view(B,-1); iou_vals2=compute_iou(rpf2[pmf2],rtf2[pmf2])
                    ctf=ctf.clone(); ctf[pmf2]=iou_vals2.detach()
                    cl2=2.0*varifocal_loss(cpf.clamp(1e-6,1-1e-6),ctf)
                    gl2=giou_loss(rpf2[pmf2],rtf2[pmf2]); io2=1.0-gl2.item()
                    ll2=torch.tensor(0.0,device=device)
                else:
                    cl2=2.0*varifocal_loss(cpf.clamp(1e-6,1-1e-6),ctf)
                    gl2=torch.tensor(0.0,device=device); io2=0.; ll2=torch.tensor(0.0,device=device)
                vl2=cl2.item()+2.0*gl2.item(); vt+=vl2; vcl+=cl2.item(); vgl+=gl2.item()
                vio+=io2; vl1+=ll2.item(); vpr+=np_/(B*feat_h*feat_w)
        nv=len(val_loader)
        print(f'  Val | Loss:{vt/nv:.4f} Cls:{vcl/nv:.4f} GIoU:{vgl/nv:.4f} IoU:{vio/nv:.3f} L1:{vl1/nv:.4f} Pos%:{vpr/nv*100:.1f}')

        torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict()},
                   os.path.join(args.output_dir,f'checkpoint_epoch{epoch:03d}.pth'))
        print(f'Checkpoint saved: checkpoint_epoch{epoch:03d}.pth')


if __name__ == '__main__':
    main()
