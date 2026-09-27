#!/usr/bin/env python3
"""ConcatFuse 微调: 冻结 Backbone+Encoder+Decoder, 只训 concat融合+Head"""
import argparse, math, os, sys, time, random
import numpy as np, cv2 as cv, torch, torch.nn as nn, torch.nn.functional as F
import core.amp_compat  # AMP 兼容层 (服务器旧版 torch 无 torch.amp.GradScaler)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _PRJ_ROOT not in sys.path: sys.path.insert(0, _PRJ_ROOT)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]; _IMAGENET_STD = [0.229, 0.224, 0.225]
_TIR_MEAN = [0.485, 0.456, 0.406]; _TIR_STD = [0.229, 0.224, 0.225]

# Dataset (复用 AINet 管线)
from train_rgbt_dec_fuse import LasHeRSiameseDataset, compute_targets, compute_iou, varifocal_loss, giou_loss

def build_model():
    from core.run.event_dispatcher.register import EventRegister
    from models.methods.SwinRGBTrack.builder_concat_fuse import build_rgbt_concat_fuse
    from miscellanies.yaml_ops import load_yaml
    config = load_yaml(os.path.join(_PRJ_ROOT, 'config', 'SwinRGBTrack', 'Base-384-concat-fuse', 'config.yaml'))
    return build_rgbt_concat_fuse(config, False, 1, 1, EventRegister('model/'), True), config

def main():
    parser = argparse.ArgumentParser('ConcatFuse 微调 (fusion+head only)')
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--lasher_root', type=str, default=os.environ.get('LASHER_ROOT', '/home/fzg/data/lasher'))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Output: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    print("Building ConcatFuse (backbone/encoder/decoder frozen)...")
    model, config = build_model()

    print(f"Loading pretrained: {args.weight}")
    ckpt = torch.load(args.weight, map_location='cpu')
    sd = ckpt.get('model', ckpt)
    ms = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.to(device)

    # 冻结 backbone/encoder/decoder
    for name, param in model.named_parameters():
        if 'backbone' in name or 'encoder' in name or 'decoder' in name:
            param.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable: {sum(p.numel() for p in trainable):,} params (fusion_proj + head)")

    train_ds = LasHeRSiameseDataset(root=args.lasher_root, split='train', seed=args.seed)
    val_ds = LasHeRSiameseDataset(root=args.lasher_root, split='val', samples_per_epoch=4000, seed=args.seed+1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 1) + 1
        print(f"Resumed at epoch {start_epoch}")

    feat_h, feat_w = 24, 24
    print(f"Training {args.epochs} epochs, {len(train_loader)} iters/epoch, LR={args.lr}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); et=ecl=egl=eio=el1=epr=0.; optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, (z_rgb,z_tir,x_rgb,x_tir,bbox_gt) in enumerate(pbar):
            z_rgb=z_rgb.to(device);z_tir=z_tir.to(device)
            x_rgb=x_rgb.to(device);x_tir=x_tir.to(device);bbox_gt=bbox_gt.to(device)

            with torch.amp.autocast('cuda'):
                output = model(z_rgb, z_tir, x_rgb, x_tir)

            cls_pred=output['class_score'].float();reg_pred=output['bbox'].float();B=cls_pred.shape[0]
            cls_target,reg_target=compute_targets(bbox_gt,(feat_h,feat_w),(384,384))
            pos_mask=cls_target>0.5;n_pos=pos_mask.sum().item();pos_ratio=n_pos/(B*feat_h*feat_w)
            cls_pred_f=cls_pred.view(B,-1);cls_target_f=cls_target.view(B,-1)

            if n_pos>0:
                rpf=reg_pred.view(B,feat_h*feat_w,4);rtf=reg_target.view(B,feat_h*feat_w,4);pmf=pos_mask.view(B,-1)
                iou_vals=compute_iou(rpf[pmf],rtf[pmf])
                cls_target_f=cls_target_f.clone();cls_target_f[pmf]=iou_vals.detach()
                cls_loss=1.5*varifocal_loss(cls_pred_f.clamp(1e-6,1-1e-6),cls_target_f)
                gl=giou_loss(rpf[pmf],rtf[pmf]);io=1.0-gl.item();ll=torch.tensor(0.0,device=device)
            else:
                cls_loss=1.5*varifocal_loss(cls_pred_f.clamp(1e-6,1-1e-6),cls_target_f)
                gl=torch.tensor(0.0,device=device);io=0.;ll=torch.tensor(0.0,device=device)

            loss=cls_loss+1.5*gl
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.1)
            scaler.step(optimizer);scaler.update();optimizer.zero_grad()

            et+=loss.item();ecl+=cls_loss.item();egl+=gl.item();eio+=io;el1+=ll.item();epr+=pos_ratio
            if (batch_idx+1)%50==0:
                n=batch_idx+1
                print(f'  Step {batch_idx+1:5d} | loss={et/n:.4f} cls={ecl/n:.4f} GIoU={egl/n:.4f} IoU={eio/n:.3f} Pos%={epr/n*100:.1f}',flush=True)

        scheduler.step()
        nb=len(train_loader)
        print(f'Epoch {epoch}/{args.epochs} | Loss:{et/nb:.4f} Cls:{ecl/nb:.4f} GIoU:{egl/nb:.4f} IoU:{eio/nb:.3f} Pos%:{epr/nb*100:.1f} LR:{optimizer.param_groups[0]["lr"]:.2e}')

        # Val
        model.eval();vt=vcl=vgl=vio=vl1=vpr=0.
        with torch.no_grad():
            for z_rgb,z_tir,x_rgb,x_tir,bbox_gt in val_loader:
                z_rgb=z_rgb.to(device);z_tir=z_tir.to(device)
                x_rgb=x_rgb.to(device);x_tir=x_tir.to(device);bbox_gt=bbox_gt.to(device)
                output=model(z_rgb,z_tir,x_rgb,x_tir)
                cp=output['class_score'].float();rp=output['bbox'].float();B=cp.shape[0]
                ct,rt=compute_targets(bbox_gt,(feat_h,feat_w),(384,384))
                pm=ct>0.5;np_=pm.sum().item();rp_=np_/(B*feat_h*feat_w)
                cpf=cp.view(B,-1);ctf=ct.view(B,-1)
                if np_>0:
                    rpf=rp.view(B,feat_h*feat_w,4);rtf=rt.view(B,feat_h*feat_w,4);pmf=pm.view(B,-1)
                    iou_v=compute_iou(rpf[pmf],rtf[pmf]);ctf=ctf.clone();ctf[pmf]=iou_v.detach()
                    cl=1.5*varifocal_loss(cpf.clamp(1e-6,1-1e-6),ctf)
                    gl_=giou_loss(rpf[pmf],rtf[pmf]);io_=1.0-gl_.item();ll_=torch.tensor(0.0,device=device)
                else:
                    cl=1.5*varifocal_loss(cpf.clamp(1e-6,1-1e-6),ctf)
                    gl_=torch.tensor(0.0,device=device);io_=0.;ll_=torch.tensor(0.0,device=device)
                vt+=(cl+1.5*gl_).item();vcl+=cl.item();vgl+=gl_.item();vio+=io_;vl1+=ll_.item();vpr+=rp_
        nv=len(val_loader)
        print(f'  Val | Loss:{vt/nv:.4f} Cls:{vcl/nv:.4f} GIoU:{vgl/nv:.4f} IoU:{vio/nv:.3f} Pos%:{vpr/nv*100:.1f}')

        ckpt_path=os.path.join(args.output_dir,f'checkpoint_epoch{epoch:03d}.pth')
        torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict()},ckpt_path)

    torch.save({'model':model.state_dict()},os.path.join(args.output_dir,'final_model.pth'))
    print('Done!')

if __name__=='__main__': main()
