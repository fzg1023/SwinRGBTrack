#!/usr/bin/env python3
"""
RGBTSwinTrack-LayerFuse — LasHeR RGBT 测试 (实时输出)
======================================================
Encoder逐层融合 + 跨层聚合 + Decoder
每序列实时打印 AO/SS/SR50/SR75/PS/NPS + 累计均值
"""
from __future__ import annotations
import argparse, csv, json, math, multiprocessing as mp, os, sys, time, traceback
from typing import Dict, List
import cv2, numpy as np, torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)

_LASHER_ROOT = os.environ.get('LASHER_ROOT', '/home/fzg/data/lasher')
_LASHER_TEST_ROOT = os.path.join(_LASHER_ROOT, 'testingset')

MODEL_CONFIG = {
    'template_size': [192, 192], 'search_size': [384, 384],
    'template_area_factor': 2.0, 'search_area_factor': 4.0,
    'window_penalty': 0.49,
    'backbone_name': 'swin_base_patch4_window12_384_in22k',
    'backbone_out_stage': 2, 'transformer_dim': 512, 'num_heads': 8,
    'mlp_ratio': 4, 'encoder_num_layers': 8, 'decoder_num_layers': 1,
    'template_feat_shape': [12, 12], 'search_feat_shape': [24, 24],
}

_METRICS = ['AO', 'SS', 'SR50', 'SR75', 'PS', 'NPS']
_IMAGENET_MEAN = torch.tensor(IMAGENET_DEFAULT_MEAN)
_IMAGENET_STD = torch.tensor(IMAGENET_DEFAULT_STD)
_TIR_MEAN = torch.tensor([0.449, 0.449, 0.449])
_TIR_STD = torch.tensor([0.226, 0.226, 0.226])
_DEFAULT_WEIGHT = os.path.join(_PRJ_ROOT, 'SwinTrack-B-384.pth')

# ── 复用 DecFuse 测试脚本的 tool 函数 ──
from test_lasher_rgbt_dec_fuse import (
    list_frames, read_frame_rgb, read_frame_tir, read_bboxes,
    count_frames, get_sequence_dirs,
    compute_curation_params, crop_and_resize, map_bbox_to_original,
    clamp_bbox, iou, compute_metrics, save_preds,
)


def load_model(weight_path, device):
    from core.run.event_dispatcher.register import EventRegister
    from models.methods.SwinRGBTrack.builder_layer_fuse import build_rgbt_layer_fuse
    from miscellanies.yaml_ops import load_yaml
    config = load_yaml(os.path.join(_PRJ_ROOT, 'config', 'SwinRGBTrack', 'Base-384-layer-fuse', 'config.yaml'))
    er = EventRegister('model/')
    model = build_rgbt_layer_fuse(config, False, 1, 1, er, False)
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    remapped_sd = {}
    for k, v in state_dict.items():
        if k.startswith('module.'): k = k[7:]
        if k.startswith('encoder.'): k = 'layer_fuse_encoder.' + k[8:]
        remapped_sd[k] = v
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in remapped_sd.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
    model.load_state_dict(filtered_state, strict=False)
    model.to(device); model.eval()
    return model


def _worker(worker_id, seq_dirs, result_dir, weight_path, gpu_id, out_q):
    try:
        device = torch.device(f'cuda:{gpu_id}') if gpu_id >= 0 and torch.cuda.is_available() else torch.device('cpu')
        model = load_model(weight_path, device)
        cfg = MODEL_CONFIG
        t_sz = tuple(cfg['template_size']); s_sz = tuple(cfg['search_size'])
        taf, saf = cfg['template_area_factor'], cfg['search_area_factor']
        wp = cfg['window_penalty']; fh, fw = cfg['search_feat_shape']
        hann = torch.outer(torch.hann_window(fh, periodic=False),
                           torch.hann_window(fw, periodic=False)).flatten().to(device)
        rgb_norm = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        tir_norm = transforms.Normalize(mean=_TIR_MEAN, std=_TIR_STD)
        out_q.put(('ready', worker_id))

        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            t0 = time.perf_counter()
            try:
                vpaths = list_frames(seq_dir, 'visible')
                ipaths = list_frames(seq_dir, 'infrared')
                nf = min(len(vpaths), len(ipaths))
                if nf == 0:
                    out_q.put(('result', worker_id, dict(seq_name=seq_name, AO=-1., n_valid=0,
                               SS=-1., SR50=-1., SR75=-1., PS=-1., NPS=-1., fps=0., elapsed=time.perf_counter()-t0)))
                    continue
                bboxes = read_bboxes(os.path.join(seq_dir, 'init.txt'))
                if not bboxes or len(bboxes[0]) < 4:
                    out_q.put(('result', worker_id, dict(seq_name=seq_name, AO=-1., n_valid=0,
                               SS=-1., SR50=-1., SR75=-1., PS=-1., NPS=-1., fps=0., elapsed=time.perf_counter()-t0)))
                    continue
                v0 = read_frame_rgb(vpaths[0])
                init_bbox = clamp_bbox(bboxes[0], v0.shape[1], v0.shape[0])
                tp = compute_curation_params(init_bbox, taf, t_sz)
                i0 = read_frame_tir(ipaths[0])
                z_rgb_np, zm_rgb = crop_and_resize(v0, t_sz, tp)
                z_tir_np, zm_tir = crop_and_resize(i0, t_sz, tp)
                z_rgb_t = rgb_norm(torch.from_numpy(z_rgb_np/255.).permute(2,0,1)).unsqueeze(0).to(device)
                z_tir_t = tir_norm(torch.from_numpy(z_tir_np/255.).permute(2,0,1)).unsqueeze(0).to(device)

                with torch.no_grad():
                    cached = model.initialize(z_rgb_t, z_tir_t)

                preds = [init_bbox]
                t_track = time.perf_counter()
                for fi in range(1, nf):
                    vf = read_frame_rgb(vpaths[fi]); itf = read_frame_tir(ipaths[fi])
                    sp = compute_curation_params(preds[-1], saf, s_sz)
                    x_rgb_np, _ = crop_and_resize(vf, s_sz, sp, image_mean=zm_rgb)
                    x_tir_np, _ = crop_and_resize(itf, s_sz, sp, image_mean=zm_tir)
                    x_rgb_t = rgb_norm(torch.from_numpy(x_rgb_np/255.).permute(2,0,1)).unsqueeze(0).to(device)
                    x_tir_t = tir_norm(torch.from_numpy(x_tir_np/255.).permute(2,0,1)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = model(cached=cached, z_rgb=z_rgb_t, z_tir=z_tir_t, x_rgb=x_rgb_t, x_tir=x_tir_t)
                    cls = out['class_score'].view(fh*fw)
                    reg = out['bbox'].view(fh*fw, 4)
                    cls_pen = cls*(1-wp) + hann*wp
                    best_idx = cls_pen.argmax().item()
                    rx, ry, rw, rh = reg[best_idx].cpu().tolist()
                    # CXCYWH → XYXY (像素坐标)
                    cx = rx*s_sz[0]; cy = ry*s_sz[1]
                    w = rw*s_sz[0]; h = rh*s_sz[1]
                    bbox_xyxy = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
                    pred_orig = map_bbox_to_original(bbox_xyxy, sp)
                    preds.append(clamp_bbox(pred_orig, v0.shape[1], v0.shape[0]))

                elapsed = time.perf_counter() - t0
                fps = (nf-1)/max(time.perf_counter()-t_track, 1e-6)
                gt_all = bboxes[:nf] if len(bboxes)>=nf else bboxes+[[0,0,0,0]]*(nf-len(bboxes))
                m = compute_metrics(preds, gt_all)
                m.update(seq_name=seq_name, fps=float(fps), elapsed=float(elapsed))
                save_preds(preds, seq_name, result_dir)
                out_q.put(('result', worker_id, m))
            except Exception:
                out_q.put(('seq_error', worker_id, seq_name, traceback.format_exc(), time.perf_counter()-t0))
    except Exception:
        out_q.put(('init_error', worker_id, traceback.format_exc()))


def main():
    mp.set_start_method('spawn', force=True)
    p = argparse.ArgumentParser('RGBTSwinTrack-LayerFuse LasHeR 测试')
    p.add_argument('--weight', default=_DEFAULT_WEIGHT)
    p.add_argument('--dataset_root', default=_LASHER_TEST_ROOT)
    p.add_argument('--save_dir', default=os.path.join(_PRJ_ROOT, 'test_results'))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--sequence', default='')
    args = p.parse_args()

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    save_name = 'swintrack_b384_layerfuse'
    result_dir = os.path.join(args.save_dir, 'lasher', save_name)
    os.makedirs(result_dir, exist_ok=True)

    print('=' * 78)
    print(f'  权重文件 : {args.weight}')
    print(f'  模态     : RGBT LayerFuse (Encoder逐层融合+跨层聚合)')
    print(f'  结果目录 : {result_dir}')
    print(f'  GPU 数量 : {num_gpus}')
    print('=' * 78)

    seq_dirs = get_sequence_dirs(args.dataset_root)
    if args.sequence:
        seq_dirs = [d for d in seq_dirs if os.path.basename(d) == args.sequence]
    n_seqs = len(seq_dirs)
    if n_seqs == 0:
        print('[ERROR] 没有找到任何序列'); sys.exit(1)

    # ── 帧数均衡分配 ──
    sorted_dirs = sorted(seq_dirs, key=lambda d: count_frames(d, 'visible'))
    n_workers = max(1, args.workers)
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

    HDR = (f"\n{'#':<6} {'序列名':<30} "
           f"{'AO':>6} {'SS':>6} {'SR50':>6} {'SR75':>6} "
           f"{'PS':>6} {'NPS':>6} {'FPS':>7} {'耗时s':>6} "
           f"{'avgAO':>7} {'avgSR50':>7} {'avgSS':>7} {'avgPS':>7}")
    SEP = '-' * (len(HDR)+1)
    print(HDR); print(SEP)

    out_q = mp.Queue()
    procs = []
    for wid, chunk in enumerate(chunks):
        gpu_id = 0 if num_gpus > 0 else -1
        proc = mp.Process(target=_worker, args=(wid, chunk, result_dir, args.weight, gpu_id, out_q), daemon=True)
        proc.start(); procs.append(proc)

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
    ao_cum, sr50_cum, ss_cum, ps_cum = 0., 0., 0., 0.
    n_valid_cum = 0

    while done < n_seqs:
        msg = out_q.get()
        if msg[0] == 'result':
            _, wid, m = msg
            done += 1
            all_recs[m['seq_name']] = m
            if m['AO'] >= 0:
                ao_cum += m['AO']; sr50_cum += m['SR50']
                ss_cum += m['SS']; ps_cum += m['PS']
                n_valid_cum += 1
                avg_ao = ao_cum / n_valid_cum; avg_sr50 = sr50_cum / n_valid_cum
                avg_ss = ss_cum / n_valid_cum; avg_ps = ps_cum / n_valid_cum
                print(f"[{done:03d}/{n_seqs:03d}] {m['seq_name']:<30s} "
                      f"{m['AO']:6.3f} {m['SS']:6.3f} {m['SR50']:6.3f} {m['SR75']:6.3f} "
                      f"{m['PS']:6.3f} {m['NPS']:6.3f} {m['fps']:7.1f} {m['elapsed']:6.1f} "
                      f"{avg_ao:7.3f} {avg_sr50:7.3f} {avg_ss:7.3f} {avg_ps:7.3f}", flush=True)
            else:
                print(f"[{done:03d}/{n_seqs:03d}] {m['seq_name']:<30s} "
                      f"{'---':>6} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'---':>6} "
                      f"{m.get('fps',0.):7.1f} {m.get('elapsed',0.):6.1f} {'---':>7} {'---':>7} {'---':>7} {'---':>7}", flush=True)
        elif msg[0] == 'seq_error':
            _, wid, seq_name, tb, elapsed = msg
            done += 1
            err_line = tb.strip().splitlines()[-1][:80] if tb.strip().splitlines() else 'unknown'
            print(f"[{done:03d}/{n_seqs:03d}] {seq_name:<30s} [ERROR] {err_line}  ({elapsed:.1f}s) w={wid}", flush=True)
            all_recs[seq_name] = dict(seq_name=seq_name, AO=-1., SS=-1., SR50=-1., SR75=-1.,
                                       PS=-1., NPS=-1., n_valid=0, fps=0., elapsed=elapsed)

    for proc in procs:
        proc.join(timeout=30)

    print(SEP)
    print()

    # ── 汇总 ──
    recs = list(all_recs.values())
    valid = [r for r in recs if r['AO'] >= 0]
    if valid:
        total_frames = sum(r.get('n_valid', 0) for r in valid)
        seq_ao = float(np.mean([r['AO'] for r in valid]))
        seq_ss = float(np.mean([r['SS'] for r in valid]))
        seq_sr50 = float(np.mean([r['SR50'] for r in valid]))
        seq_sr75 = float(np.mean([r['SR75'] for r in valid]))
        seq_ps = float(np.mean([r['PS'] for r in valid]))
        seq_nps = float(np.mean([r['NPS'] for r in valid]))
        frm_ao = float(sum(r['AO']*r.get('n_valid',1) for r in valid)/total_frames) if total_frames>0 else -1.
        mfps = float(np.mean([r['fps'] for r in valid if r['fps']>0]))
    else:
        seq_ao = seq_ss = seq_sr50 = seq_sr75 = seq_ps = seq_nps = frm_ao = -1.
        mfps = 0.

    print(f'  序列级: AO={seq_ao:.4f}  SS={seq_ss:.4f}  SR50={seq_sr50:.4f}  SR75={seq_sr75:.4f}')
    print(f'           PS={seq_ps:.4f}  NPS={seq_nps:.4f}')
    print(f'  帧级:   AO={frm_ao:.4f}')
    print(f'  平均 FPS: {mfps:.1f}')
    print(f'  有效序列: {len(valid)}/{n_seqs}')
    print('=' * 78)

    summary = dict(seq_AO=seq_ao, seq_SS=seq_ss, seq_SR50=seq_sr50, seq_SR75=seq_sr75,
                   seq_PS=seq_ps, seq_NPS=seq_nps, frm_AO=frm_ao, mean_fps=mfps,
                   n_valid=len(valid), n_total=n_seqs)
    with open(os.path.join(result_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(os.path.dirname(result_dir), 'eval_history.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['ckpt_tag', 'checkpoint', 'dataset', 'modality',
                             'n_valid', 'n_total', 'mean_fps',
                             'seq_AO', 'seq_SS', 'seq_SR50', 'seq_SR75', 'seq_PS', 'seq_NPS',
                             'frm_AO', 'frm_SS', 'frm_SR50', 'frm_SR75', 'frm_PS', 'frm_NPS'])
        writer.writerow([save_name, os.path.basename(args.weight), 'lasher', 'rgbt_layerfuse',
                         len(valid), n_seqs, f'{mfps:.2f}',
                         f'{seq_ao:.4f}', f'{seq_ss:.4f}', f'{seq_sr50:.4f}', f'{seq_sr75:.4f}',
                         f'{seq_ps:.4f}', f'{seq_nps:.4f}',
                         f'{frm_ao:.4f}', '-', '-', '-', '-', '-'])
    print(f'Results saved: {result_dir}')


if __name__ == '__main__':
    main()
