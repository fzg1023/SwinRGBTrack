"""
LasHeR RGBT 数据集 — 参考 OmniAdapt 的 mmap 缓存 + 文件索引方式
===============================================================
特性:
  - 内存映射 (mmap) 元数据缓存: 序列名 / 帧路径 / bbox / valid mask
  - 跳过 os.listdir 开销: 帧路径预序列化在 mmap 中
  - Bbox 预加载缓冲: 标注一次性读入 numpy npz
  - 返回分离的 RGB (3ch) 和 TIR (3ch) 图像
"""
import os
import hashlib
import numpy as np
import torch
import pickle
from typing import List, Tuple
import pandas


def _get_mmap_cache_path(root, split):
    h = hashlib.md5(root.encode()).hexdigest()[:16]
    return os.path.join(root, f'lasher_{split}_{h}.npz')


# 支持两种根目录布局: 老式 <root>/{trainingset,testingset}, 官方 <root>/{train,test}
_SPLIT_SUBDIRS = {
    'train': ('trainingset', 'train'),
    'val': ('testingset', 'test'),
    'all': ('trainingset', 'train'),
}


def _resolve_split_root(root, split):
    """把 split 解析到实际序列目录, 兼容 trainingset/testingset 与 train/test。"""
    if root is None:
        return None
    root = os.path.abspath(root)
    # root 本身已是序列目录 (含 init.txt) → 直接用
    if os.path.isfile(os.path.join(root, 'init.txt')):
        return root
    # root 已直接是某 split 的序列集合 (含多个序列子目录) → 判定其子目录有 init.txt
    for cand in _SPLIT_SUBDIRS.get(split, ()):
        sub = os.path.join(root, cand)
        if os.path.isdir(sub):
            return sub
    # 无法判定: 若 root 下没有子目录但自身像序列集则用 root
    return root


def _load_or_scan_spec(root, split):
    """读取 data_specs 中的序列清单; 若文件缺失或多数序列不在 root 下,
    则回退为扫描 root 下所有含 init.txt 的目录 (保证服务器官方 train/test 布局可用)。"""
    spec_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data_specs')
    spec_file = os.path.join(spec_dir, f'lasher_{split}.txt')
    if os.path.isfile(spec_file):
        with open(spec_file, 'r') as f:
            all_seqs = [l.strip() for l in f if l.strip()]
        # 命中率检查: 至少 50% 序列在 root 下才信任 spec
        hit = sum(1 for s in all_seqs if os.path.isdir(os.path.join(root, s)))
        if hit >= max(1, len(all_seqs) // 2):
            return all_seqs
        print(f"[WARN] spec {spec_file} 与目录 {root} 匹配率低 "
              f"({hit}/{len(all_seqs)}), 回退为目录扫描")
    else:
        print(f"[WARN] 无 spec 文件 {spec_file}, 回退为目录扫描")
    seqs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and os.path.isfile(os.path.join(root, d, 'init.txt')))
    return seqs


def build_lasher_mmap_cache(root, split):
    """一次性构建 mmap 元数据缓存 (~100MB, 启动 <1 分钟)。"""
    import tqdm
    cache_path = _get_mmap_cache_path(root, split)
    if os.path.exists(cache_path):
        print(f"Mmap cache already exists: {cache_path}")
        return cache_path

    print(f"Building LasHeR mmap metadata cache for split={split} ...")
    all_seqs = _load_or_scan_spec(root, split)

    seq_names = []
    seq_vis_paths = []
    seq_ir_paths = []
    seq_bboxes = []
    seq_valids = []

    for seq_name in tqdm.tqdm(all_seqs, desc='Caching metadata'):
        seq_dir = os.path.join(root, seq_name)
        init_file = os.path.join(seq_dir, 'init.txt')
        if not os.path.isdir(seq_dir) or not os.path.isfile(init_file):
            continue

        bbox = pandas.read_csv(init_file, delimiter=',', header=None,
                               dtype=np.float32, na_filter=False, low_memory=False).values
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)

        vis_dir = os.path.join(seq_dir, 'visible')
        ir_dir = os.path.join(seq_dir, 'infrared')
        vis_files = sorted(os.listdir(vis_dir))
        ir_files = sorted(os.listdir(ir_dir))

        seq_names.append(seq_name)
        seq_vis_paths.append(pickle.dumps([os.path.join(seq_dir, 'visible', v) for v in vis_files]))
        seq_ir_paths.append(pickle.dumps([os.path.join(seq_dir, 'infrared', v) for v in ir_files]))
        seq_bboxes.append(bbox)
        seq_valids.append(valid.astype(np.uint8))

    np.savez_compressed(cache_path,
                        seq_names=np.array(seq_names, dtype=object),
                        vis_paths=np.array(seq_vis_paths, dtype=object),
                        ir_paths=np.array(seq_ir_paths, dtype=object),
                        bboxes=np.array(seq_bboxes, dtype=object),
                        valids=np.array(seq_valids, dtype=object))
    size_mb = os.path.getsize(cache_path) / 1024**2
    print(f"Cache built: {cache_path} ({size_mb:.0f} MB, {len(seq_names)} sequences)")
    return cache_path


class LasHeRRGBTDataset:
    """LasHeR RGBT 数据集 (mmap 缓存 + 文件索引 + bbox 缓冲)。

    返回分离的 RGB 和 TIR 图像 (均为 3 通道 uint8 numpy)。
    """

    def __init__(self, root: str, split: str = 'train'):
        # 自动映射: train→trainingset|train, val→testingset|test (兼容两种布局)
        self.root = _resolve_split_root(root, split)
        if self.root is None:
            raise FileNotFoundError(f'LasHeR root 无效: {root}')
        self.split = split

        # 删除旧缓存 (如果 root 发生了变化需要重建)
        # 先检查缓存是否存在
        mmap_path = _get_mmap_cache_path(self.root, split)
        if not os.path.exists(mmap_path):
            build_lasher_mmap_cache(self.root, split)
        _c = np.load(mmap_path, allow_pickle=True)
        self._seq_names = _c['seq_names']
        self._vis_paths = _c['vis_paths']
        self._ir_paths = _c['ir_paths']
        self._bboxes = _c['bboxes']
        self._valids = _c['valids']

        self.num_sequences = len(self._seq_names)
        print(f"LasHeR {split}: {self.num_sequences} sequences loaded.")

    def get_sequence_info(self, seq_id: int) -> dict:
        """返回 bbox (N,4) float32, valid (N,) bool, num_frames (int)。"""
        bbox = torch.tensor(self._bboxes[seq_id])
        valid = torch.tensor(self._valids[seq_id].astype(bool))
        return {'bbox': bbox, 'valid': valid, 'num_frames': len(bbox)}

    def get_frame_paths(self, seq_id: int, frame_id: int) -> Tuple[str, str]:
        """返回 (rgb_path, tir_path)。"""
        vis_paths = pickle.loads(self._vis_paths[seq_id])
        ir_paths = pickle.loads(self._ir_paths[seq_id])
        # 防止 bbox 帧数 > 图像帧数导致的越界
        frame_id = min(frame_id, len(vis_paths) - 1, len(ir_paths) - 1)
        return vis_paths[frame_id], ir_paths[frame_id]

    def get_seq_name(self, seq_id: int) -> str:
        return str(self._seq_names[seq_id])
