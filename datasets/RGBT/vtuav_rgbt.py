"""
VTUAV / GTOT / RGBT210 / RGBT234 数据集加载器
===============================================================
参考 SGTrack 项目 (lib/train/dataset/vtuav.py, lib/test/evaluation/vtuavdataset.py,
RGBT_workspace/test.py) 中四个数据集的目录组织:

服务器数据根目录 (与 SGTrack 一致, 可用环境变量覆盖):
  VTUAV_HOME  = /root/RGBTData/VTUAV
  RGBTData 根 = /root/RGBTData

  VTUAV:   <root>/train/<seq>/rgb|ir + rgb.txt        (训练)
           <root>/test_ST/<group>/<seq>/rgb|ir        (短时测试, 两级目录)
           <root>/test_LT/<group>/<seq>/rgb|ir        (长时测试, 两级目录)
           rgb.txt: 8 列 (rgb x,y,w,h + ir x,y,w,h), 取前 4 列
           帧命名: frame_id*10+init_idx (init_frame.npy 映射)

  GTOT:    <root>/GTOT/<seq>/v|i + groundTruth_v.txt  (x1,y1,x2,y2)
  RGBT210: <root>/RGBT210/<seq>/visible|infrared + visible.txt  (x,y,w,h)
  RGBT234: <root>/RGBT234/<seq>/visible|infrared + visible.txt  (x,y,w,h)

VtuavRGBTDataset 接口与 LasHeRRGBTDataset 对齐:
  num_sequences / get_sequence_info(seq_id) / get_frame_paths(seq_id, frame_id)
  / get_seq_name(seq_id)
"""
import os
import numpy as np
import pandas
import torch

_MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
_INIT_FRAME_NPY = os.path.join(_MODULE_DIR, 'init_frame.npy')

# VTUAV 官方测试组 (参考 SGTrack vtuavdataset.py: ST 13 组 176 序列, LT 10 组 74 序列)
_VTUAV_ST_GROUPS = [f'test_ST_{i:03d}' for i in range(1, 14)]
_VTUAV_LT_GROUPS = [f'test_LT_{i:03d}' for i in range(1, 11)]

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def _load_init_idx():
    """加载 VTUAV init_idx 映射 (序列名 → 起始帧偏移), 缺失时返回空字典。"""
    if os.path.isfile(_INIT_FRAME_NPY):
        try:
            return np.load(_INIT_FRAME_NPY, allow_pickle=True).item()
        except Exception as e:
            print(f'[WARN] init_frame.npy 加载失败, 使用 init_idx=0: {e}')
    return {}


_INIT_IDX = _load_init_idx()


def _list_sorted_images(seq_dir, modal):
    """列出某模态目录下所有图像文件路径 (按文件名排序)。"""
    d = os.path.join(seq_dir, modal)
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.lower().endswith(_IMG_EXTS)
    )


def _read_rgb_txt(seq_dir):
    """读取 VTUAV rgb.txt, 返回 (N, 4) float32 (x, y, w, h)。"""
    gt_file = os.path.join(seq_dir, 'rgb.txt')
    if not os.path.isfile(gt_file):
        return np.zeros((0, 4), dtype=np.float32)
    gt = pandas.read_csv(gt_file, delimiter=' ', header=None,
                         dtype=np.float32, na_filter=False,
                         low_memory=False).values
    return gt[:, :4].astype(np.float32)


def build_vtuav_frame_paths(seq_dir, modal):
    """按 frame_id*10+init_idx 构造帧路径; 若映射缺失则回退排序文件名。"""
    seq_name = os.path.basename(seq_dir)
    init_idx = _INIT_IDX.get(seq_name, 0)
    d = os.path.join(seq_dir, modal)
    if not os.path.isdir(d):
        return []
    all_files = {f for f in os.listdir(d) if f.lower().endswith(_IMG_EXTS)}
    paths = []
    fid = 0
    while fid < 20000:
        name = f'{fid * 10 + init_idx:06d}'
        hit = next((f for f in all_files if f.startswith(name + '.')), None)
        if hit is None:
            # 尝试未填充数字文件名 (部分服务器版本无 6 位填充)
            hit = next((f for f in all_files
                        if f.startswith(str(fid * 10 + init_idx) + '.')), None)
        if hit is None:
            break
        paths.append(os.path.join(d, hit))
        fid += 1
    if not paths:
        # 回退: 排序文件名 (不依赖 init_idx)
        paths = sorted(
            os.path.join(d, f) for f in all_files
            if f.lower().endswith(_IMG_EXTS)
        )
    return paths


def _discover_vtuav_seq_dirs(root):
    """发现 VTUAV 序列目录, 兼容两种布局:
      1) 嵌套: <root>/<group>/<seq>/rgb   (官方 test_ST_XXX/<seq> 结构)
      2) 扁平: <root>/<seq>/rgb
    返回 [(rel_name, seq_dir)], 按名称排序。"""
    found = []
    for entry in sorted(os.listdir(root)):
        p = os.path.join(root, entry)
        if not os.path.isdir(p):
            continue
        if os.path.isdir(os.path.join(p, 'rgb')):
            found.append((entry, p))
        else:
            for sub in sorted(os.listdir(p)):
                sp = os.path.join(p, sub)
                if os.path.isdir(sp) and os.path.isdir(os.path.join(sp, 'rgb')):
                    found.append((f'{entry}/{sub}', sp))
    return found


class VtuavRGBTDataset:
    """VTUAV RGBT 数据集 (训练 + 短时/长时测试)。

    split:
      train  → <root>/train  (一级序列目录)
      val_st → <root>/test_ST (兼容 两级嵌套/扁平 布局)
      val_lt → <root>/test_LT (兼容 两级嵌套/扁平 布局)
    """

    def __init__(self, root: str, split: str = 'train'):
        if split == 'train':
            self.root = os.path.join(root, 'train')
        elif split in ('val_st', 'val_lt'):
            sub = 'test_ST' if split == 'val_st' else 'test_LT'
            self.root = os.path.join(root, sub)
        else:
            raise ValueError(f'VTUAV 无 split={split}, 支持 train/val_st/val_lt')
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f'VTUAV 数据目录不存在: {self.root} (检查 VTUAV_HOME)')
        self.split = split

        # 相对路径 (组/序列 或 序列)
        self._seq_rel = [rel for rel, _ in _discover_vtuav_seq_dirs(self.root)]

        self.num_sequences = len(self._seq_rel)
        if self.num_sequences == 0:
            top = os.listdir(self.root)
            raise RuntimeError(
                f'VTUAV {split}: 未找到任何序列 ({self.root}), '
                f'顶层目录内容: {top[:20]}')

        # 逐序列缓存: bbox (N,4), rgb/ir 路径列表
        self._bboxes = []
        self._rgb_paths = []
        self._ir_paths = []
        for rel in self._seq_rel:
            sd = os.path.join(self.root, rel)
            bbox = _read_rgb_txt(sd)
            rgb_paths = build_vtuav_frame_paths(sd, 'rgb')
            ir_paths = build_vtuav_frame_paths(sd, 'ir')
            self._bboxes.append(bbox)
            self._rgb_paths.append(rgb_paths)
            self._ir_paths.append(ir_paths)
        print(f'VTUAV {split}: {self.num_sequences} sequences loaded '
              f'({self.root})')

    def get_sequence_info(self, seq_id: int) -> dict:
        """返回 bbox (N,4) float32 tensor, valid (N,) bool tensor, num_frames。
        (接口与 LasHeRRGBTDataset 对齐, 供 Siamese 采样复用)"""
        bbox = torch.tensor(self._bboxes[seq_id])
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        return {'bbox': bbox, 'valid': valid, 'num_frames': len(bbox)}

    def get_frame_paths(self, seq_id: int, frame_id: int):
        rgb = self._rgb_paths[seq_id]
        ir = self._ir_paths[seq_id]
        frame_id = min(frame_id, len(rgb) - 1, len(ir) - 1)
        return rgb[frame_id], ir[frame_id]

    def get_seq_name(self, seq_id: int) -> str:
        return os.path.basename(self._seq_rel[seq_id])


# ══════════════════════════════════════════════════════════════════════════════
# 测试集序列收集 (GTOT / RGBT210 / RGBT234 / VTUAV ST / VTUAV LT)
# 参考 SGTrack RGBT_workspace/test.py 的 DATASET_CFG 与读取逻辑
# ══════════════════════════════════════════════════════════════════════════════

def _read_gt_file(path, convert_xyxy_to_xywh=False):
    """读取 GT 文件 (支持空格/逗号/制表符分隔), 返回 list[list[float]] (x,y,w,h)。"""
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
            vals = [float(v) for v in parts[:4]]
            if convert_xyxy_to_xywh:
                vals = [vals[0], vals[1], vals[2] - vals[0], vals[3] - vals[1]]
            bboxes.append(vals)
        except ValueError:
            continue
    return bboxes


def collect_test_sequences(dataset: str, root: str):
    """收集测试序列。

    dataset ∈ {gtot, rgbt210, rgbt234, vtuav_st, vtuav_lt}
    root: 服务器 RGBT 数据根目录 (如 /root/RGBTData)
    返回: list[dict(seq_name, rgb_paths, tir_paths, bbox(N,4))]
    """
    if dataset == 'gtot':
        base = os.path.join(root, 'GTOT')
        if not os.path.isdir(base):
            print(f'[WARN] 目录不存在: {base}')
            return
        for s in sorted(os.listdir(base)):
            sd = os.path.join(base, s)
            if not os.path.isdir(sd):
                continue
            rgb = _list_sorted_images(sd, 'v')
            tir = _list_sorted_images(sd, 'i')
            gt = _read_gt_file(os.path.join(sd, 'groundTruth_v.txt'),
                               convert_xyxy_to_xywh=True)
            if not gt:
                gt = _read_gt_file(os.path.join(sd, 'groundTruth.txt'),
                                   convert_xyxy_to_xywh=True)
            if rgb and tir and gt:
                yield {'seq_name': s, 'rgb_paths': rgb, 'tir_paths': tir,
                       'bbox': np.asarray(gt, dtype=np.float32)}
    elif dataset in ('rgbt210', 'rgbt234'):
        sub = 'RGBT210' if dataset == 'rgbt210' else 'RGBT234'
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            print(f'[WARN] 目录不存在: {base}')
            return
        for s in sorted(os.listdir(base)):
            sd = os.path.join(base, s)
            if not os.path.isdir(sd):
                continue
            rgb = _list_sorted_images(sd, 'visible')
            tir = _list_sorted_images(sd, 'infrared')
            gt = _read_gt_file(os.path.join(sd, 'visible.txt'))
            if not gt:
                gt = _read_gt_file(os.path.join(sd, 'init.txt'))
            if rgb and tir and gt:
                yield {'seq_name': s, 'rgb_paths': rgb, 'tir_paths': tir,
                       'bbox': np.asarray(gt, dtype=np.float32)}
    elif dataset in ('vtuav_st', 'vtuav_lt'):
        sub = 'test_ST' if dataset == 'vtuav_st' else 'test_LT'
        base = os.path.join(root, 'VTUAV', sub)
        if not os.path.isdir(base):
            print(f'[WARN] 目录不存在: {base}')
            return
        for rel, sd in _discover_vtuav_seq_dirs(base):
            rgb = build_vtuav_frame_paths(sd, 'rgb')
            tir = build_vtuav_frame_paths(sd, 'ir')
            gt = _read_rgb_txt(sd)
            if rgb and tir and len(gt) > 0:
                yield {'seq_name': rel, 'rgb_paths': rgb,
                       'tir_paths': tir, 'bbox': gt}
    else:
        raise ValueError(f'未知数据集: {dataset} (支持 gtot/rgbt210/rgbt234/'
                         f'vtuav_st/vtuav_lt)')
