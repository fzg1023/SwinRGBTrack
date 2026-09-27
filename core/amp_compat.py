"""
AMP API 兼容层 — 服务器旧版 torch 兼容
===============================================================
新版 torch (>=1.13) 提供 torch.amp.GradScaler('cuda', ...) 与
torch.amp.autocast('cuda', ...); 旧版 torch (1.10~1.12) 的 torch.amp
模块下没有这些 API (只在 torch.cuda.amp 下)。

import 本模块即对 torch.amp 打补丁:
  - torch.amp.GradScaler('cuda', enabled=..., init_scale=...)
  - torch.amp.autocast('cuda', enabled=...)
旧版会透明映射到 torch.cuda.amp.GradScaler / autocast;
新版 torch 原样放行, 无任何副作用。
"""
import torch


def _patch():
    amp = getattr(torch, 'amp', None)
    if amp is None:
        # 极旧版本 (torch < 1.10): 直接用 torch.cuda.amp 顶替
        amp = torch.cuda.amp
        torch.amp = amp

    if hasattr(amp, 'GradScaler'):
        return  # 新版 API, 无需补丁

    from torch.cuda.amp import GradScaler as _GS, autocast as _AC

    class _GradScaler(_GS):
        def __init__(self, *args, **kwargs):
            # 新版签名 GradScaler(device, enabled=..., init_scale=...)
            # 旧版不接受 device 位置参数, 剥掉后透传
            if args and isinstance(args[0], str):
                args = args[1:]
            super().__init__(*args, **kwargs)

    def _autocast(*args, **kwargs):
        # 新版签名 autocast(device_type, enabled=...)
        # 旧版不接受 device_type 位置参数, 剥掉后透传
        if args and isinstance(args[0], str):
            args = args[1:]
        return _AC(*args, **kwargs)

    amp.GradScaler = _GradScaler
    amp.autocast = _autocast


_patch()
