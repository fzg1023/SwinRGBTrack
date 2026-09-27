"""
BimodalLinearFuse — 跨模态线性融合模块

来自 DropFuseRGBT 的设计: Concat → Linear(2C→C) → LayerNorm → GELU
每层独立参数, 不共享权重。

注意: 不使用 DropPath, 因为融合模块无残差连接, 随机丢弃整个输出会破坏特征。
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class BimodalLinearFuse(nn.Module):
    """跨模态线性融合: 拼接 RGB+TIR 特征 → 线性投影 → LN → GELU。

    Args:
        dim: 单模态特征维度
    """

    def __init__(self, dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feat_rgb, feat_tir):
        """跨模态融合。

        Args:
            feat_rgb: (B, N, C) RGB 模态特征 (模板+搜索拼接)
            feat_tir: (B, N, C) TIR 模态特征 (模板+搜索拼接)
        Returns:
            fused: (B, N, C) 融合后的跨模态特征
        """
        return self.fuse(torch.cat([feat_rgb, feat_tir], dim=-1))
