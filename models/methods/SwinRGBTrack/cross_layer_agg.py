"""
CrossLayerAggregation — 跨层特征聚合模块

来自 DropFuseRGBT 的设计:
  1. 丢弃模板 token (前 z_len 个)
  2. 所有层等权逐元素求和
  3. LayerNorm 归一化
"""
import torch
import torch.nn as nn


class CrossLayerAggregation(nn.Module):
    """跨层聚合: 丢弃模板token → 等权求和 → LayerNorm。

    Args:
        dim: 特征维度
        z_len: 模板 token 数量 (12×12=144)
    """

    def __init__(self, dim, z_len=144):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.z_len = z_len

    def forward(self, fused_feats):
        """聚合多层融合特征。

        Args:
            fused_feats: List of (B, L_z+L_x, C), 每层一个融合特征
        Returns:
            aggregated_search: (B, L_x, C) 仅搜索区的聚合特征
        """
        # 1) 丢弃模板区 token, 只保留搜索区
        search_feats = [f[:, self.z_len:] for f in fused_feats]

        # 2) 等权求和
        aggregated = torch.stack(search_feats, dim=0).sum(dim=0)

        # 3) LayerNorm
        return self.norm(aggregated)
