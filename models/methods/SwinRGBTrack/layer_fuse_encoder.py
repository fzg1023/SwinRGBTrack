"""
LayerFuseEncoder — 层间融合编码器

基于 SwinTrack ConcatenatedFusion Encoder, 增加:
  - RGB/TIR 双流并行处理 (共享自注意力权重)
  - 每层后 BimodalLinearFuse 跨模态融合
  - 跨层聚合 (CrossLayerAggregation)
"""
import torch
import torch.nn as nn


class LayerFuseEncoder(nn.Module):
    """层间融合编码器: 逐层自注意力 + 跨模态融合 + 跨层聚合。

    Args:
        self_attention_layers: SelfAttentionBlock 列表 (8层)
        cross_fuse_modules:   BimodalLinearFuse 列表 (8层, 每层一个)
        cross_layer_agg:      CrossLayerAggregation 模块
        z_untied_pos_enc:     Untied2DPositionalEncoder (模板)
        x_untied_pos_enc:     Untied2DPositionalEncoder (搜索)
        rpe_bias_table:       相对位置编码偏置表
        rpe_index:            相对位置编码索引
    """

    def __init__(self, self_attention_layers, cross_fuse_modules,
                 cross_layer_agg,
                 z_untied_pos_enc, x_untied_pos_enc,
                 rpe_bias_table, rpe_index):
        super().__init__()
        self.layers = nn.ModuleList(self_attention_layers)
        self.cross_fuse = nn.ModuleList(cross_fuse_modules)
        self.cross_layer_agg = cross_layer_agg
        self.z_untied_pos_enc = z_untied_pos_enc
        self.x_untied_pos_enc = x_untied_pos_enc
        if rpe_index is not None:
            self.register_buffer('rpe_index', rpe_index, False)
        self.rpe_bias_table = rpe_bias_table

    def _build_pos_encoding(self, z, x, z_pos, x_pos):
        """构建位置编码 (与 ConcatenatedFusion 一致)。"""
        attn_pos_enc = None
        if self.z_untied_pos_enc is not None:
            z_q_pos, z_k_pos = self.z_untied_pos_enc()
            x_q_pos, x_k_pos = self.x_untied_pos_enc()
            attn_pos_enc = (torch.cat((z_q_pos, x_q_pos), dim=1) @
                           torch.cat((z_k_pos, x_k_pos), dim=1).transpose(-2, -1)).unsqueeze(0)

        if self.rpe_bias_table is not None:
            if attn_pos_enc is not None:
                attn_pos_enc = attn_pos_enc + self.rpe_bias_table(self.rpe_index)
            else:
                attn_pos_enc = self.rpe_bias_table(self.rpe_index)

        concatenated_pos_enc = None
        if z_pos is not None:
            assert x_pos is not None
            concatenated_pos_enc = torch.cat((z_pos, x_pos), dim=1)
        return attn_pos_enc, concatenated_pos_enc

    def forward(self, z_rgb, x_rgb, z_tir, x_tir, z_pos, x_pos):
        """双流逐层处理 + 融合 + 聚合。

        Args:
            z_rgb/x_rgb: RGB 模板/搜索骨干特征
            z_tir/x_tir: TIR 模板/搜索骨干特征
            z_pos/x_pos: 位置编码 (可选)
        Returns:
            fused_z: (B, L_z, C) 聚合后的模板融合特征
            fused_x: (B, L_x, C) 聚合后的搜索融合特征
            all_fused: List of (B, L_z+L_x, C) 每层的融合特征 (用于调试)
        """
        L_z = z_rgb.shape[1]
        L_x = x_rgb.shape[1]

        # 拼接模板+搜索 → RGB 流 & TIR 流
        cat_r = torch.cat((z_rgb, x_rgb), dim=1)  # (B, L_z+L_x, C)
        cat_t = torch.cat((z_tir, x_tir), dim=1)

        # 位置编码 (两条流共享)
        attn_pos_enc, concatenated_pos_enc = self._build_pos_encoding(
            z_rgb, x_rgb, z_pos, x_pos)

        # 逐层处理 + 融合
        fused_feats = []
        for i, layer in enumerate(self.layers):
            # RGB 自注意力
            cat_r = layer(cat_r, concatenated_pos_enc, concatenated_pos_enc, attn_pos_enc)
            # TIR 自注意力 (共享权重)
            cat_t = layer(cat_t, concatenated_pos_enc, concatenated_pos_enc, attn_pos_enc)
            # 跨模态融合
            fused_i = self.cross_fuse[i](cat_r, cat_t)  # (B, L_z+L_x, C)
            fused_feats.append(fused_i)

        # 跨层聚合: 只保留搜索区
        fused_x = self.cross_layer_agg(fused_feats)  # (B, L_x, C)

        # 模板区聚合: 等权求和 + LayerNorm
        # 对每层的模板部分求平均, 用于 Decoder cross-attention
        template_feats = [f[:, :L_z] for f in fused_feats]
        fused_z = torch.stack(template_feats, dim=0).mean(dim=0)  # (B, L_z, C)

        return fused_z, fused_x, fused_feats
