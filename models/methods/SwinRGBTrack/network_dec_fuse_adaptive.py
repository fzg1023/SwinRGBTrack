"""
RGBTSwinTrack-DecFuse-Adaptive — 场景自适应门控融合
=====================================================
Step 2: 根据当前输入动态决定每通道信 RGB 还是 TIR

  pooled_r = avg_pool(dec_x_r)      (B,512)
  pooled_t = avg_pool(dec_x_t)      (B,512)
  α = sigmoid( MLP([pooled_r ∥ pooled_t]) )   (B,512)
  x_fused = α⊙dec_x_r + (1-α)⊙dec_x_t

约 0.4M 新增参数, 从 DecFuse 最优权重初始化。
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class RGBTSwinTrackDecFuseAdaptive(nn.Module):
    """场景自适应门控融合 — 不同场景自动调整 RGB/TIR 偏好。"""

    def __init__(self, backbone, encoder, decoder, out_norm, head,
                 z_backbone_out_stage, x_backbone_out_stage,
                 z_input_projection, x_input_projection,
                 z_pos_enc, x_pos_enc, dim):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.out_norm = out_norm
        self.head = head
        self.z_backbone_out_stage = z_backbone_out_stage
        self.x_backbone_out_stage = x_backbone_out_stage
        self.z_input_projection = z_input_projection
        self.x_input_projection = x_input_projection
        self.z_pos_enc = z_pos_enc
        self.x_pos_enc = x_pos_enc

        # ── 场景自适应门控 MLP ──
        hidden = dim // 2  # 256
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )
        # 初始化为接近 0 → sigmoid ≈ 0.5 → 等价于原始 DecFuse
        self.gate_mlp[-1].weight.data.zero_()
        self.gate_mlp[-1].bias.data.zero_()
        self.reset_parameters()

    def reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m is not self.gate_mlp[-1]:  # 不覆盖 gate 最后层的零初始化
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        if self.z_input_projection is not None:
            self.z_input_projection.apply(_init_weights)
        if self.x_input_projection is not None:
            self.x_input_projection.apply(_init_weights)
        self.encoder.apply(_init_weights)
        self.decoder.apply(_init_weights)

    def _backbone_feat(self, x, stage):
        feat, = self.backbone(x, (stage,), False)
        return feat

    def _get_template_feat(self, z_rgb, z_tir):
        z_r = self._backbone_feat(z_rgb, self.z_backbone_out_stage)
        z_t = self._backbone_feat(z_tir, self.z_backbone_out_stage)
        if self.z_input_projection is not None:
            z_r = self.z_input_projection(z_r); z_t = self.z_input_projection(z_t)
        return z_r, z_t

    def _get_search_feat(self, x_rgb, x_tir):
        x_r = self._backbone_feat(x_rgb, self.x_backbone_out_stage)
        x_t = self._backbone_feat(x_tir, self.x_backbone_out_stage)
        if self.x_input_projection is not None:
            x_r = self.x_input_projection(x_r); x_t = self.x_input_projection(x_t)
        return x_r, x_t

    def _encode_decode(self, z_feat, x_feat):
        z_pos = self.z_pos_enc().unsqueeze(0) if self.z_pos_enc is not None else None
        x_pos = self.x_pos_enc().unsqueeze(0) if self.x_pos_enc is not None else None
        enc_z, enc_x = self.encoder(z_feat, x_feat, z_pos, x_pos)
        return self.decoder(enc_z, enc_x, z_pos, x_pos)

    def initialize(self, z_rgb, z_tir):
        return self._get_template_feat(z_rgb, z_tir)

    def track(self, cached, x_rgb, x_tir):
        z_r, z_t = cached
        x_r, x_t = self._get_search_feat(x_rgb, x_tir)
        dec_x_r = self._encode_decode(z_r, x_r)
        dec_x_t = self._encode_decode(z_t, x_t)

        # ── 场景自适应门控 ──
        # 全局空间池化 → 拼接 → MLP → sigmoid
        pool_r = dec_x_r.mean(dim=1)  # (B, 512)
        pool_t = dec_x_t.mean(dim=1)  # (B, 512)
        alpha = self.gate_mlp(torch.cat([pool_r, pool_t], dim=-1)).sigmoid()  # (B, 512)
        # 逐通道、逐样本融合
        x_fused = alpha.unsqueeze(1) * dec_x_r + (1 - alpha.unsqueeze(1)) * dec_x_t

        x = self.out_norm(x_fused)
        return self.head(x)

    def forward(self, z_rgb, z_tir, x_rgb=None, x_tir=None, cached=None):
        if cached is None:
            cached = self.initialize(z_rgb, z_tir)
        if x_rgb is not None and x_tir is not None:
            return self.track(cached, x_rgb, x_tir)
        return cached
