"""
RGBTSwinTrack-DecFuse — Decoder 后融合的 RGBT 跟踪网络
========================================================
与 EncFuse 的区别:
  - 融合放在 Decoder 之后: Backbone → Encoder → Decoder → 0.5 Fuse → Head
  - 每个模态独立过 Encoder + Decoder (共享权重), 仅 Head 前融合
  - 推理时缓存 (z_feat_rgb, z_feat_tir) 两个 backbone 特征
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class RGBTSwinTrackDecFuse(nn.Module):
    """RGB-Thermal 双输入跟踪网络 (Decoder 后融合)。

    Backbone → Encoder → Decoder → 0.5 Fuse → Head
    """

    def __init__(self, backbone, encoder, decoder, out_norm, head,
                 z_backbone_out_stage, x_backbone_out_stage,
                 z_input_projection, x_input_projection,
                 z_pos_enc, x_pos_enc):
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
        self.reset_parameters()

    def reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        if self.z_input_projection is not None:
            self.z_input_projection.apply(_init_weights)
        if self.x_input_projection is not None:
            self.x_input_projection.apply(_init_weights)
        self.encoder.apply(_init_weights)
        self.decoder.apply(_init_weights)

    # ── Backbone ──────────────────────────────────────────────────────

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

    # ── Encoder + Decoder (per-modality, shared weights) ──────────────

    def _encode_decode(self, z_feat, x_feat):
        """单个模态过 Encoder → Decoder (共享权重)。"""
        z_pos = self.z_pos_enc().unsqueeze(0) if self.z_pos_enc is not None else None
        x_pos = self.x_pos_enc().unsqueeze(0) if self.x_pos_enc is not None else None
        enc_z, enc_x = self.encoder(z_feat, x_feat, z_pos, x_pos)
        dec_x = self.decoder(enc_z, enc_x, z_pos, x_pos)
        return dec_x

    # ── 推理接口 ──────────────────────────────────────────────────────

    def initialize(self, z_rgb, z_tir):
        """返回 (z_feat_rgb, z_feat_tir) 元组 — 缓存的 backbone 特征。"""
        return self._get_template_feat(z_rgb, z_tir)

    def track(self, cached, x_rgb, x_tir):
        """cached = (z_feat_rgb, z_feat_tir) — 缓存的 backbone 特征。

        Decoder 后融合: 每个模态独立过 Encoder(z,x) → Decoder, 然后 0.5 融合 → Head。
        """
        z_r, z_t = cached
        x_r, x_t = self._get_search_feat(x_rgb, x_tir)
        # 各模态独立过 Encoder → Decoder
        dec_x_r = self._encode_decode(z_r, x_r)
        dec_x_t = self._encode_decode(z_t, x_t)
        # ═══ Decoder 后融合 (Head 前) ═══
        x_fused = 0.5 * dec_x_r + 0.5 * dec_x_t
        x = self.out_norm(x_fused)
        return self.head(x)

    def forward(self, z_rgb, z_tir, x_rgb=None, x_tir=None, cached=None):
        if cached is None:
            cached = self.initialize(z_rgb, z_tir)
        if x_rgb is not None and x_tir is not None:
            return self.track(cached, x_rgb, x_tir)
        return cached
