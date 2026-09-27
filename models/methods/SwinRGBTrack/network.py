"""
RGBTSwinTrack — RGB+Thermal 双模态目标跟踪网络
================================================
架构:
  - 共享 Swin Transformer 骨干网络分别提取 RGB 和 TIR 特征
  - 0.5 加权融合 RGB 和 TIR 特征
  - 融合后的特征送入 encoder/decoder/head (与 SwinTrack 一致)
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class RGBTSwinTrack(nn.Module):
    """RGB-Thermal 双输入共享骨干目标跟踪网络。

    与 SwinTrack 的区别:
      - forward / initialize / track 接受双模态输入 (rgb, tir)
      - 骨干网络在不同模态间共享
      - 特征级 0.5 加权融合: feat = 0.5 * feat_rgb + 0.5 * feat_tir
      - Encoder, Decoder, Head 与原 SwinTrack 完全一致
    """

    def __init__(self, backbone, encoder, decoder, out_norm, head,
                 z_backbone_out_stage, x_backbone_out_stage,
                 z_input_projection, x_input_projection,
                 z_pos_enc, x_pos_enc):
        super(RGBTSwinTrack, self).__init__()
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
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.z_input_projection is not None:
            self.z_input_projection.apply(_init_weights)
        if self.x_input_projection is not None:
            self.x_input_projection.apply(_init_weights)

        self.encoder.apply(_init_weights)
        self.decoder.apply(_init_weights)

    # ── 双模态特征提取 + 0.5 融合 ──────────────────────────────────────

    def _get_template_feat(self, z_rgb, z_tir):
        """提取模板特征: 共享骨干 + 0.5 融合。

        Args:
            z_rgb (torch.Tensor): (B, 3, H_z, W_z) RGB 模板图像
            z_tir (torch.Tensor): (B, 3, H_z, W_z) TIR 模板图像
        Returns:
            torch.Tensor: (B, H_z*W_z, C) 融合后的模板特征
        """
        z_feat_rgb, = self.backbone(z_rgb, (self.z_backbone_out_stage,), False)
        z_feat_tir, = self.backbone(z_tir, (self.z_backbone_out_stage,), False)
        # 0.5 加权融合
        z_feat = 0.5 * z_feat_rgb + 0.5 * z_feat_tir
        if self.z_input_projection is not None:
            z_feat = self.z_input_projection(z_feat)
        return z_feat

    def _get_search_feat(self, x_rgb, x_tir):
        """提取搜索区域特征: 共享骨干 + 0.5 融合。

        Args:
            x_rgb (torch.Tensor): (B, 3, H_x, W_x) RGB 搜索图像
            x_tir (torch.Tensor): (B, 3, H_x, W_x) TIR 搜索图像
        Returns:
            torch.Tensor: (B, H_x*W_x, C) 融合后的搜索特征
        """
        x_feat_rgb, = self.backbone(x_rgb, (self.x_backbone_out_stage,), False)
        x_feat_tir, = self.backbone(x_tir, (self.x_backbone_out_stage,), False)
        # 0.5 加权融合
        x_feat = 0.5 * x_feat_rgb + 0.5 * x_feat_tir
        if self.x_input_projection is not None:
            x_feat = self.x_input_projection(x_feat)
        return x_feat

    # ── 推理接口 ───────────────────────────────────────────────────────

    def initialize(self, z_rgb, z_tir):
        """模板初始化 (推理阶段)。

        Args:
            z_rgb: (B, 3, H_z, W_z) RGB 模板
            z_tir: (B, 3, H_z, W_z) TIR 模板
        Returns:
            torch.Tensor: (B, L_z, C) 模板特征
        """
        return self._get_template_feat(z_rgb, z_tir)

    def track(self, z_feat, x_rgb, x_tir):
        """目标跟踪 (推理阶段)。

        Args:
            z_feat: (B, L_z, C) 缓存的模板特征
            x_rgb:  (B, 3, H_x, W_x) RGB 搜索区域
            x_tir:  (B, 3, H_x, W_x) TIR 搜索区域
        Returns:
            dict: {'class_score': ..., 'bbox': ...}
        """
        x_feat = self._get_search_feat(x_rgb, x_tir)
        return self._track(z_feat, x_feat)

    # ── 训练接口 ───────────────────────────────────────────────────────

    def forward(self, z_rgb, z_tir, x_rgb, x_tir, z_feat=None):
        """前向传播 (训练 + 推理)。

        Training:
            Input:  z_rgb, z_tir, x_rgb, x_tir (B, 3, H, W)
            Output: dict {'class_score': ..., 'bbox': ...}

        Inference - 初始化:
            Input:  z_rgb, z_tir, x_rgb=None, x_tir=None
            Output: torch.Tensor 模板特征

        Inference - 跟踪:
            Input:  z_rgb=None, z_tir=None, x_rgb, x_tir, z_feat=...
            Output: dict {'class_score': ..., 'bbox': ...}
        """
        if z_feat is None:
            z_feat = self.initialize(z_rgb, z_tir)
        if x_rgb is not None and x_tir is not None:
            return self.track(z_feat, x_rgb, x_tir)
        else:
            return z_feat

    # ── Transformer 后处理 (与原 SwinTrack 一致) ───────────────────────

    def _track(self, z_feat, x_feat):
        """融合特征的 Transformer 编码/解码 + Head 预测。

        Args:
            z_feat: (B, L_z, C) 模板特征
            x_feat: (B, L_x, C) 搜索特征
        Returns:
            dict: Head 输出
        """
        z_pos = None
        x_pos = None

        if self.z_pos_enc is not None:
            z_pos = self.z_pos_enc().unsqueeze(0)
        if self.x_pos_enc is not None:
            x_pos = self.x_pos_enc().unsqueeze(0)

        z_feat, x_feat = self.encoder(z_feat, x_feat, z_pos, x_pos)

        decoder_feat = self.decoder(z_feat, x_feat, z_pos, x_pos)
        decoder_feat = self.out_norm(decoder_feat)

        return self.head(decoder_feat)
