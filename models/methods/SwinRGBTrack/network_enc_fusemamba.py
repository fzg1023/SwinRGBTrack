"""
RGBTSwinTrack-EncFuseMamba — Encoder 后 Mamba 残差融合
=======================================================
在 EncFuse 基础上, 在 0.5 融合之上加 Mamba 空间上下文残差:

  base = 0.5·enc_z_r + 0.5·enc_z_t
  h = in_proj(concat(enc_z_r, enc_z_t))        # Linear(1024→512), 零初始化
  h = Mamba(LN(h)) 双向扫描 (正向 + 翻转)
  delta = out_proj(h)                           # Linear(512→512), 零初始化
  z_fused = base + delta

in_proj 随机初始化, out_proj 零初始化 (标准 ReZero) → delta=0 → 初始严格等价 EncFuse 0.5 融合。
⚠️ 历史版本 in+out 双零初始化是死链 (mamba_ssm 对零输入输出为 0), 已修复。
Mamba 选择性扫描提供空间邻域 + 长程上下文, 由 main loss 直接训练。
x 流同样处理 (共享同一套 Mamba 权重)。
"""
import torch
import core.amp_compat  # AMP 兼容层 (服务器旧版 torch 无 torch.amp.GradScaler)
import torch.nn as nn
from mamba_ssm import Mamba
from models.methods.SwinRGBTrack.network_enc_fuse import RGBTSwinTrackEncFuse


class RGBTSwinTrackEncFuseMamba(RGBTSwinTrackEncFuse):
    """EncFuse + Mamba 空间上下文残差融合。"""

    def __init__(self, backbone, encoder, decoder, out_norm, head,
                 z_backbone_out_stage, x_backbone_out_stage,
                 z_input_projection, x_input_projection,
                 z_pos_enc, x_pos_enc, dim, d_state=16, d_conv=4):
        super().__init__(backbone, encoder, decoder, out_norm, head,
                         z_backbone_out_stage, x_backbone_out_stage,
                         z_input_projection, x_input_projection,
                         z_pos_enc, x_pos_enc)

        self.in_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv)
        self.out_proj = nn.Linear(dim, dim)

        # 标准 ReZero: 仅 out_proj 零初始化 (in_proj 随机初始化)。
        # ⚠️ 不能 in+out 双零初始化: mamba_ssm 对零输入输出严格为 0,
        # 双零初始化会让整条分支梯度恒为 0 (历史版本 bug, 已修复)。
        nn.init.constant_(self.out_proj.weight, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def reinit_spatial(self):
        """修复历史 checkpoint (in_proj=0 死链): 重初始化 in_proj, out_proj 归零。

        热启动旧权重后调用, 输出仍严格 = 0.5 融合 (out_proj=0),
        但梯度链恢复, 空间 mamba 可以真正学习。
        """
        from timm.models.layers import trunc_normal_
        trunc_normal_(self.in_proj.weight, std=.02)
        nn.init.constant_(self.in_proj.bias, 0.)
        nn.init.constant_(self.out_proj.weight, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def revive_context(self):
        """复活死链 checkpoint 的 mamba 上下文通路 (仅重初始化 in_proj)。

        旧 checkpoint (in_proj=0, out_proj.weight=0, out_proj.bias=已学常数) 中:
        delta = out_proj.bias 是学到的逐通道常数偏移 (相对纯 0.5 融合的增益来源),
        但 in_proj=0 使 mamba 上下文恒为 0, 上下文相关梯度恒为 0。
        只随机化 in_proj 并保留 out_proj → delta 不变, 输出与 checkpoint 逐位一致,
        同时 h≠0 复活上下文梯度通路 (out_proj.weight 第一步起学习, in_proj/mamba 第二步起)。
        """
        from timm.models.layers import trunc_normal_
        trunc_normal_(self.in_proj.weight, std=.02)
        nn.init.constant_(self.in_proj.bias, 0.)

    def kill_context(self):
        """复刻历史死链机制: in_proj/out_proj.weight 全零, 仅 out_proj.bias 可学习。

        历史 AMP 检查点的增益来自死链下 out_proj.bias 学到的逐通道常数
        (bias 梯度不依赖输入), 而"激活复活"(revive_context) 已被 decft 实验
        证明会过拟合。此方法在新训练中显式复刻死链, 只学 bias 常数。
        """
        nn.init.constant_(self.in_proj.weight, 0.)
        nn.init.constant_(self.in_proj.bias, 0.)
        nn.init.constant_(self.out_proj.weight, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def _fuse(self, f_r, f_t):
        """0.5 融合基线 + Mamba 双向扫描残差。

        mamba 段固定 fp32 计算 (fp16 下反向梯度易溢出产生 NaN)。
        """
        base = 0.5 * f_r + 0.5 * f_t
        with torch.amp.autocast('cuda', enabled=False):
            h = self.in_proj(torch.cat([f_r.float(), f_t.float()], dim=-1))
            h = self.norm(h)
            h = self.mamba_fwd(h)
            h = h + torch.flip(self.mamba_bwd(torch.flip(h, dims=[1])), dims=[1])
            delta = self.out_proj(h)
        return base + delta

    def track(self, cached, x_rgb, x_tir):
        z_r, z_t = cached
        x_r, x_t = self._get_search_feat(x_rgb, x_tir)
        enc_z_r, enc_x_r = self._encode(z_r, x_r)
        enc_z_t, enc_x_t = self._encode(z_t, x_t)
        z_fused = self._fuse(enc_z_r, enc_z_t)
        x_fused = self._fuse(enc_x_r, enc_x_t)
        return self._decode(z_fused, x_fused)
