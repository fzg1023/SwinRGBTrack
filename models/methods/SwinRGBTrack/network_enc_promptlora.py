"""
RGBTSwinTrack-EncPromptLoRA — Encoder 内逐层跨模态 Prompt 注入 + TIR 专属 LoRA
=============================================================================
背景: EncFuse/EncFuseMamba 的 Encoder 对 RGB / TIR 两个模态完全独立跑 8 层
(权重共享但零交互), 只有最后 0.5 融合 (+ mamba 残差) 才第一次相遇。之前在
"融合模块"上加空间/时序 mamba (数百万新增参数) 均因 978 序列小数据过拟合而失败。

本方案改为在 Encoder 内部做早融合, 且新增参数量小一个数量级 (~450K):
  对每层 i (共 8 层):
    r_in = stream_r + PromptGen_T2R_i(stream_t)   # T→R 低秩加性提示, 零初始化
    t_in = stream_t + PromptGen_R2T_i(stream_r)   # R→T 低秩加性提示, 零初始化
    stream_r = frozen_layer_i(r_in)                # R 流: 原始 frozen 权重
    stream_t = frozen_layer_i(t_in)                # T 流: frozen 权重 + LoRA 旁路 (仅 T 流激活)

PromptGen.up 与 LoRALinear.lora_B 均零初始化 → 初始状态下 encoder 行为与原始
EncFuseMamba 严格一致, 可直接热启动 ep1 checkpoint (strict=False)。之后沿用
原有 0.5 融合 + 已训练的 mamba 残差 (network_enc_fusemamba._fuse) → decoder → head。
"""
import torch
import torch.nn as nn
from models.methods.SwinRGBTrack.network_enc_fusemamba import RGBTSwinTrackEncFuseMamba
from models.methods.SwinTrack.modules.lora import LoRALinear


class PromptGen(nn.Module):
    """跨模态低秩加性 prompt: 从对方模态流生成扰动, 零初始化保证初始 prompt=0。"""

    def __init__(self, dim, r=16):
        super().__init__()
        self.down = nn.Linear(dim, r)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(r, dim)
        nn.init.constant_(self.up.weight, 0.)
        nn.init.constant_(self.up.bias, 0.)

    def forward(self, other_stream):
        return self.up(self.act(self.down(other_stream)))


class RGBTSwinTrackEncPromptLoRA(RGBTSwinTrackEncFuseMamba):
    """EncFuseMamba + Encoder 内逐层跨模态 Prompt 注入 + TIR 专属 LoRA。"""

    def __init__(self, backbone, encoder, decoder, out_norm, head,
                 z_backbone_out_stage, x_backbone_out_stage,
                 z_input_projection, x_input_projection,
                 z_pos_enc, x_pos_enc, dim, d_state=16, d_conv=4,
                 prompt_r=16, lora_r=8, lora_alpha=16):
        super().__init__(backbone, encoder, decoder, out_norm, head,
                         z_backbone_out_stage, x_backbone_out_stage,
                         z_input_projection, x_input_projection,
                         z_pos_enc, x_pos_enc, dim, d_state, d_conv)

        num_layers = len(self.encoder.layers)
        self.prompt_t2r = nn.ModuleList([PromptGen(dim, prompt_r) for _ in range(num_layers)])
        self.prompt_r2t = nn.ModuleList([PromptGen(dim, prompt_r) for _ in range(num_layers)])

        # 原始 encoder 8 层权重全部冻结, 只在 T 流上挂 LoRA 旁路 (R 流走纯 frozen 权重)。
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 普通 list (非 nn.ModuleList): 这些 LoRALinear 已挂在 encoder.layers.*.attn 下,
        # 若再用 ModuleList 收集会被 state_dict()/parameters() 二次遍历, 产生重复 key
        # (同一份 frozen qkv 权重在 checkpoint 里存两份)。这里仅作为 active 开关的引用表。
        self._lora_modules = []
        for layer in self.encoder.layers:
            attn = layer.attn
            if attn.attn_pos_encoding_only:
                attn.qkv = LoRALinear(attn.qkv, r=lora_r, alpha=lora_alpha)
                self._lora_modules.append(attn.qkv)
            else:
                attn.q = LoRALinear(attn.q, r=lora_r, alpha=lora_alpha)
                attn.k = LoRALinear(attn.k, r=lora_r, alpha=lora_alpha)
                attn.v = LoRALinear(attn.v, r=lora_r, alpha=lora_alpha)
                self._lora_modules.append(attn.q)
                self._lora_modules.append(attn.k)
                self._lora_modules.append(attn.v)

    def _set_lora_active(self, active):
        for m in self._lora_modules:
            m.active = active

    def _joint_encode(self, z_r, x_r, z_t, x_t):
        """双流联合逐层前向: 层间跨模态 prompt 注入 + 对应模态 LoRA 开关。"""
        enc = self.encoder
        z_pos = self.z_pos_enc().unsqueeze(0) if self.z_pos_enc is not None else None
        x_pos = self.x_pos_enc().unsqueeze(0) if self.x_pos_enc is not None else None

        # attn_pos_enc / concatenated_pos_enc 只依赖位置编码表, 与流内容无关,
        # 两个模态共用同一 encoder 对象, 计算一次即可复用 (逻辑与 ConcatenatedFusion.forward 对齐)。
        attn_pos_enc = None
        if enc.z_untied_pos_enc is not None:
            z_q_pos, z_k_pos = enc.z_untied_pos_enc()
            x_q_pos, x_k_pos = enc.x_untied_pos_enc()
            attn_pos_enc = (torch.cat((z_q_pos, x_q_pos), dim=1) @
                            torch.cat((z_k_pos, x_k_pos), dim=1).transpose(-2, -1)).unsqueeze(0)
        if enc.rpe_bias_table is not None:
            rpe = enc.rpe_bias_table(enc.rpe_index)
            attn_pos_enc = rpe if attn_pos_enc is None else attn_pos_enc + rpe

        concatenated_pos_enc = None
        if z_pos is not None:
            concatenated_pos_enc = torch.cat((z_pos, x_pos), dim=1)

        n_z = z_r.shape[1]
        stream_r = torch.cat((z_r, x_r), dim=1)
        stream_t = torch.cat((z_t, x_t), dim=1)

        for i, layer in enumerate(enc.layers):
            r_in = stream_r + self.prompt_t2r[i](stream_t)
            t_in = stream_t + self.prompt_r2t[i](stream_r)

            self._set_lora_active(False)
            stream_r = layer(r_in, concatenated_pos_enc, concatenated_pos_enc, attn_pos_enc)
            self._set_lora_active(True)
            stream_t = layer(t_in, concatenated_pos_enc, concatenated_pos_enc, attn_pos_enc)
        self._set_lora_active(False)

        return stream_r[:, :n_z], stream_r[:, n_z:], stream_t[:, :n_z], stream_t[:, n_z:]

    def track(self, cached, x_rgb, x_tir):
        z_r, z_t = cached
        x_r, x_t = self._get_search_feat(x_rgb, x_tir)
        enc_z_r, enc_x_r, enc_z_t, enc_x_t = self._joint_encode(z_r, x_r, z_t, x_t)
        z_fused = self._fuse(enc_z_r, enc_z_t)
        x_fused = self._fuse(enc_x_r, enc_x_t)
        return self._decode(z_fused, x_fused)
