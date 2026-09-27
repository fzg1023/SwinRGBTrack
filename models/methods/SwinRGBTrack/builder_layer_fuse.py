"""
RGBTSwinTrack-LayerFuse Builder — 层间融合
"""
from core.run.event_dispatcher.register import EventRegister
from models.backbone.builder import build_backbone
from models.head.builder import build_head
from models.methods.SwinTrack.modules.encoder.builder import build_encoder
from models.methods.SwinTrack.modules.decoder.builder import build_decoder
from models.methods.SwinTrack.positional_encoding.builder import build_position_embedding
from models.methods.SwinRGBTrack.network_layer_fuse import RGBTSwinTrackLayerFuse
from models.methods.SwinRGBTrack.bimodal_fuse import BimodalLinearFuse
from models.methods.SwinRGBTrack.cross_layer_agg import CrossLayerAggregation
from models.methods.SwinRGBTrack.layer_fuse_encoder import LayerFuseEncoder
from models.utils.drop_path import DropPathAllocator, DropPathScheduler
import torch.nn as nn


def build_rgbt_layer_fuse_main_components(config, num_epochs, iterations_per_epoch,
                                           event_register: EventRegister, has_training_run):
    """构建 LayerFuse 核心组件: LayerFuseEncoder + Decoder + Head。"""
    transformer_config = config['transformer']
    drop_path_allocator = DropPathAllocator(transformer_config['drop_path']['rate'])

    backbone_dim = transformer_config['backbone']['dim']
    transformer_dim = transformer_config['dim']
    z_shape = transformer_config['backbone']['template']['shape']
    x_shape = transformer_config['backbone']['search']['shape']

    z_proj = nn.Linear(backbone_dim, transformer_dim) if backbone_dim != transformer_dim else None
    x_proj = nn.Linear(backbone_dim, transformer_dim) if backbone_dim != transformer_dim else None

    num_heads = transformer_config['num_heads']
    mlp_ratio = transformer_config['mlp_ratio']
    qkv_bias = transformer_config['qkv_bias']
    drop_rate = transformer_config['drop_rate']
    attn_drop_rate = transformer_config['attn_drop_rate']

    # 位置编码
    pe_cfg = transformer_config['position_embedding']
    z_pos_enc, x_pos_enc = build_position_embedding(pe_cfg, z_shape, x_shape, transformer_dim)

    # ── 自注意力层 (8层, 共享权重用于 RGB/TIR 双流) ──
    num_encoders = transformer_config['encoder']['num_layers']
    traditional_pe = pe_cfg.get('enabled', False)

    # Untied position encoding (for relative position bias)
    untied_z_pos_enc = None
    untied_x_pos_enc = None
    rpe_index = None
    rpe_bias_table = None

    untied_pe_cfg = transformer_config.get('untied_position_embedding', {})
    if untied_pe_cfg.get('absolute', {}).get('enabled', False):
        from models.methods.SwinTrack.positional_encoding.untied.absolute import Untied2DPositionalEncoder
        untied_z_pos_enc = Untied2DPositionalEncoder(transformer_dim, num_heads, z_shape[0], z_shape[1])
        untied_x_pos_enc = Untied2DPositionalEncoder(transformer_dim, num_heads, x_shape[0], x_shape[1])

    if untied_pe_cfg.get('relative', {}).get('enabled', False):
        from models.methods.SwinTrack.positional_encoding.untied.relative import (
            RelativePosition2DEncoder,
            generate_2d_concatenated_self_attention_relative_positional_encoding_index
        )
        rpe_index = generate_2d_concatenated_self_attention_relative_positional_encoding_index(
            (z_shape[1], z_shape[0]), (x_shape[1], x_shape[0]))
        rpe_bias_table = RelativePosition2DEncoder(num_heads, rpe_index.max() + 1)

    # 构建自注意力层
    from models.methods.SwinTrack.modules.self_attention_block import SelfAttentionBlock
    sa_layers = []
    with drop_path_allocator:
        for i in range(num_encoders):
            sa_layers.append(
                SelfAttentionBlock(
                    transformer_dim, num_heads, mlp_ratio, qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=drop_path_allocator.allocate(),
                    attn_pos_encoding_only=not traditional_pe
                )
            )
            drop_path_allocator.increase_depth()

    # ── 跨模态融合模块 (每层一个, 不共享, 不使用 DropPath) ──
    cross_fuse_modules = [
        BimodalLinearFuse(transformer_dim)
        for _ in range(num_encoders)
    ]

    # ── 跨层聚合 ──
    z_len = z_shape[0] * z_shape[1]  # 12×12 = 144
    cross_layer_agg = CrossLayerAggregation(transformer_dim, z_len=z_len)

    # ── LayerFuseEncoder ──
    layer_fuse_encoder = LayerFuseEncoder(
        sa_layers, cross_fuse_modules, cross_layer_agg,
        untied_z_pos_enc, untied_x_pos_enc,
        rpe_bias_table, rpe_index
    )

    # ── Decoder ──
    with drop_path_allocator:
        decoder = build_decoder(config, drop_path_allocator,
                                transformer_dim, num_heads, mlp_ratio,
                                qkv_bias, drop_rate, attn_drop_rate,
                                z_shape, x_shape)
    out_norm = nn.LayerNorm(transformer_dim)

    # ── DropPath Scheduler ──
    dp_cfg = transformer_config['drop_path']
    if 'warmup' in dp_cfg and has_training_run:
        from models.utils.build_warmup_scheduler import build_warmup_scheduler
        sched = build_warmup_scheduler(dp_cfg['warmup'], dp_cfg['rate'],
                                        iterations_per_epoch, num_epochs)
        dps = DropPathScheduler(drop_path_allocator.get_all_allocated(), sched)
        event_register.register_iteration_end_hook(dps)
        event_register.register_epoch_begin_hook(dps)

    stage = transformer_config['backbone']['stage']
    return (layer_fuse_encoder, decoder, out_norm,
            stage, stage, z_proj, x_proj, z_pos_enc, x_pos_enc)


def build_rgbt_layer_fuse(config, load_pretrained, num_epochs, iterations_per_epoch,
                           event_register, has_training_run):
    """构建完整的 LayerFuse 模型。"""
    backbone = build_backbone(config, load_pretrained)
    (layer_fuse_encoder, decoder, out_norm,
     z_stage, x_stage, z_proj, x_proj, z_pe, x_pe) = \
        build_rgbt_layer_fuse_main_components(config, num_epochs, iterations_per_epoch,
                                               event_register, has_training_run)
    head = build_head(config)
    return RGBTSwinTrackLayerFuse(backbone, layer_fuse_encoder, decoder, out_norm, head,
                                   z_stage, x_stage, z_proj, x_proj, z_pe, x_pe)
