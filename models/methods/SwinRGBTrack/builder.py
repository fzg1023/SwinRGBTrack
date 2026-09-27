"""
RGBTSwinTrack Builder — 构建 RGBT 双模态模型
==============================================
复用 SwinTrack 的所有子模块 (encoder, decoder, head, pos_enc)，
仅替换顶层网络为 RGBTSwinTrack (双模态 + 0.5 融合)。
"""
from core.run.event_dispatcher.register import EventRegister
from models.utils.drop_path import DropPathAllocator, DropPathScheduler
import torch.nn as nn

from models.backbone.builder import build_backbone
from models.head.builder import build_head
from models.methods.SwinTrack.modules.encoder.builder import build_encoder
from models.methods.SwinTrack.modules.decoder.builder import build_decoder
from models.methods.SwinTrack.positional_encoding.builder import build_position_embedding
from models.methods.SwinRGBTrack.network import RGBTSwinTrack


def build_rgbt_swin_track_main_components(config, num_epochs, iterations_per_epoch,
                                           event_register: EventRegister, has_training_run):
    """构建 RGBT model 的主要组件 (encoder, decoder, projections, pos_enc)。"""
    transformer_config = config['transformer']

    drop_path_config = transformer_config['drop_path']
    drop_path_allocator = DropPathAllocator(drop_path_config['rate'])

    backbone_dim = transformer_config['backbone']['dim']
    transformer_dim = transformer_config['dim']

    z_shape = transformer_config['backbone']['template']['shape']
    x_shape = transformer_config['backbone']['search']['shape']

    z_input_projection = None
    x_input_projection = None
    if backbone_dim != transformer_dim:
        z_input_projection = nn.Linear(backbone_dim, transformer_dim)
        x_input_projection = nn.Linear(backbone_dim, transformer_dim)

    num_heads = transformer_config['num_heads']
    mlp_ratio = transformer_config['mlp_ratio']
    qkv_bias = transformer_config['qkv_bias']
    drop_rate = transformer_config['drop_rate']
    attn_drop_rate = transformer_config['attn_drop_rate']

    position_embedding_config = transformer_config['position_embedding']
    z_pos_enc, x_pos_enc = build_position_embedding(
        position_embedding_config, z_shape, x_shape, transformer_dim)

    with drop_path_allocator:
        encoder = build_encoder(config, drop_path_allocator,
                                transformer_dim, num_heads, mlp_ratio,
                                qkv_bias, drop_rate, attn_drop_rate,
                                z_shape, x_shape)

        decoder = build_decoder(config, drop_path_allocator,
                                transformer_dim, num_heads, mlp_ratio,
                                qkv_bias, drop_rate, attn_drop_rate,
                                z_shape, x_shape)

    out_norm = nn.LayerNorm(transformer_dim)

    if 'warmup' in drop_path_config and len(drop_path_allocator) > 0:
        if has_training_run:
            from models.utils.build_warmup_scheduler import build_warmup_scheduler
            scheduler = build_warmup_scheduler(
                drop_path_config['warmup'], drop_path_config['rate'],
                iterations_per_epoch, num_epochs)
            drop_path_scheduler = DropPathScheduler(
                drop_path_allocator.get_all_allocated(), scheduler)
            event_register.register_iteration_end_hook(drop_path_scheduler)
            event_register.register_epoch_begin_hook(drop_path_scheduler)

    backbone_out_stage = transformer_config['backbone']['stage']

    return (encoder, decoder, out_norm,
            backbone_out_stage, backbone_out_stage,
            z_input_projection, x_input_projection,
            z_pos_enc, x_pos_enc)


def build_rgbt_swin_track(config, load_pretrained, num_epochs, iterations_per_epoch,
                           event_register: EventRegister, has_training_run):
    """构建完整的 RGBTSwinTrack 模型。

    Returns:
        RGBTSwinTrack: RGBT 双模态模型
    """
    backbone = build_backbone(config, load_pretrained)

    (encoder, decoder, out_norm,
     z_backbone_out_stage, x_backbone_out_stage,
     z_input_projection, x_input_projection,
     z_pos_enc, x_pos_enc) = build_rgbt_swin_track_main_components(
        config, num_epochs, iterations_per_epoch,
        event_register, has_training_run)

    head = build_head(config)

    return RGBTSwinTrack(
        backbone, encoder, decoder, out_norm, head,
        z_backbone_out_stage, x_backbone_out_stage,
        z_input_projection, x_input_projection,
        z_pos_enc, x_pos_enc)
