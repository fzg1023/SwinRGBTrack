"""
RGBTSwinTrack-DecFuse-Adaptive Builder
"""
from core.run.event_dispatcher.register import EventRegister
from models.backbone.builder import build_backbone
from models.head.builder import build_head
from models.methods.SwinTrack.modules.encoder.builder import build_encoder
from models.methods.SwinTrack.modules.decoder.builder import build_decoder
from models.methods.SwinTrack.positional_encoding.builder import build_position_embedding
from models.methods.SwinRGBTrack.network_dec_fuse_adaptive import RGBTSwinTrackDecFuseAdaptive
from models.utils.drop_path import DropPathAllocator, DropPathScheduler
import torch.nn as nn


def build_rgbt_dec_fuse_adaptive_main_components(config, num_epochs, iterations_per_epoch,
                                                   event_register: EventRegister, has_training_run):
    transformer_config = config['transformer']
    drop_path_allocator = DropPathAllocator(transformer_config['drop_path']['rate'])
    dim = transformer_config['dim']; backbone_dim = transformer_config['backbone']['dim']
    z_shape = transformer_config['backbone']['template']['shape']
    x_shape = transformer_config['backbone']['search']['shape']
    z_proj = nn.Linear(backbone_dim, dim) if backbone_dim != dim else None
    x_proj = nn.Linear(backbone_dim, dim) if backbone_dim != dim else None
    nh = transformer_config['num_heads']; mr = transformer_config['mlp_ratio']
    qb = transformer_config['qkv_bias']; dr = transformer_config['drop_rate']
    adr = transformer_config['attn_drop_rate']
    pe_cfg = transformer_config['position_embedding']
    z_pos_enc, x_pos_enc = build_position_embedding(pe_cfg, z_shape, x_shape, dim)
    with drop_path_allocator:
        encoder = build_encoder(config, drop_path_allocator, dim, nh, mr, qb, dr, adr, z_shape, x_shape)
        decoder = build_decoder(config, drop_path_allocator, dim, nh, mr, qb, dr, adr, z_shape, x_shape)
    out_norm = nn.LayerNorm(dim)
    dp_cfg = transformer_config['drop_path']
    if 'warmup' in dp_cfg and has_training_run:
        from models.utils.build_warmup_scheduler import build_warmup_scheduler
        sched = build_warmup_scheduler(dp_cfg['warmup'], dp_cfg['rate'], iterations_per_epoch, num_epochs)
        dps = DropPathScheduler(drop_path_allocator.get_all_allocated(), sched)
        event_register.register_iteration_end_hook(dps); event_register.register_epoch_begin_hook(dps)
    stage = transformer_config['backbone']['stage']
    return encoder, decoder, out_norm, stage, stage, z_proj, x_proj, z_pos_enc, x_pos_enc, dim


def build_rgbt_dec_fuse_adaptive(config, load_pretrained, num_epochs, iterations_per_epoch,
                                   event_register, has_training_run):
    backbone = build_backbone(config, load_pretrained)
    encoder, decoder, out_norm, z_stage, x_stage, z_proj, x_proj, z_pe, x_pe, dim = \
        build_rgbt_dec_fuse_adaptive_main_components(config, num_epochs, iterations_per_epoch,
                                                       event_register, has_training_run)
    head = build_head(config)
    return RGBTSwinTrackDecFuseAdaptive(backbone, encoder, decoder, out_norm, head,
                                         z_stage, x_stage, z_proj, x_proj, z_pe, x_pe, dim)
