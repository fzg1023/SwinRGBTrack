from core.run.event_dispatcher.register import EventRegister


def build_model(config: dict, runtime_vars, max_batch_size, num_epochs: int, iterations_per_epoch: int, event_register: EventRegister, has_training_run: bool):
    load_pretrain = runtime_vars.resume is None and runtime_vars.weight_path is None
    if config['version'] == 1 and config['type'] == 'SwinTrack':
        from .SwinTrack.builder import build_swin_track
        return build_swin_track(config, load_pretrain, num_epochs, iterations_per_epoch, event_register, has_training_run)
    elif config['version'] == 1 and config['type'] == 'RGBTSwinTrack':
        from .SwinRGBTrack.builder import build_rgbt_swin_track
        return build_rgbt_swin_track(config, load_pretrain, num_epochs, iterations_per_epoch, event_register, has_training_run)
    elif config['version'] == 1 and config['type'] == 'RGBTSwinTrackEncFuse':
        from .SwinRGBTrack.builder_enc_fuse import build_rgbt_enc_fuse
        model = build_rgbt_enc_fuse(config, load_pretrain, num_epochs, iterations_per_epoch, event_register, has_training_run)
        from data.tracking.methods.SiamFC.pseudo_data import build_siamfc_pseudo_data_generator
        return model, build_siamfc_pseudo_data_generator(config, event_register)
    elif config['version'] == 1 and config['type'] == 'RGBTSwinTrackDecFuse':
        from .SwinRGBTrack.builder_dec_fuse import build_rgbt_dec_fuse
        model = build_rgbt_dec_fuse(config, load_pretrain, num_epochs, iterations_per_epoch, event_register, has_training_run)
        from data.tracking.methods.SiamFC.pseudo_data import build_siamfc_pseudo_data_generator
        return model, build_siamfc_pseudo_data_generator(config, event_register)
    elif config['version'] == 1 and config['type'] == 'RGBTSwinTrackConcatFuse':
        from .SwinRGBTrack.builder_concat_fuse import build_rgbt_concat_fuse
        model = build_rgbt_concat_fuse(config, load_pretrain, num_epochs, iterations_per_epoch, event_register, has_training_run)
        from data.tracking.methods.SiamFC.pseudo_data import build_siamfc_pseudo_data_generator
        return model, build_siamfc_pseudo_data_generator(config, event_register)
    else:
        raise NotImplementedError(f'Unknown version {config["version"]} with type {config.get("type", "Unknown")}')
