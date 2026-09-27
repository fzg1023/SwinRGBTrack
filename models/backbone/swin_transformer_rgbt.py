"""
RGBT Swin Transformer — 6ch input for RGBT tracking
DIF-Net style: 3ch RGB + 3ch TIR concatenated, pretrained weight 3→6ch
"""
import os
import torch
import torch.nn as nn


class SwinTransformerRGBT(nn.Module):
    """Wrapper: loads standard Swin-B with 6ch input, handles 3→6ch pretrained weights."""

    def __init__(self, name='swin_base_patch4_window12_384_in22k', load_pretrained=True,
                 pretrained_path=None):
        super().__init__()
        from models.backbone.swin_transformer import build_swin_transformer_backbone

        self.backbone = build_swin_transformer_backbone(name, load_pretrained=False,
                                                        output_layers=(2,), in_chans=6)

        if load_pretrained:
            if pretrained_path and os.path.exists(pretrained_path):
                self._load_local(pretrained_path)
            else:
                self._load_url(name)

    def _load_local(self, path):
        """Load from local .pth file, handling 3ch→6ch."""
        print(f"[RGBT Backbone] Loading local: {path}")
        ck = torch.load(path, map_location='cpu')
        state_dict = ck.get('model', ck)
        for key in list(state_dict.keys()):
            if ('proj.weight' in key or 'patch_embed.proj.weight' in key) and state_dict[key].shape[1] == 3:
                state_dict[key] = state_dict[key].repeat(1, 2, 1, 1) * 0.5
                break
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"[RGBT Backbone] Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def _load_url(self, name):
        from models.backbone.swin_transformer import _cfg
        if name not in _cfg:
            print(f"[WARN] No pretrained config for {name}")
            return
        url = _cfg[name].get('url', '')
        if not url:
            print(f"[WARN] No pretrained URL for {name}")
            return
        try:
            state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', progress=False)
            if 'model' in state_dict:
                state_dict = state_dict['model']

            # Handle 3ch→6ch: patch_embed.proj weight [out, 3, k, k] → [out, 6, k, k]
            for key in list(state_dict.keys()):
                if key.endswith('proj.weight') and state_dict[key].shape[1] == 3:
                    w3 = state_dict[key]
                    state_dict[key] = w3.repeat(1, 2, 1, 1) * 0.5
                    break

            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[RGBT Backbone] Pretrained loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            print(f"[WARN] Pretrained loading failed: {e}")

    def forward(self, x, output_layers, *args, **kwargs):
        return self.backbone.forward(x, output_layers, *args, **kwargs)
