# vit_model/transformer/encoder.py

import torch.nn as nn
from .block import Block  # 注意：确保你在 block.py 中定义了 Block(config, vis, mode)

class Encoder(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm_rgb = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.encoder_norm_hsi = nn.LayerNorm(config.hidden_size, eps=1e-6)

        num_layers = config.transformer["num_layers"]
        for i in range(num_layers):
            if i < num_layers // 2:
                # 前半部分使用 self-attention（单模态注意力）
                layer = Block(config, vis=vis, mode="sa")
            else:
                # 后半部分使用 multi-branch attention（跨模态融合）
                layer = Block(config, vis=vis, mode="mba")
            self.layer.append(layer)

    def forward(self, rgb, hsi):
        attn_weights = []

        for block in self.layer:
            rgb, hsi, weights = block(rgb, hsi)
            if self.vis and weights is not None:
                attn_weights.append(weights)

        rgb = self.encoder_norm_rgb(rgb)
        hsi = self.encoder_norm_hsi(hsi)
        return rgb, hsi, attn_weights
