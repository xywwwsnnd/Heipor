# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from .decoder_block import DecoderBlock, Conv2dGELU


class DecoderCup(nn.Module):
    """
    UNet-style decoder (CA-free):
      - Input transformer tokens [B, N, hidden] with N = (H/16)*(W/16)
      - Project to [B, 64, H/16, W/16], then upsample ×2 per block
      - Each block optionally concatenates a skip feature
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        head_channels = 64
        self.conv_more = Conv2dGELU(
            in_channels=config.hidden_size,
            out_channels=head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

        decoder_channels: List[int] = list(config.decoder_channels)
        skip_channels_cfg: List[int] = list(config.skip_channels)
        assert len(skip_channels_cfg) == len(decoder_channels), \
            f"len(skip_channels)={len(skip_channels_cfg)} must equal len(decoder_channels)={len(decoder_channels)}"

        # Build blocks
        in_channels = [head_channels] + decoder_channels[:-1]
        blocks = []
        for in_ch, out_ch, sk_ch in zip(in_channels, decoder_channels, skip_channels_cfg):
            blocks.append(DecoderBlock(in_ch, out_ch, sk_ch))
        self.blocks = nn.ModuleList(blocks)
        self.n_skip = len(self.blocks)

    def forward(self, hidden_states: torch.Tensor, features: Optional[List[torch.Tensor]] = None):
        """
        hidden_states: [B, N, hidden], N=(H/16)*(W/16)
        features: expected order with 4 items: [top, s3, s2, s1]
                  with 3 items: [s3, s2, s1]
        Rule when extra is present:
          - If len(features) >= n_skip + 1, KEEP THE LAST n_skip (drop 'top' in 4->3 case).
        """
        B, n_patch, hidden = hidden_states.size()
        h = w = int(np.sqrt(n_patch))
        if h * w != n_patch:
            raise RuntimeError(f"[DecoderCup] Non-square token grid: N={n_patch} not a perfect square.")
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)

        # Normalize feature list length
        if features is not None:
            if len(features) >= self.n_skip + 1:
                features = features[-self.n_skip:]
            elif len(features) > self.n_skip:
                features = features[:self.n_skip]

        for i, block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < len(features)) else None
            x = block(x, skip=skip)

        return x
