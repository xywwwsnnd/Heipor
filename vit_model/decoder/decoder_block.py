# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)


class DecoderBlock(nn.Module):
    """
    Plain UNet-style decoder block (no attention):
      - Upsample by 2
      - (Optional) concatenate skip if provided and skip_channels > 0
      - Two conv blocks
    """
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.skip_channels = int(skip_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = Conv2dGELU(in_channels + (self.skip_channels if self.skip_channels > 0 else 0),
                                out_channels, kernel_size=3, padding=1, use_batchnorm=True)
        self.conv2 = Conv2dGELU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        # upsample
        x = self.upsample(x)

        # optional skip concat
        if skip is not None and self.skip_channels > 0:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            # if channel mismatch, raise explicit error to surface config problems
            if skip.shape[1] != self.skip_channels:
                raise RuntimeError(f"[DecoderBlock] Skip channels mismatch: got {skip.shape[1]}, expected {self.skip_channels}")
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x
