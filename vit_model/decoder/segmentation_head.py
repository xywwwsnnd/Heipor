# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SegmentationHead(nn.Module):
    """
    Segmentation head:
      Conv(k, same padding, bias off when BN) -> BN -> ReLU -> (optional Dropout)
      -> 1x1 Conv(out) -> (optional Upsample) -> (optional Activation)

    Args:
        in_channels (int)
        out_channels (int)
        kernel_size (int): default 3
        upsampling (int): scale factor for bilinear upsample, default 1 (no upsample)
        with_bn (bool): use BatchNorm2d after the first conv, default True
        bn_momentum (float): BN momentum, default 0.05 (稍微平滑些，batch=16 很合适)
        dropout (float): dropout rate after ReLU, default 0.0 (不需要就设 0)
        activation (str|None): 'sigmoid' | 'softmax' | None
            ⚠️ 训练用 BCEWithLogitsLoss 时，这里应为 None
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        upsampling: int = 1,
        with_bn: bool = True,
        bn_momentum: float = 0.05,
        dropout: float = 0.0,
        activation: Optional[str] = None,
    ):
        super().__init__()
        padding = kernel_size // 2  # odd k -> "same"
        layers = [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not with_bn,        # 有 BN 就关掉 bias
            ),
        ]
        if with_bn:
            layers.append(nn.BatchNorm2d(in_channels, momentum=bn_momentum))
        layers.append(nn.ReLU(inplace=True))
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True))

        self.block = nn.Sequential(*layers)
        self.upsampling = int(upsampling)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.upsampling > 1:
            x = F.interpolate(x, scale_factor=self.upsampling, mode="bilinear", align_corners=False)
        if self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'softmax':
            x = torch.softmax(x, dim=1)
        return x
