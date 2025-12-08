# vit_model/vit_transformer/batch_prototype.py
# Lightweight batch-prototype fusion (unsupervised)
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchPrototypeFusion(nn.Module):
    """
    无监督原型融合（轻量版）：
    - 从整个 batch 的特征里提一个全局原型 p（C 维，按 B,H,W 求均值）
    - 计算每个像素与原型的相似度（cosine 或 L2），得到一个 gate
    - 用 gate 对输入特征进行细微重标定（residual/gating）

    Args:
        channels:  输入通道数 C
        mode:      "cos"（默认）或 "l2"
        tau:       温度/锐化系数（越大门控越“硬”），建议 5~20
        mix:       残差融合强度，0~1（建议 0.2~0.5）
        affine:    是否用 1×1 卷积做一个可学习的仿射（提高灵活性）
    """
    def __init__(self, channels: int, mode: str = "cos", tau: float = 10.0, mix: float = 0.35, affine: bool = True):
        super().__init__()
        self.mode = mode.lower()
        self.tau = float(tau)
        self.mix = float(mix)
        self.affine = nn.Conv2d(channels, channels, 1, bias=True) if affine else nn.Identity()
        if isinstance(self.affine, nn.Conv2d):
            nn.init.zeros_(self.affine.weight)
            nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 全局原型（跨 B,H,W 求均值）
        proto = x.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)

        if self.mode == "l2":
            # 负的 L2 距离作为相似度
            sim = - (x - proto).pow(2).sum(dim=1, keepdim=False)  # (B, H, W)
            # 标准化
            sim = (sim - sim.mean(dim=(1, 2), keepdim=True)) / (sim.std(dim=(1, 2), keepdim=True) + 1e-6)
        else:
            # cosine 相似度
            xp = F.normalize(x, dim=1)
            pp = F.normalize(proto, dim=1)
            sim = (xp * pp).sum(dim=1, keepdim=False)  # (B, H, W)

        # 温度缩放 + sigmoid
        gate = torch.sigmoid(self.tau * sim)          # (B, H, W)
        gate = gate.unsqueeze(1)                      # (B, 1, H, W)

        # 轻量残差重标定
        y = x * (1.0 + self.mix * gate)

        # 可学习仿射（初始为恒等）
        y = self.affine(y)
        return y
