# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicBandSelector(nn.Module):
    """
    输入: HSI [B, C_hsi, H, W]
    输出: pseudo_rgb [B, 3, H, W]
    同时返回正则项:
      - peak: 选中波段的能量峰值(或Top-k均值)，值越大越好
      - div : 选择分布的熵 H(p) = -Σ p log p  (未归一化，单位 nats)
      - entropy_max: ln(K) 供上层做归一化; K=通道数= C_hsi
    """
    def __init__(self, in_ch: int, hidden: int = 128, temperature: float = 1.0, topk: int = 1):
        super().__init__()
        self.temperature = float(temperature)
        self.topk = int(max(1, topk))
        self.in_ch = int(in_ch)

        # 生成每个输出通道对应的 band 权重
        self.head_r = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(in_ch, hidden, 1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(hidden, in_ch, 1, bias=False))
        self.head_g = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(in_ch, hidden, 1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(hidden, in_ch, 1, bias=False))
        self.head_b = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(in_ch, hidden, 1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(hidden, in_ch, 1, bias=False))

        # 可选：初始化最后一层为零，开局接近均匀
        for m in (self.head_r[-1], self.head_g[-1], self.head_b[-1]):
            nn.init.zeros_(m.weight)

    def _softsel(self, x: torch.Tensor, head: nn.Module):
        # x: [B,C,H,W] -> 权重 logits: [B,C,1,1]
        logits = head(x)  # [B,C,1,1]
        w = F.softmax(logits.view(x.size(0), self.in_ch) / self.temperature, dim=1)  # [B,C]
        # 线性混合得到单通道
        y = torch.einsum('bc, bchw -> bhw', w, x)  # [B,H,W]
        return y, w  # w 是对 C 维度的概率分布

    def forward(self, x_hsi: torch.Tensor):
        """
        Returns:
          pseudo_rgb: [B,3,H,W]
          reg: dict{"peak": Tensor, "div": Tensor, "entropy_max": float}
        """
        assert x_hsi.dim() == 4, "HSI should be [B,C,H,W]"
        B, C, H, W = x_hsi.shape
        K = C  # 波段数

        r, wr = self._softsel(x_hsi, self.head_r)  # wr: [B,K]
        g, wg = self._softsel(x_hsi, self.head_g)
        b, wb = self._softsel(x_hsi, self.head_b)

        # 拼 pseudo-RGB
        y = torch.stack([r, g, b], dim=1)  # [B,3,H,W]

        # --------- 正则项 ---------
        # 1) peak：对三个头分别取 top-k 平均，再求平均
        # 这里的“能量”按权重分布的 topk 权重来衡量（也可以改成按响应能量）
        def topk_mean(w):
            # w: [B,K]
            v, _ = torch.topk(w, k=min(self.topk, K), dim=1)  # [B,k]
            return v.mean(dim=1)  # [B]
        peak_r = topk_mean(wr); peak_g = topk_mean(wg); peak_b = topk_mean(wb)
        peak = (peak_r + peak_g + peak_b) / 3.0                      # [B]

        # 2) 熵（未归一化）
        eps = 1e-8
        def entropy(w):
            return -(w * (w + eps).log()).sum(dim=1)  # [B], nats
        H_r = entropy(wr); H_g = entropy(wg); H_b = entropy(wb)
        H_mean = (H_r + H_g + H_b) / 3.0                                  # [B]
        H = H_mean.mean()                                                  # scalar

        peak = peak.mean()  # 用 batch 均值做全局指标（也方便对偶法的EMA）
        reg = {
            "peak": peak,              # 越大越好
            "div":  H,                 # 熵(未归一化)，越小越好
            "entropy_max": float(math.log(K)),  # 上界 ln(K)，供外层归一化
            "K": K,
        }
        return y, reg
