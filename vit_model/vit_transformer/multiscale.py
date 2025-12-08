import torch
import torch.nn as nn
from vit_model.layers.frft_fdconv import FrFDConv
from vit_model.config import Config

# 模块级共享 alpha（当 alpha_shared=True 时复用同一参数）
_SHARED_ALPHA_MSC = None

class SpatioSpectralMultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatioSpectralMultiScale, self).__init__()

        # ---- 读取 FRFT 配置（无配置则用安全默认） ----
        frft_cfg     = getattr(Config.model_config, "frft", None)
        alpha_init   = 0.5  if frft_cfg is None else float(getattr(frft_cfg, "alpha_init", 0.5))
        use_polar    = True if frft_cfg is None else bool(getattr(frft_cfg, "frfdconv_use_polar", True))
        alpha_shared = False if frft_cfg is None else bool(getattr(frft_cfg, "alpha_shared", False))

        # ---- 共享或独立 alpha 参数 ----
        global _SHARED_ALPHA_MSC
        if alpha_shared:
            if _SHARED_ALPHA_MSC is None:
                _SHARED_ALPHA_MSC = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
            self.alpha = _SHARED_ALPHA_MSC
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # 1×1 先把通道统一到 64（用于多尺度分支）
        self.input_fix = FrFDConv(
            in_channels, 64, kernel_size=1, stride=1, bias=False,
            alpha=self.alpha, use_polar=use_polar
        )

        # 多尺度深度卷积（保持 64 通道的划分：21 + 21 + 22 = 64）
        self.conv3 = nn.Conv2d(64, 21, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(64, 21, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv2d(64, 22, kernel_size=7, padding=3, bias=False)

        # 注意：这里按你的原设计，使用 out_channels 做 BN/注意力与融合的通道数
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.spectral_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 1×1 融合（显式带上 α / use_polar）
        self.fusion_conv = FrFDConv(
            out_channels, out_channels, kernel_size=1, bias=False,
            alpha=self.alpha, use_polar=use_polar
        )

        # 输出统一到 64（供后续 Transformer 路径使用）
        self.output_fix = FrFDConv(
            out_channels, 64, kernel_size=1, stride=1, bias=False,
            alpha=self.alpha, use_polar=use_polar
        )

    def forward(self, x):
        # 先归一到 64
        x = self.input_fix(x)

        # 3/5/7 多尺度并联
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x  = torch.cat([x3, x5, x7], dim=1)  # (B, 64, H, W)

        # BN+ReLU（保持与原逻辑一致，BN 维度按 out_channels）
        x = self.bn(x)
        x = self.relu(x)

        # 谱注意力（SE 风格）
        x = x * self.spectral_attention(x)

        # 1×1 融合 + 输出到 64
        x = self.fusion_conv(x)
        return self.output_fix(x)
