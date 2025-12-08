# vit_model/transformer/block.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .mlp import Mlp


def _window_partition(x4d: torch.Tensor, ws: int):
    """
    x4d: [B, H, W, C] -> [B*nw, ws*ws, C], meta for reverse
    """
    B, H, W, C = x4d.shape
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h > 0 or pad_w > 0:
        x4d = F.pad(x4d, (0, 0, 0, pad_w, 0, pad_h))  # pad on W then H
        H, W = H + pad_h, W + pad_w

    x = x4d.view(B, H // ws, ws, W // ws, ws, C)              # [B, H/ws, ws, W/ws, ws, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()              # [B, H/ws, W/ws, ws, ws, C]
    x = x.view(B * (H // ws) * (W // ws), ws * ws, C)         # [B*nw, Lw, C]
    meta = (H, W, pad_h, pad_w)
    return x, meta


def _window_reverse(xw: torch.Tensor, meta, ws: int, B: int):
    """
    xw: [B*nw, ws*ws, C] -> [B, H, W, C] (remove pad)
    """
    H, W, pad_h, pad_w = meta
    C = xw.shape[-1]
    x = xw.view(B, H // ws, W // ws, ws, ws, C)               # [B, H/ws, W/ws, ws, ws, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()              # [B, H/ws, ws, W/ws, ws, C]
    x = x.view(B, H, W, C)                                    # [B, H, W, C]
    if pad_h > 0 or pad_w > 0:
        x = x[:, :H - pad_h, :W - pad_w, :].contiguous()
    return x


class Block(nn.Module):
    """
    - mode == "sa": 单模态自注意力（这里套 W-MSA，只对 SA 生效）
    - mode == "mba": 跨模态融合注意力（保持你原来的 Attention 全局实现不变）
    """
    def __init__(self, config, vis: bool = False, mode: str = "sa"):
        super().__init__()
        self.mode = mode  # "sa" or "mba"
        self.vis = vis

        self.attn_norm_rgb = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn_norm_hsi = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm_rgb = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm_hsi = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # 注意力与 MLP
        self.attn = Attention(config, vis=vis, mode=mode)
        self.ffn_rgb = Mlp(config)
        self.ffn_hsi = Mlp(config)

        # 仅 SA 使用的窗口大小（MBA 始终全局）
        tr_cfg = getattr(config, "transformer", {}) if isinstance(getattr(config, "transformer", {}), dict) else {}
        self.ws = int(tr_cfg.get("window_size_sa", 8))

    def _forward_attn_sa_windowed(self, rgb: torch.Tensor, hsi: torch.Tensor):
        """
        只在 SA 路径启用窗口注意力：
          输入:  [B, N, C] 两个模态
          输出:  [B, N, C] 两个模态
        """
        B, N, C = rgb.shape
        S = int(math.sqrt(N))
        # 若不是完美平方，仍然尝试兜底（一般不会发生，因为你的 tokens 固定 16×16）
        if S * S != N:
            S = int(round(N ** 0.5))

        # 归一化后的 token -> 4D
        rgb4d = rgb.view(B, S, S, C)
        hsi4d = hsi.view(B, S, S, C)

        # 分窗口
        rgb_w, meta = _window_partition(rgb4d, self.ws)
        hsi_w, _    = _window_partition(hsi4d, self.ws)  # 同尺寸，这里 meta 一致

        # 在窗口内做“原来的 SA”（Attention 的 SA 分支无需改动）
        out_rgb_w, out_hsi_w, _ = self.attn(rgb_w, hsi_w)

        # 还原窗口 -> token
        out_rgb4d = _window_reverse(out_rgb_w, meta, self.ws, B)
        out_hsi4d = _window_reverse(out_hsi_w, meta, self.ws, B)
        out_rgb   = out_rgb4d.view(B, -1, C)
        out_hsi   = out_hsi4d.view(B, -1, C)
        return out_rgb, out_hsi, None  # 不聚合 window-wise 的权重，避免显存暴涨

    def forward(self, rgb: torch.Tensor, hsi: torch.Tensor):
        # -------- Attention --------
        rgb_residual = rgb
        hsi_residual = hsi

        rgb = self.attn_norm_rgb(rgb)
        hsi = self.attn_norm_hsi(hsi)

        if self.mode == "sa" and self.ws and self.ws > 0:
            # 只对 SA 开窗口
            rgb, hsi, weights = self._forward_attn_sa_windowed(rgb, hsi)
        else:
            # MBA 或者显式关闭窗口时，走全局（与你原 Attention 的行为一致）
            rgb, hsi, weights = self.attn(rgb, hsi)

        rgb = rgb + rgb_residual
        hsi = hsi + hsi_residual

        # -------- FFN --------
        rgb_residual = rgb
        hsi_residual = hsi

        rgb = self.ffn_norm_rgb(rgb)
        hsi = self.ffn_norm_hsi(hsi)

        rgb = self.ffn_rgb(rgb) + rgb_residual
        hsi = self.ffn_hsi(hsi) + hsi_residual

        return rgb, hsi, weights
