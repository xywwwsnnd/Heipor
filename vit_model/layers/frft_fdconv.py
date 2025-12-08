# frft_layers.py  (phase-aware v2 — 2025-07-23, autograd-safe)
from __future__ import annotations
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ #
# 0)  MMCV Registry（没有 mmcv 也能跑）
# ------------------------------------------------------------------ #
try:                                   # mmcv-full ≥2.0
    from mmcv.cnn import CONV_LAYERS
except Exception:                      # 无 mmcv 时兜底
    class _NoOpReg:
        def register_module(self, name=None):
            def _decorator(cls): return cls
            return _decorator
        # 便于 _safe 查询
        _module_dict = {}
    CONV_LAYERS = _NoOpReg()

# ------------------------------------------------------------------ #
# 1)  Differentiable 1-D  / 2-D  FRFT  (pure torch, supports autograd)
# ------------------------------------------------------------------ #
def _frft_1d(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Chirp-z shear algorithm — torch autograd friendly.
    x: (..., N) real/complex tensor
    a: scalar tensor (nn.Parameter), fractional order (continuous)
    """
    # 避免 sin(pi*a/2)=0 的奇点；可根据需要收紧到 (0,1)
    # 若希望限制在 (0,1)，可以把 a = a.sigmoid() 再线性映射。
    eps = torch.as_tensor(1e-4, device=x.device, dtype=(x.real if x.is_complex() else x).dtype)
    a = torch.clamp(a, eps, 2 - eps)

    N   = x.shape[-1]
    dev = x.device
    # 选择计算 dtype（与输入一致）
    rdt = x.real.dtype if x.is_complex() else x.dtype

    x = torch.fft.ifftshift(x, -1)

    pi     = torch.pi
    alpha  = a * pi / 2
    tan_a2 = torch.tan(a * pi / 4)
    sin_a  = torch.sin(alpha)

    n = torch.arange(-N // 2, N // 2, device=dev, dtype=rdt)

    c1 = torch.exp(-1j * pi * (n ** 2) * tan_a2 / N)          # chirp1
    c2 = torch.exp(-1j * pi * (n ** 2) / (N * sin_a))          # chirp2

    y  = torch.fft.ifft(torch.fft.fft(x * c1) * c2)
    y  = y * c1 / torch.sqrt(torch.abs(sin_a) + 1e-12)         # 幅度校正
    return torch.fft.fftshift(y, -1)


def frft2d(x: torch.Tensor, ax: torch.Tensor, ay: torch.Tensor) -> torch.Tensor:
    """Separable 2-D FRFT: width(−2) then height(−1), autograd-safe."""
    X = x if x.is_complex() else torch.complex(
        x.to(torch.float32), torch.zeros_like(x, dtype=torch.float32)
    )
    # width
    X = torch.stack([_frft_1d(r, ay) for r in X.movedim(-2, 0)], 0).movedim(0, -2)
    # height
    X = torch.stack([_frft_1d(c, ax) for c in X.movedim(-1, 0)], 0).movedim(0, -1)
    return X

# ------------------------------------------------------------------ #
# 2)  Layer-style FRFT  (cartesian / polar 输出)
# ------------------------------------------------------------------ #
class FrFT2DHSI(nn.Module):
    r"""
    Parameters
    ----------
    repr   :  `"polar"` →  返回 **幅值 |X|** 和 **相位 ∠X / π**  (2 × C)
              `"cartesian"` → 返回实/虚拼接 (2 × C)
    reduce :  若 `True` 且 `repr='polar'` ，自动用 1×1 Conv 把 2C → C
              （初始仅复制幅值通道，保证与旧模型对齐）
    alpha  :  分数阶参数；可传入 float 或 nn.Parameter（支持共享）
    """
    def __init__(self, channels: int, alpha: float | torch.Tensor = .5,
                 *, repr: str = 'polar', reduce: bool = True):
        super().__init__()
        if isinstance(alpha, nn.Parameter):
            self.alpha = alpha                           # 支持外部共享同一参数
        else:
            self.alpha = nn.Parameter(torch.as_tensor(float(alpha), dtype=torch.float32))

        self.repr   = repr.lower()
        self.reduce = reduce and (self.repr == 'polar')

        if self.reduce:                                  # 1×1 Conv 作幅相融合
            self.back = nn.Conv2d(2 * channels, channels, 1, bias=False)
            nn.init.constant_(self.back.weight, 0.)
            for c in range(channels):                    # 幅值直通
                self.back.weight.data[c, c, 0, 0] = 1.

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ax = ay = self.alpha                             # 保持 tensor，保梯度
        Xc = x if x.is_complex() else torch.complex(
            x.to(torch.float32), torch.zeros_like(x, dtype=torch.float32)
        )
        Y = frft2d(Xc, ax, ay)

        if self.repr == 'polar':
            out = torch.cat([torch.abs(Y), torch.angle(Y) / torch.pi], 1)   # (B,2C,H,W)
            if self.reduce:
                out = self.back(out)
            return out.type_as(x)

        # cartesian
        out = torch.view_as_real(Y)                                       # (...,2)
        out = out.movedim(-1, 1).reshape(x.size(0), -1, *x.shape[2:])     # (B,2C,H,W)
        return out.type_as(x)

# ------------------------------------------------------------------ #
# 3)  FRFT Dynamic Convolution (幅-相可学习融合)
# ------------------------------------------------------------------ #
@CONV_LAYERS.register_module()
class FrFDConv(nn.Module):
    """
    use_polar=True  ⇒  频域 (|W|, ∠W) 由可学习 `polar_w=(w_m,w_p)` 融合
    use_polar=False ⇒  退化为实/虚 Cartesian 并取实部 (等价旧版)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | Tuple[int, int],
                 stride: int | Tuple[int, int] = 1,
                 padding: int | Tuple[int, int] | None = None,
                 bias: bool = False,
                 groups: int = 1,
                 *,
                 alpha: float | torch.Tensor = .5,
                 dilation: int | Tuple[int, int] = 1,
                 use_polar: bool = True):
        super().__init__()

        k1, k2 = (kernel_size,)*2 if isinstance(kernel_size, int) else kernel_size
        self.stride   = (stride, stride) if isinstance(stride, int) else stride
        self.padding  = (k1 // 2, k2 // 2) if padding is None else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups   = groups
        self.use_polar = use_polar

        # learnable “频域” 权重
        self.spec = nn.Parameter(
            torch.randn(out_channels, in_channels, k1, k2) * 1e-3
        )

        # FrFT 模块（alpha 可为共享 nn.Parameter）
        self.frft = FrFT2DHSI(
            1, alpha,
            repr='polar' if use_polar else 'cartesian',
            reduce=False  # 保持 2-channel 输出以便后续融合
        )

        if use_polar:
            # 两个标量/输出通道：幅/相的线性融合权重
            self.polar_w = nn.Parameter(torch.tensor([1., 0.]).view(1, 2, 1, 1))

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    # ---------------- priv ---------------- #
    def _spec2kernel(self) -> torch.Tensor:
        S = self.spec.view(-1, 1, *self.spec.shape[2:])   # (N,1,k1,k2)
        K = self.frft(S)                                  # (N,2,k1,k2)

        if self.use_polar:                                # 幅-相融合
            K = (K * self.polar_w).sum(1)                 # (N,k1,k2)
        else:                                             # Cartesian → 取实部
            # K: (N,2,k1,k2)  =>  real = K[:,0], imag = K[:,1]
            K = K[:, 0, ...]                              # 仅取实部，等价旧版

        C_out, C_in, k1, k2 = self.spec.shape
        return K.view(C_out, C_in, k1, k2)

    # ---------------- fwd ----------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._spec2kernel()
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

# ------------------------------------------------------------------ #
# 4)  Kernel Spatial Modulation / FBM  (Polar + 自动 1×1)
# ------------------------------------------------------------------ #
class FrKSM(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int | None = None, alpha: float | torch.Tensor = .5):
        super().__init__()
        mid_ch = mid_ch or max(in_ch // 4, 16)

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.fc2  = nn.Conv2d(mid_ch, in_ch, 1, bias=True)
        self.act  = nn.SiLU()
        self.sig  = nn.Sigmoid()

        self.frft = FrFT2DHSI(in_ch, alpha, repr='polar', reduce=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.sig(self.fc2(self.act(self.fc1(self.gap(x)))))
        return x * g + self.frft(x)


class FrFBM(nn.Module):
    """Depth-wise 频带调制，Polar 表示。"""
    def __init__(self,
                 in_channels: int,
                 k_list: List[int] | None = None,
                 n_bands: int | None = None,
                 lowfreq_att: bool = False,
                 alpha: float | torch.Tensor = .5):
        super().__init__()

        if k_list is None:
            k_list = [2 ** (i + 1) for i in range(n_bands or 3)]

        self.k_list  = k_list
        self.low_att = lowfreq_att

        self.frft = FrFT2DHSI(in_channels, alpha, repr='polar', reduce=True)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, 3, 1, 1,
                       groups=in_channels, bias=False) for _ in k_list]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.frft(x)                               # (B,C,H,W)
        y = sum(conv(base / k) for k, conv in zip(self.k_list, self.convs))
        if self.low_att:
            y = y + x
        return x + y

# ------------------------------------------------------------------ #
# 5)  Safe aliases  (避免多次 import 时重复注册)
# ------------------------------------------------------------------ #
def _safe(name: str, cls):
    reg = getattr(CONV_LAYERS, '_module_dict', None)
    if reg is not None and name not in reg:
        CONV_LAYERS.register_module(name=name)(cls)

_safe('FrFDConv', FrFDConv)
_safe('FDConv',  FrFDConv)        # legacy alias

__all__ = ['FrFDConv', 'FrKSM', 'FrFBM', 'FrFT2DHSI']
