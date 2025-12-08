from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ComplexConv2d", "DualGroupConv2d"]


def _is_1x1(conv: nn.Conv2d) -> bool:
    k, s, d = conv.kernel_size, conv.stride, conv.dilation
    return k == (1, 1) and s == (1, 1) and d == (1, 1)


class ComplexConv2d(nn.Module):
    """
    复数卷积：将输入视作 (real, imag)，分别做实数卷积后再按复数规则合成。
    为保证 1×1 时空间不变化，遇到 1×1/stride=1/dilation=1 强制使用 padding=0 的 F.conv2d。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        dilation: int | tuple = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

    def _conv_no_pad(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, conv.weight, conv.bias, stride=conv.stride, padding=0,
                        dilation=conv.dilation, groups=conv.groups)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zr, zi = (z.real, z.imag) if z.is_complex() else (z, z.new_zeros(z.shape))
        if _is_1x1(self.real) and _is_1x1(self.imag):
            r_zr = self._conv_no_pad(self.real, zr)
            i_zi = self._conv_no_pad(self.imag, zi)
            r_zi = self._conv_no_pad(self.real, zi)
            i_zr = self._conv_no_pad(self.imag, zr)
        else:
            r_zr = self.real(zr); i_zi = self.imag(zi)
            r_zi = self.real(zi); i_zr = self.imag(zr)
        yr = r_zr - i_zi
        yi = r_zi + i_zr
        return torch.complex(yr, yi)


class DualGroupConv2d(nn.Conv2d):
    """
    Dual（实/虚分组）卷积：groups=2，把 real/imag 两组通道各自卷积、互不混合。
    为保证 1×1 时空间不变化，遇到 1×1/stride=1/dilation=1 强制使用 padding=0 的 F.conv2d。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        dilation: int | tuple = 1,
        bias: bool = False,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == (1, 1) and self.stride == (1, 1) and self.dilation == (1, 1):
            return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=0,
                            dilation=self.dilation, groups=self.groups)
        return super().forward(x)
