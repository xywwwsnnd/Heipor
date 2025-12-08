# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class FRFT(nn.Module):
    def __init__(self, in_channels, order=0.5,
                 alpha_min=0.1, alpha_max=0.9,
                 gate_hidden_ratio=4):
        """
        in_channels: 输入通道数 C
        order:       初始分数阶 α_init（物理意义），例如 0.5
        alpha_min/alpha_max: 中路分数阶 α 的取值范围，默认 (0.1, 0.9)
        """
        super(FRFT, self).__init__()

        # 通道三分：上支路 C0，中支路 C1，下支路 C0
        C0 = int(in_channels / 3)
        C1 = int(in_channels) - 2 * C0
        self.C0 = C0
        self.C1 = C1

        # 三条支路的卷积
        self.conv_0 = nn.Conv2d(C0, C0, kernel_size=3, padding=1)           # 空域 3×3 分支
        self.conv_05 = nn.Conv2d(2 * C1, 2 * C1, kernel_size=1, padding=0)  # 分数域 1×1 分支
        self.conv_1 = nn.Conv2d(2 * C0, 2 * C0, kernel_size=1, padding=0)   # FFT 1×1 分支
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 拼接后的融合

        # =========================
        # 1) 中支路：动态 α + 分数域 1×1 动态门控（per-sample）
        # =========================
        hidden_freq = max(8, C1 // gate_hidden_ratio)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        # α 分支: C1 -> hidden -> 1 -> sigmoid
        # 物理 α_b = alpha_min + (alpha_max - alpha_min) * sigmoid(·)_b   （per-sample）
        self.fc_alpha = nn.Sequential(
            nn.Linear(C1, hidden_freq),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_freq, 1),
            nn.Sigmoid(),  # 输出 alpha_raw ∈ (0,1)
        )

        # 分数域 1×1 的 gate: C1 -> hidden -> (C_out + C_in)
        num_1x1 = 2 * C1
        self.fc_gate_freq = nn.Sequential(
            nn.Linear(C1, hidden_freq),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_freq, num_1x1 + num_1x1),  # C_out + C_in
        )

        # 初始化 α 分支，使初始“物理 α” ≈ 传进来的 order
        with torch.no_grad():
            self.fc_alpha[-2].weight.zero_()

            eps = 1e-4
            span = max(self.alpha_max - self.alpha_min, eps)
            order_f = float(order)

            # (order - alpha_min) / span → [0,1] 再裁剪
            raw_target = (order_f - self.alpha_min) / span
            raw_target = max(eps, min(1.0 - eps, raw_target))

            logit = math.log(raw_target / (1.0 - raw_target))
            self.fc_alpha[-2].bias.fill_(logit)

        # =========================
        # 2) 上支路：空域 3×3 per-sample 动态门控
        # =========================
        hidden_spatial = max(8, C0 // gate_hidden_ratio)
        num_3x3 = C0   # conv_0: C0 -> C0

        self.fc_gate_spatial = nn.Sequential(
            nn.Linear(C0, hidden_spatial),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_spatial, num_3x3 + num_3x3),  # C_out + C_in（这里都是 C0）
        )

        # =========================
        # 3) 下支路：FFT 1×1 per-sample 动态门控
        # =========================
        # conv_1 的 in/out 通道都是 2*C0
        hidden_fft = max(8, C0 // gate_hidden_ratio)
        num_fft = 2 * C0

        self.fc_gate_fft = nn.Sequential(
            nn.Linear(C0, hidden_fft),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_fft, num_fft + num_fft),  # C_out + C_in（这里都是 2*C0）
        )

    # =============== 下面是 dfrtmtrx_batch / dis_s / cconvm（带 device） ===============

    def cconvm(self, N, s, device):
        M = torch.zeros((N, N), device=device, dtype=s.dtype)
        dum = s
        for i in range(N):
            M[:, i] = dum
            dum = torch.roll(dum, 1)
        return M

    def dis_s(self, N, app_ord, device):
        app_ord = int(app_ord / 2)
        s = torch.cat(
            (
                torch.tensor([0.0, 1.0], device=device),
                torch.zeros(N - 1 - 2 * app_ord, device=device),
                torch.tensor([1.0], device=device),
            )
        )
        S = self.cconvm(N, s, device=device) + torch.diag(torch.fft.fft(s).real)

        p = N
        r = math.floor(N / 2)
        P = torch.zeros((p, p), device=device)
        P[0, 0] = 1.0
        even = 1 - (p % 2)

        for i in range(1, r - even + 1):
            P[i, i] = 1 / math.sqrt(2)
            P[i, p - i] = 1 / math.sqrt(2)

        if even:
            P[r, r] = 1.0

        for i in range(r + 1, p):
            P[i, i] = -1 / math.sqrt(2)
            P[i, p - i] = 1 / math.sqrt(2)

        CS = torch.einsum("ij,jk,ni->nk", S, P.T, P)
        C2 = CS[0: math.floor(N / 2 + 1), 0: math.floor(N / 2 + 1)]
        S2 = CS[math.floor(N / 2 + 1): N, math.floor(N / 2 + 1): N]
        ec, vc = torch.linalg.eig(C2)
        es, vs = torch.linalg.eig(S2)
        ec = ec.real
        vc = vc.real
        es = es.real
        vs = vs.real
        qvc = torch.vstack(
            (
                vc,
                torch.zeros(
                    (math.ceil(N / 2 - 1), math.floor(N / 2 + 1)), device=device
                ),
            )
        )
        SC2 = P @ qvc
        qvs = torch.vstack(
            (
                torch.zeros(
                    (math.floor(N / 2 + 1), math.ceil(N / 2 - 1)), device=device
                ),
                vs,
            )
        )
        SS2 = P @ qvs
        idx = torch.argsort(-ec)
        SC2 = SC2[:, idx]
        idx = torch.argsort(-es)
        SS2 = SS2[:, idx]

        if N % 2 == 0:
            S2C2 = torch.zeros((N, N + 1), device=device)
            SS2 = torch.hstack([SS2, torch.zeros((SS2.shape[0], 1), device=device)])
            S2C2[:, range(0, N + 1, 2)] = SC2
            S2C2[:, range(1, N, 2)] = SS2
            S2C2 = S2C2[:, torch.arange(S2C2.size(1)) != N - 1]
        else:
            S2C2 = torch.zeros((N, N), device=device)
            S2C2[:, range(0, N + 1, 2)] = SC2
            S2C2[:, range(1, N, 2)] = SS2
        return S2C2

    def dfrtmtrx_batch(self, N, a_vec, device):
        """
        批量生成一组 FrFT 矩阵 F(a_b)，每个样本一个分数阶 a_b。
        a_vec: [B]，每个样本一个 α
        返回:  [B, N, N]
        """
        app_ord = 2
        Evec = self.dis_s(N, app_ord, device=device)          # [N, N], real
        Evec = Evec.to(dtype=torch.complex64)                 # [N, N], complex

        even = 1 - (N % 2)
        l = torch.cat(
            (
                torch.arange(0, N - 1, device=device, dtype=torch.float32),
                torch.tensor([N - 1 + even], device=device, dtype=torch.float32),
            )
        )                                                      # [N]

        # a_vec: [B] -> [B,1]
        a_vec = a_vec.view(-1, 1).to(device=device, dtype=torch.float32)   # [B,1]
        l_row = l.view(1, -1)                                              # [1,N]

        # phase[b,k] = exp(-j*pi/2 * a_b * l_k)
        phase = torch.exp(-1j * torch.pi / 2 * a_vec * l_row)             # [B,N]

        # diag(phase_b) @ Evec  ≡  phase[b,:,None] * Evec[None,:,:]
        B = phase.unsqueeze(-1) * Evec.unsqueeze(0)                        # [B,N,N]

        # F_b = sqrt(N) * Evec^T @ B_b
        F = (N ** 0.5) * torch.einsum("ij,bjk->bik", Evec.T, B)           # [B,N,N]
        return F

    def FRFT2D(self, matrix, orders):
        """
        matrix: [N, C, H, W]
        orders: [N]，per-sample α
        """
        N, C, H, W = matrix.shape
        device = matrix.device

        if not torch.is_tensor(orders):
            orders = torch.tensor(orders, device=device, dtype=torch.float32)
        orders = orders.view(N).to(device=device, dtype=torch.float32)     # [N]

        # 对 H 和 W 分别批量生成 [N,H,H] 和 [N,W,W] 的 FrFT 核
        Fh = self.dfrtmtrx_batch(H, orders, device=device)                 # [N,H,H]
        Fw = self.dfrtmtrx_batch(W, orders, device=device)                 # [N,W,W]

        Fh = Fh.to(torch.complex64)
        Fw = Fw.to(torch.complex64)

        # 扩展到 [N, C, H, H] / [N, C, W, W]
        h_test = Fh.unsqueeze(1).expand(N, C, H, H)                        # [N,C,H,H]
        w_test = Fw.unsqueeze(1).expand(N, C, W, W)                        # [N,C,W,W]

        matrix_c = torch.fft.fftshift(matrix, dim=(2, 3)).to(dtype=torch.complex64)

        # (*,H,H) x (*,H,W) → (*,H,W)，* = (N,C)
        out = torch.matmul(h_test, matrix_c)                               # [N,C,H,W]
        out = torch.matmul(out, w_test)                                    # [N,C,H,W]
        out = torch.fft.fftshift(out, dim=(2, 3))
        return out

    def IFRFT2D(self, matrix, orders):
        """
        逆 FrFT：对每个样本使用 -α_b
        """
        return self.FRFT2D(matrix, -orders)

    # ==============================
    # forward: 三支路 + 动态门控
    # ==============================
    def forward(self, x):
        """
        x: [N, C, H, W]
        """
        N, C, H, W = x.shape
        C0 = self.C0
        C1 = self.C1

        # 通道三分
        x_0  = x[:, 0:C0, :, :]         # 空域分支
        x_05 = x[:, C0:C - C0, :, :]    # FrFT 分支
        x_1  = x[:, C - C0:C, :, :]     # FFT 分支

        # ============================================================
        # 1) 上支路: 空域 3×3 per-sample 动态卷积（avg pooling + group conv）
        # ============================================================
        # GAP(x_0) → 直接对 H,W 求均值
        gap0 = x_0.mean(dim=(2, 3), keepdim=True)        # [N, C0, 1, 1]
        gap0_vec = gap0.view(N, C0)                     # [N, C0]

        gate0 = self.fc_gate_spatial(gap0_vec)          # [N, 2*C0]
        gate0 = gate0.view(N, 2 * C0)
        g0_out, g0_in = torch.split(gate0, C0, dim=1)   # 各 [N, C0]

        # Y0[b,i,j] = sigmoid( g0_out[b,i] * g0_in[b,j] )
        Y0 = torch.sigmoid(
            g0_out.unsqueeze(2) * g0_in.unsqueeze(1)
        )                                               # [N, C0, C0]
        Y0 = Y0.unsqueeze(-1).unsqueeze(-1)             # [N, C0, C0, 1, 1]

        w0_base = self.conv_0.weight                    # [C0, C0, 3, 3]
        b0_base = self.conv_0.bias                      # [C0] or None

        w0_eff = w0_base.unsqueeze(0) * Y0              # [N, C0, C0, 3, 3]

        w0_eff_group = w0_eff.reshape(N * C0, C0, 3, 3)        # [N*C0, C0, 3, 3]
        x0_group = x_0.reshape(1, N * C0, H, W)                # [1, N*C0, H, W]

        if b0_base is not None:
            b0_eff_group = b0_base.unsqueeze(0).expand(N, -1).reshape(-1)  # [N*C0]
        else:
            b0_eff_group = None

        x0_out = F.conv2d(
            x0_group,
            w0_eff_group,
            b0_eff_group,
            stride=1,
            padding=1,
            groups=N          # 每个 sample 一组
        )
        x_0 = x0_out.view(N, C0, H, W)                  # [N, C0, H, W]

        # ============================================================
        # 2) 中支路: 分数域 FrFT + per-sample 动态 α + per-sample 动态 1×1
        # ============================================================
        # 2.1 per-sample α：对每个样本、每个通道做 avg pooling
        gap_feat_all = x_05.mean(dim=(2, 3))            # [N, C1]
        alpha_raw = self.fc_alpha(gap_feat_all).view(N) # [N] in (0,1)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_raw  # [N]

        # 2.2 per-sample gate for 1×1
        num_1x1 = 2 * C1
        gap_vec = gap_feat_all                          # [N, C1]
        gate_vec = self.fc_gate_freq(gap_vec)           # [N, 4*C1]
        gate_vec = gate_vec.view(N, 2 * num_1x1)        # [N, 4*C1]

        g_out, g_in = torch.split(gate_vec, num_1x1, dim=1)  # 各 [N, 2*C1]

        Y = torch.sigmoid(
            g_out.unsqueeze(2) * g_in.unsqueeze(1)
        )                                                    # [N, 2*C1, 2*C1]
        Y = Y.unsqueeze(-1).unsqueeze(-1)                    # [N, 2*C1, 2*C1, 1, 1]

        w_base = self.conv_05.weight                        # [2*C1, 2*C1, 1, 1]
        b_base = self.conv_05.bias                          # [2*C1] or None

        w_eff = w_base.unsqueeze(0) * Y                     # [N, 2*C1, 2*C1, 1, 1]
        w_eff_group = w_eff.reshape(N * 2 * C1, 2 * C1, 1, 1)   # [N*2C1, 2C1, 1, 1]

        if b_base is not None:
            b_eff_group = b_base.unsqueeze(0).expand(N, -1).reshape(-1)    # [N*2C1]
        else:
            b_eff_group = None

        # 2.3 FrFT + per-sample 动态 1×1 + IFrFT
        Fre = self.FRFT2D(x_05, alpha)                      # [N, C1, H, W] (complex)
        Real = Fre.real
        Imag = Fre.imag
        Mix = torch.cat((Real, Imag), dim=1)                # [N, 2*C1, H, W]

        Mix_group = Mix.reshape(1, N * 2 * C1, H, W)        # [1, N*2C1, H, W]
        Mix_out = F.conv2d(
            Mix_group,
            w_eff_group,
            b_eff_group,
            stride=1,
            padding=0,
            groups=N                                       # per-sample 动态 1×1
        )
        Mix = Mix_out.view(N, 2 * C1, H, W)

        Real1, Imag1 = torch.chunk(Mix, 2, dim=1)
        Fre_out = torch.complex(Real1, Imag1)
        IFRFT = self.IFRFT2D(Fre_out, alpha)                # IFrFT 使用同一个 α_b
        IFRFT = torch.abs(IFRFT) / (H * W)

        # ============================================================
        # 3) 下支路: FFT 1×1 per-sample 动态门控 + IFFT
        # ============================================================
        gap1_all = x_1.mean(dim=(2, 3), keepdim=True)       # [N, C0, 1, 1]
        gap1_vec = gap1_all.view(N, C0)                     # [N, C0]

        num_fft = 2 * C0
        gate1_vec = self.fc_gate_fft(gap1_vec)              # [N, 4*C0]
        gate1_vec = gate1_vec.view(N, 2 * num_fft)          # [N, 4*C0]

        g1_out, g1_in = torch.split(gate1_vec, num_fft, dim=1)  # 各 [N, 2*C0]

        Y1 = torch.sigmoid(
            g1_out.unsqueeze(2) * g1_in.unsqueeze(1)
        )                                                    # [N, 2*C0, 2*C0]
        Y1 = Y1.unsqueeze(-1).unsqueeze(-1)                  # [N, 2*C0, 2*C0, 1, 1]

        w1_base = self.conv_1.weight                        # [2*C0, 2*C0, 1, 1]
        b1_base = self.conv_1.bias                          # [2*C0] or None

        w1_eff = w1_base.unsqueeze(0) * Y1                  # [N, 2*C0, 2*C0, 1, 1]
        w1_eff_group = w1_eff.reshape(N * 2 * C0, 2 * C0, 1, 1)  # [N*2C0, 2C0, 1, 1]

        if b1_base is not None:
            b1_eff_group = b1_base.unsqueeze(0).expand(N, -1).reshape(-1)  # [N*2C0]
        else:
            b1_eff_group = None

        # FFT 分支：rfft2 → per-sample 动态 1×1 → irfft2
        fre = torch.fft.rfft2(x_1, norm='backward')         # [N, C0, H, W_r]
        real = fre.real
        imag = fre.imag
        mix = torch.cat((real, imag), dim=1)                # [N, 2*C0, H, W_r]

        N_, C_fft, H_, W_r = mix.shape
        assert N_ == N and C_fft == 2 * C0

        mix_group = mix.reshape(1, N * 2 * C0, H_, W_r)     # [1, N*2C0, H, W_r]
        mix_out = F.conv2d(
            mix_group,
            w1_eff_group,
            b1_eff_group,
            stride=1,
            padding=0,
            groups=N                                       # per-sample 动态 1×1 conv
        )
        mix = mix_out.view(N, 2 * C0, H_, W_r)

        real1, imag1 = torch.chunk(mix, 2, dim=1)
        fre_out = torch.complex(real1, imag1)
        x_1 = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        # ============================================================
        # 4) 拼接 + 3×3 融合
        # ============================================================
        output = torch.cat([x_0, IFRFT, x_1], dim=1)
        output = self.conv2(output)
        return output


class SSFENetBlock(nn.Module):
    """
    包一层外壳，把 FRFT 模块当作 SSFENetBlock 来用。

    - 外部接口保持与原来 SSFENetBlock 一致：
        SSFENetBlock(in_channels, out_channels, stride=1,
                     fixed_channels=64, frft_cfg=None, shared_alpha=None)
    - 内部实际做的事情：
        现在只做一件事： y = FRFT(x)
        不再做 1×1 卷积、BN、ReLU、下采样或通道统一。
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 *,
                 fixed_channels: int = 60,
                 frft_cfg=None,
                 shared_alpha: Optional[nn.Parameter] = None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fixed_channels = fixed_channels
        self.frft_cfg = frft_cfg
        self.shared_alpha = shared_alpha

        # 核心：直接用上面的 FRFT 模块
        # 所有超参（order / alpha_min / alpha_max / gate_hidden_ratio）
        # 都在 FRFT 里面写死，不需要外部传入
        self.core = FRFT(in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            (B, in_channels, H, W) —— 形状跟 FRFT 输出一致
        """
        y = self.core(x)
        return y

