# vit_model/modules/bi_gated_decoder_ca.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _win_part(x, win):
    B, C, H, W = x.shape
    assert H % win == 0 and W % win == 0, f"H/W must be multiple of win={win}"
    x = x.view(B, C, H // win, win, W // win, win).permute(0,2,4,1,3,5).contiguous()
    x = x.view(-1, C, win, win)
    return x, (B, C, H, W)

def _win_unpart(x, meta, win):
    B, C, H, W = meta
    Nh, Nw = H // win, W // win
    x = x.view(B, Nh, Nw, C, win, win).permute(0,3,1,4,2,5).contiguous()
    x = x.view(B, C, H, W)
    return x

class BiGatedDecoderCA(nn.Module):
    """
    双向门控解码端 Cross-Attention（窗口化 + SDPA）
    - 方向1：Q = q_src（解码器 H/8 拼接后），K/V = kv_src（HSI s2）
    - 方向2：Q = kv_src，K/V = q_src
    输出：回到 q_src 的通道空间（解码器主干），并以残差形式叠加：out = q_src + δ * fused
    """
    def __init__(self, q_channels, kv_channels, win=7, heads=2, qkv_reduce=2, drop=0.0):
        super().__init__()
        self.win = win
        self.heads = heads
        q_inter  = max(4, q_channels  // qkv_reduce)
        kv_inter = max(4, kv_channels // qkv_reduce)

        # 方向1投影（到 q 空间）
        self.q1 = nn.Conv2d(q_channels, q_inter, 1, bias=False)
        self.k1 = nn.Conv2d(kv_channels, q_inter, 1, bias=False)
        self.v1 = nn.Conv2d(kv_channels, q_inter, 1, bias=False)
        self.o1 = nn.Conv2d(q_inter, q_channels, 1, bias=False)

        # 方向2投影（到 kv 空间）
        self.q2 = nn.Conv2d(kv_channels, kv_inter, 1, bias=False)
        self.k2 = nn.Conv2d(q_channels, kv_inter, 1, bias=False)
        self.v2 = nn.Conv2d(q_channels, kv_inter, 1, bias=False)
        self.o2 = nn.Conv2d(kv_inter, kv_channels, 1, bias=False)

        # 方向门控（可学习，初始0，安全起步）
        z = 0.0
        self.gamma_q  = nn.Parameter(torch.tensor(z))   # 作用在 attn_q
        self.gamma_kv = nn.Parameter(torch.tensor(z))   # 作用在 attn_kv
        self.delta    = nn.Parameter(torch.tensor(z))   # 最终残差门

        # 把 kv_enh 对齐到 q 通道再融合
        self.kv_to_q = nn.Conv2d(kv_channels, q_channels, 1, bias=False)

        # 融合头：concat -> BN -> 1x1 -> GELU -> Dropout
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(q_channels * 2),
            nn.Conv2d(q_channels * 2, q_channels, 1, bias=False),
            nn.GELU(),
            nn.Dropout(drop)
        )
        # 初始化使一开始更接近恒等
        with torch.no_grad():
            self.fuse[1].weight.zero_()

        self.nq1 = nn.BatchNorm2d(q_inter)
        self.nk1 = nn.BatchNorm2d(q_inter)
        self.nv1 = nn.BatchNorm2d(q_inter)
        self.nq2 = nn.BatchNorm2d(kv_inter)
        self.nk2 = nn.BatchNorm2d(kv_inter)
        self.nv2 = nn.BatchNorm2d(kv_inter)

    @staticmethod
    def _to_heads(x, heads):
        # x: [Bn, C, w, w] -> [Bn, heads, L, Ch]
        Bn, C, w, _ = x.shape
        L = w * w
        x = x.flatten(2).transpose(1, 2)            # [Bn, L, C]
        Ch = C // heads
        x = x.view(Bn, L, heads, Ch).permute(0, 2, 1, 3).contiguous()
        return x

    @staticmethod
    def _from_heads(x, C_out, w):
        # x: [Bn, heads, L, Ch] -> [Bn, C_out, w, w]
        Bn, H, L, Ch = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(Bn, L, H * Ch)  # [Bn, L, C']
        x = x.transpose(1, 2).view(Bn, C_out, w, w)
        return x

    def _sdpa_block(self, Q, K, V, heads):
        # Q/K/V: [Bn, C', w, w]
        Bn, C_, w, _ = Q.shape
        qh = self._to_heads(Q, heads)
        kh = self._to_heads(K, heads)
        vh = self._to_heads(V, heads)
        ah = F.scaled_dot_product_attention(qh, kh, vh)             # [Bn, heads, L, Ch]
        out = self._from_heads(ah, C_, w)                            # [Bn, C', w, w]
        return out

    def forward(self, q_src, kv_src):
        """
        q_src : [B, Cq, H, W]  (解码器 H/8 上采样+拼接后的特征，作为 Q1 空间)
        kv_src: [B, Ck, H?,W?] (HSI s2，同尺度；内部自动对齐到 H×W)
        返回：与 q_src 同通道/分辨率的张量
        """
        B, Cq, H, W = q_src.shape
        if kv_src is None:
            return q_src
        if kv_src.shape[-2:] != (H, W):
            kv_src = F.interpolate(kv_src, size=(H, W), mode="bilinear", align_corners=False)

        # 方向1：Q=q_src, K/V=kv_src ——> 回到 q 通道
        Q1 = self.nq1(self.q1(q_src))
        K1 = self.nk1(self.k1(kv_src))
        V1 = self.nv1(self.v1(kv_src))
        Q1w, meta1 = _win_part(Q1, self.win)
        K1w, _     = _win_part(K1, self.win)
        V1w, _     = _win_part(V1, self.win)
        A1w = self._sdpa_block(Q1w, K1w, V1w, self.heads)
        A1  = _win_unpart(A1w, meta1, self.win)
        attn_q = self.o1(A1)                       # [B, Cq, H, W]

        # 方向2：Q=kv_src, K/V=q_src ——> 回到 kv 通道
        Q2 = self.nq2(self.q2(kv_src))
        K2 = self.nk2(self.k2(q_src))
        V2 = self.nv2(self.v2(q_src))
        Q2w, meta2 = _win_part(Q2, self.win)
        K2w, _     = _win_part(K2, self.win)
        V2w, _     = _win_part(V2, self.win)
        A2w = self._sdpa_block(Q2w, K2w, V2w, self.heads)
        A2  = _win_unpart(A2w, meta2, self.win)
        attn_kv = self.o2(A2)                      # [B, Ck, H, W]

        # 残差门控增强
        q_enh  = q_src  + self.gamma_q  * attn_q
        kv_enh = kv_src + self.gamma_kv * attn_kv

        # 对齐到 q 通道并融合
        kv_in_q = self.kv_to_q(kv_enh)             # [B, Cq, H, W]
        fused = self.fuse(torch.cat([q_enh, kv_in_q], dim=1))  # [B, Cq, H, W]

        # 最终残差
        out = q_src + self.delta * fused
        return out
