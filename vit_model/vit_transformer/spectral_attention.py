import torch
import torch.nn as nn
from vit_model.layers.frft_fdconv import FrFDConv
from vit_model.config import Config

# 模块级共享 alpha（当 alpha_shared=True 时，所有实例复用）
_SHARED_ALPHA_SSA = None


class SpectralSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=2, stride=1):
        super(SpectralSelfAttention, self).__init__()
        self.stride = stride
        self.attn_dim = 16  # <<< 固定为16，进一步降低显存

        # ---- 读取 FRFT 配置（无则用安全默认）----
        frft_cfg     = getattr(Config.model_config, "frft", None)
        alpha_init   = 0.5  if frft_cfg is None else float(getattr(frft_cfg, "alpha_init", 0.5))
        use_polar    = True if frft_cfg is None else bool(getattr(frft_cfg, "frfdconv_use_polar", True))
        alpha_shared = False if frft_cfg is None else bool(getattr(frft_cfg, "alpha_shared", False))

        # ---- 共享或独立 alpha 参数 ----
        global _SHARED_ALPHA_SSA
        if alpha_shared:
            if _SHARED_ALPHA_SSA is None:
                _SHARED_ALPHA_SSA = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
            self.alpha = _SHARED_ALPHA_SSA
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # ① 先 1×1 + stride 下采样/规整通道：(B,C,H,W)->(B,outC,H',W')
        self.pre_reduce = FrFDConv(
            in_channels, out_channels, kernel_size=1, stride=stride,
            bias=False, alpha=self.alpha, use_polar=use_polar
        )

        # ② 压到 16 维做注意力（极省显存），再拉回 out_channels
        self.attn_in  = FrFDConv(
            out_channels, self.attn_dim, kernel_size=1, stride=1,
            bias=False, alpha=self.alpha, use_polar=use_polar
        )
        self.attn     = nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=num_heads, batch_first=True)
        self.norm1    = nn.LayerNorm(self.attn_dim)
        self.mlp      = nn.Sequential(
            nn.Linear(self.attn_dim, self.attn_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.attn_dim * 4, self.attn_dim)
        )
        self.norm2    = nn.LayerNorm(self.attn_dim)
        self.attn_out = FrFDConv(
            self.attn_dim, out_channels, kernel_size=1, stride=1,
            bias=False, alpha=self.alpha, use_polar=use_polar
        )

        # ③ 统一到 64 通道给后续模块
        self.channel_fix = FrFDConv(
            out_channels, 64, kernel_size=1, stride=1,
            bias=False, alpha=self.alpha, use_polar=use_polar
        )

        # BN 的通道数与 channel_fix 输出一致（=64）
        self.bn   = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 降采样/规整通道 → 极低维注意力
        x = self.pre_reduce(x)                      # (B, outC, H', W')
        B, C2, H2, W2 = x.size()
        S = H2 * W2

        x = self.attn_in(x)                         # (B, 16, H', W')
        x = x.view(B, self.attn_dim, S).permute(0, 2, 1)  # (B, S, 16)

        # pre-LN 自注意力
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y)           # batch_first=True
        x = x + attn_out

        # pre-LN MLP
        x = x + self.mlp(self.norm2(x))

        # 回到 (B, 16, H', W') → (B, outC, H', W') → (B, 64, H', W')
        x = x.permute(0, 2, 1).view(B, self.attn_dim, H2, W2)
        x = self.attn_out(x)
        x = self.channel_fix(x)
        x = self.bn(x)
        return self.relu(x)


class SpatioSpectralAttention(nn.Module):
    def __init__(self, in_channels, num_heads=2, dropout_rate=0.0):
        super(SpatioSpectralAttention, self).__init__()
        # 纯空间注意力；降头数 + 关dropout 便于触发高效内核
        self.attn  = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(in_channels)
        self.mlp   = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(dropout_rate)
        )
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.size()
        S = H * W
        x = x.view(B, C, S).permute(0, 2, 1)

        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 2, 1).view(B, C, H, W)
