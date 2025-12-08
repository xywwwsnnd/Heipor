# vit_model/vit_transformer/cb_context.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossBatchContext(nn.Module):
    """
    轻量跨样本上下文模块（训练期启用，评估期自动关闭）：
    - 将每个样本 (B, C, H, W) 通过自适应平均池化压成 g×g token 网格（S=g*g）。
    - 将整个 batch 的 token 拼接成一个“上下文库”（最多 k_ctx 个，避免显存爆炸）。
    - 对每个样本的 token 作为 Query，在“上下文库”（Key/Value）上做一次 MHA，
      得到上下文增强后的 token，再上采样回 H×W 并以残差方式注入。
    - 初始时残差系数为 0，训练过程中逐步学到跨样本一致性模式。
    - eval() 模式下直接返回 x（恒等映射，不影响测速/推理）。

    Args:
        in_ch:      输入通道
        attn_dim:   注意力内部维度（投影后维数），建议 16~64
        num_heads:  多头数（整除 attn_dim）
        k_ctx:      全局上下文库的 token 上限（例如 64/128），控制显存
        grid_min:   最小网格 g（保证 S=g*g 不为 0）
        mix_init:   初始残差强度（通常设为 0.0，安全）
        dropout:    注意力后 dropout
    """
    def __init__(
        self,
        in_ch: int,
        attn_dim: int = 32,
        num_heads: int = 2,
        k_ctx: int = 128,
        grid_min: int = 4,
        mix_init: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.in_ch     = in_ch
        self.attn_dim  = attn_dim
        self.num_heads = num_heads
        self.k_ctx     = int(k_ctx)
        self.grid_min  = int(grid_min)

        # 线性投影到注意力空间（Q/K/V），再投回
        self.to_q = nn.Linear(in_ch, attn_dim, bias=False)
        self.to_k = nn.Linear(in_ch, attn_dim, bias=False)
        self.to_v = nn.Linear(in_ch, attn_dim, bias=False)
        self.proj = nn.Linear(attn_dim, in_ch, bias=False)

        self.drop = nn.Dropout(dropout)
        # 安全残差门：从 0 开始学习
        self.mix  = nn.Parameter(torch.tensor(float(mix_init), dtype=torch.float32))

        # 将 token 网格回到 H×W 前，做一个轻量 1×1 仿射（提升柔性；初始化为恒等）
        self.out_affine = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.out_affine.weight)
        nn.init.zeros_(self.out_affine.bias)

    @staticmethod
    def _choose_grid(h: int, w: int, k_ctx: int, grid_min: int) -> int:
        # 尝试让每个样本的 token 数 S ~= k_ctx^(1/2)，保证总上下文不超过 k_ctx
        g_auto = int(math.sqrt(max(grid_min * grid_min, k_ctx)))
        g = min(max(grid_min, g_auto), min(h, w))
        return g

    def _pool_to_tokens(self, x: torch.Tensor, g: int) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, S=g*g, C)
        B, C, H, W = x.shape
        px = F.adaptive_avg_pool2d(x, output_size=(g, g))  # (B, C, g, g)
        t  = px.flatten(2).transpose(1, 2).contiguous()    # (B, S, C)
        return t

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q: (S_q, C), k/v: (S_k, C) —— 单样本的 query，对全局库做注意力
        返回: (S_q, C)
        """
        H = self.num_heads
        Dh = self.attn_dim // H

        # 线性投影
        q = self.to_q(q)  # (S_q, attn_dim)
        k = self.to_k(k)  # (S_k, attn_dim)
        v = self.to_v(v)  # (S_k, attn_dim)

        # 拆头: (H, S, Dh)
        q = q.view(-1, H, Dh).transpose(0, 1).contiguous()
        k = k.view(-1, H, Dh).transpose(0, 1).contiguous()
        v = v.view(-1, H, Dh).transpose(0, 1).contiguous()

        # SDPA
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0, is_causal=False)
        # 合并头 -> (S_q, attn_dim)
        out = out.transpose(0, 1).contiguous().view(-1, H * Dh)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # eval() 模式直接短路（不做跨样本操作）
        if not self.training:
            return x

        B, C, H, W = x.shape
        if B < 2:
            # 单样本 batch 时，跨样本上下文无意义，直接返回
            return x

        # 1) 压成 token 网格
        g = self._choose_grid(H, W, self.k_ctx, self.grid_min)
        local_tokens = self._pool_to_tokens(x, g)    # (B, S, C), S=g*g

        # 2) 构建全局上下文库（合并整个 batch 的 token），并限制为 k_ctx
        T_all = local_tokens.reshape(-1, C)          # (B*S, C)
        if T_all.size(0) > self.k_ctx:
            # 轻量随机子采样（也可换成能量 topk）
            idx = torch.randperm(T_all.size(0), device=x.device)[: self.k_ctx]
            ctx = T_all.index_select(0, idx)         # (k_ctx, C)
        else:
            ctx = T_all                               # (<=k_ctx, C)

        # 3) 对每个样本分别做注意力：Q=自己的 tokens，K/V=全局 ctx
        out_tokens = []
        for b in range(B):
            q = local_tokens[b]                       # (S, C)
            att = self._attend(q, ctx, ctx)          # (S, attn_dim)
            att = self.proj(att)                     # (S, C)
            out_tokens.append(att)

        out_tokens = torch.stack(out_tokens, dim=0)   # (B, S, C)

        # 4) tokens -> (B, C, g, g) -> 上采样回 H×W
        y = out_tokens.transpose(1, 2).contiguous().view(B, C, g, g)  # (B, C, g, g)
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)

        # 5) 残差注入（带可学习门控），并做轻量仿射
        y = x + self.mix * y
        y = self.out_affine(y)
        return y
