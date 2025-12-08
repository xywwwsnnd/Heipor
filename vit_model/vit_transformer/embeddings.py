# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from typing import Optional

from vit_model.layers.complex_conv import ComplexConv2d, DualGroupConv2d
from vit_model.layers.frft_fdconv import FrFDConv  # <<< 使用 FRFDConv 做 1x1 对齐
from vit_model import config as cfg  # USE_COMPLEX / FULL_COMPLEX flags
from vit_model.vit_transformer.band_selector import DynamicBandSelector
from vit_model.vit_transformer.hybrid_resnet import FuseResNetV2, SELayer

# 路径确认（防止导入了别处的 complex_conv）
import vit_model.layers.complex_conv as cc
print("[Path] complex_conv.py =", cc.__file__)


# --------------------------- Cross-Attention 2D --------------------------- #
class CrossAttn2D(nn.Module):
    """
    Multi-head cross-attention over 2D feature maps.

    - Primary direction: HSI as Query, RGB as Key/Value.
    - Optional symmetric direction: RGB as Query, HSI as Key/Value.

    New:
      - forward_bi(...) -> returns two directional outputs aligned to each query.
      - proj layers are zero-initialized to make residual start as identity.
    """
    def __init__(
        self,
        channels: int,
        heads: int = 4,
        dim_head: int = 32,
        dropout: float = 0.0,
        symmetric: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.dim_head = dim_head
        self.inner = heads * dim_head
        self.symmetric = symmetric

        # HSI <- RGB
        self.to_q_hsi = nn.Linear(channels, self.inner, bias=False)
        self.to_k_rgb = nn.Linear(channels, self.inner, bias=False)
        self.to_v_rgb = nn.Linear(channels, self.inner, bias=False)
        self.proj_hsi = nn.Linear(self.inner, channels, bias=True)

        self.drop = nn.Dropout(dropout)

        if self.symmetric:
            # RGB <- HSI
            self.to_q_rgb = nn.Linear(channels, self.inner, bias=False)
            self.to_k_hsi = nn.Linear(channels, self.inner, bias=False)
            self.to_v_hsi = nn.Linear(channels, self.inner, bias=False)
            self.proj_rgb = nn.Linear(self.inner, channels, bias=True)

        # --- Stabilizing init: make residual path start ~ identity ---
        nn.init.zeros_(self.proj_hsi.weight); nn.init.zeros_(self.proj_hsi.bias)
        if self.symmetric:
            nn.init.zeros_(self.proj_rgb.weight); nn.init.zeros_(self.proj_rgb.bias)

    def _mh_attn(self, q, k, v):
        """
        q, k, v: [B, N, C]
        return:  [B, N, C]
        """
        B, N, _ = q.shape
        H = self.heads
        Dh = self.dim_head

        q = q.view(B, N, H, Dh).transpose(1, 2).contiguous()  # [B, H, N, Dh]
        k = k.view(B, N, H, Dh).transpose(1, 2).contiguous()
        v = v.view(B, N, H, Dh).transpose(1, 2).contiguous()

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=False
        )  # [B, H, N, Dh]

        out = out.transpose(1, 2).contiguous().view(B, N, H * Dh)  # [B, N, C]
        return out

    def forward_once(self, q_2d: torch.Tensor, k_2d: torch.Tensor) -> torch.Tensor:
        """
        One direction: Q from q_2d (HSI), K/V from k_2d (RGB).
        q_2d: [B, C, H, W]; k_2d will be aligned to (H, W)
        """
        B, C, H, W = q_2d.shape
        if k_2d.shape[-2:] != (H, W):
            k_2d = F.interpolate(k_2d, size=(H, W), mode="bilinear", align_corners=False)

        q = q_2d.flatten(2).transpose(1, 2)  # [B, N, C]
        k = k_2d.flatten(2).transpose(1, 2)
        v = k

        q = self.to_q_hsi(q)
        k = self.to_k_rgb(k)
        v = self.to_v_rgb(v)

        o = self._mh_attn(q, k, v)  # [B, N, C]
        o = self.proj_hsi(o).transpose(1, 2).view(B, C, H, W)
        return o

    def forward_bi(self, feat_rgb: torch.Tensor, feat_hsi: torch.Tensor):
        """
        Returns:
          out_hsi: aligned to HSI (HSI <- RGB)
          out_rgb: aligned to RGB (RGB <- HSI)
        """
        assert self.symmetric, "forward_bi requires symmetric=True"

        # HSI <- RGB
        out_hsi = self.forward_once(q_2d=feat_hsi, k_2d=feat_rgb)

        # RGB <- HSI
        B, C, Hr, Wr = feat_rgb.shape
        hsi_kv = feat_hsi
        if hsi_kv.shape[-2:] != (Hr, Wr):
            hsi_kv = F.interpolate(hsi_kv, size=(Hr, Wr), mode="bilinear", align_corners=False)

        q = feat_rgb.flatten(2).transpose(1, 2)
        k = hsi_kv.flatten(2).transpose(1, 2)
        v = k

        q = self.to_q_rgb(q)
        k = self.to_k_hsi(k)
        v = self.to_v_hsi(v)

        o = self._mh_attn(q, k, v)
        out_rgb = self.proj_rgb(o).transpose(1, 2).view(B, C, Hr, Wr)
        return out_hsi, out_rgb

    def forward(self, feat_rgb: torch.Tensor, feat_hsi: torch.Tensor) -> torch.Tensor:
        out_hsi = self.forward_once(q_2d=feat_hsi, k_2d=feat_rgb)
        if not self.symmetric:
            return out_hsi
        out_hsi2, out_rgb = self.forward_bi(feat_rgb, feat_hsi)
        if out_rgb.shape[-2:] != out_hsi.shape[-2:]:
            out_rgb = F.interpolate(out_rgb, size=out_hsi.shape[-2:], mode="bilinear", align_corners=False)
        return 0.5 * (out_hsi + out_hsi2)


# --------------------------- Embeddings --------------------------- #
class Embeddings(nn.Module):
    """
    Two-branch design with per-stage fusion.
    RGB (TransPath) pyramid: [256, 256, 256, 128]
    HSI (Hybrid-ResNet)    : [256, 256, 128,  64] -> 1x1 to [256, 256, 256, 128]

    Tokens: force 16x16 -> 256 tokens; fixed 256-length pos-embed.

    New (config 可选项，未设置也有默认值):
      - use_ssfenet: bool
      - ssfenet_passes: int, default 3
      - ssfenet_strides: tuple/list, default (1,1,4) when passes=3; (1,4) when passes=2; (4,) when passes=1
      - ssfenet_out_ch: int, default 64
      - ssfenet_shared_alpha: bool, default False

      - use_long_skip: bool, default True
      - long_skip_beta_init: float in (0,1), default 0.6  # 凸组合权重初值，更偏向第1遍
      - long_skip_rms_norm: bool, default True            # 逐样本RMS等能量再相加
    """
    def __init__(self, config, img_size, in_channels_rgb: int = 3, in_channels_hsi: int = 60):
        super().__init__()
        self.use_ssfenet = getattr(config, "use_ssfenet", False)

        # Pseudo-RGB band selection
        self.use_pseudo_rgb = bool(getattr(config, "use_pseudo_rgb", False))
        self.pseudo_rgb_temperature = float(getattr(config, "pseudo_rgb_temperature", 1.0))
        self.lambda_peak = float(getattr(config, "pseudo_rgb_lambda_peak", 0.02))
        self.lambda_div  = float(getattr(config, "pseudo_rgb_lambda_div", 0.05))

        # TransPath (RGB)
        self.use_transpath_rgb = bool(getattr(config, "use_transpath_rgb", False))
        self.transpath_hidden  = int(getattr(config, "transpath_hidden", 384))
        self.transpath_ckpt    = getattr(config, "transpath_ckpt", None)
        self.transpath_variant = getattr(config, "transpath_variant", "vit_small")

        # Hybrid-ResNet (HSI)
        self.use_hybrid_resnet = bool(getattr(config, "use_hybrid_resnet", False))

        # Fusion policy
        self.use_shallow_fusion = bool(getattr(config, "use_shallow_fusion", True))
        self.fusion_policy = getattr(config, "fusion_policy", "se_only")
        self.ca_heads      = int(getattr(config, "cross_attn_heads", 16))
        self.ca_dim_head   = int(getattr(config, "cross_attn_dim_head", 32))
        self.ca_dropout    = float(getattr(config, "cross_attn_dropout", 0.0))
        self.ca_symmetric  = bool(getattr(config, "cross_attn_symmetric", True))
        alpha_init = float(getattr(config, "alpha_init", 0.5))
        beta_init  = float(getattr(config, "beta_init", 0.5))

        # === 新增：保留 FRFT 配置以透传下游模块（如 SSFENetBlock） ===
        self.frft_cfg = getattr(config, "frft", None)

        # === 新增：长残差凸组合的开关与参数 ===
        self.use_long_skip       = bool(getattr(config, "use_long_skip", True))
        self.long_skip_beta_init = float(getattr(config, "long_skip_beta_init", 0.6))
        self.long_skip_rms_norm  = bool(getattr(config, "long_skip_rms_norm", True))
        # learnable beta in (0,1)
        self.beta_long_raw = nn.Parameter(torch.logit(torch.tensor(self.long_skip_beta_init)))
        # 懒创建的 FRFD 1×1 对齐层（仅在形状不一致时创建）
        self.long_align: Optional[nn.Module] = None

        img_size = _pair(img_size)

        # Force 16x16 grid to make 256 tokens
        self.grid_size = (16, 16)
        self.num_patches = 256
        patch = (img_size[0] // 16, img_size[1] // 16)
        self.patch_hw = patch

        # Band selector
        self.band_sel = None
        if self.use_pseudo_rgb:
            self.band_sel = DynamicBandSelector(
                in_ch=in_channels_hsi,
                hidden=getattr(config, "pseudo_rgb_hidden", 128),
                temperature=self.pseudo_rgb_temperature
            )

        # TransPath RGB encoder
        if self.use_transpath_rgb:
            from vit_model.vit_transformer.transpath_wrapper import TransPathRGBEncoder
            self.rgb_encoder_tp = TransPathRGBEncoder(
                img_size=img_size[0],
                patch_size=patch[0],
                hidden=self.transpath_hidden,
                variant=self.transpath_variant,
                out_channels_top=256,
                skip_channels=(256, 256, 128),
                pretrained_path=self.transpath_ckpt,
            )
            self.rgb_token_proj = nn.Conv2d(256, config.hidden_size, kernel_size=1, bias=False)
        else:
            self.patch_embeddings_rgb = nn.Conv2d(
                in_channels_rgb, config.hidden_size, kernel_size=patch, stride=patch
            )

        # Hybrid-ResNet for HSI
        if self.use_hybrid_resnet:
            self.hybrid_model = FuseResNetV2(
                block_units=config.resnet["num_layers"],
                width_factor=config.resnet["width_factor"],
                num_classes=config.n_classes,
                # SSFENet 输出已经是 60 通道，这里直接用 in_channels_hsi（=60）
                in_channels_hsi=in_channels_hsi,
            )

            # === 支持多遍 SSFENet（串行级联），默认 3 遍 strides=(1,1,4)；向下兼容单块 ===
            if self.use_hybrid_resnet:
                self.hybrid_model = FuseResNetV2(
                    block_units=config.resnet["num_layers"],
                    width_factor=config.resnet["width_factor"],
                    num_classes=config.n_classes,
                    # HSI 输入通道就是 60
                    in_channels_hsi=in_channels_hsi,
                )

                # === 仅一遍 SSFENet，stride=1，通道固定 60 ===
                if self.use_ssfenet:
                    from vit_model.vit_transformer.ssfenet_block import SSFENetBlock

                    # 输出通道固定为 HSI 通道数（=60）
                    self.ssfenet_out_ch = in_channels_hsi  # 60

                    # 可选共享 α，减少自由度
                    shared_alpha = None
                    if bool(getattr(config, "ssfenet_shared_alpha", False)):
                        shared_alpha = nn.Parameter(
                            torch.tensor(float(getattr(config, "alpha_init", 0.5)))
                        )
                        self.register_parameter("ssfenet_shared_alpha", shared_alpha)

                    # 只建一个模块：SSFENetBlock(60, 60, stride=1)
                    self.my_spectral_module = SSFENetBlock(
                        in_channels_hsi,  # in_channels = 60
                        in_channels_hsi,  # out_channels = 60（虽然内部现在没用）
                        stride=1,
                        frft_cfg=self.frft_cfg,
                        shared_alpha=shared_alpha,
                    )

                    print(f"[SSFENet] single-pass, stride=1, ch={self.ssfenet_out_ch}")


            # ---- HSI s2/s1 的 1×1 通道对齐（保持原逻辑不变） ----
            self.hsi_proj_s2 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True)
            )
            self.hsi_proj_s1 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True)
            )

            # ---- HSI token projection: 强制 1x1 / s=1 / p=0 ----
            if cfg.USE_COMPLEX and not cfg.FULL_COMPLEX:
                self.hsi_mode = 'dual'
                hsi_in = 256 * 2
                self.hsi_token_proj = nn.Conv2d(
                    hsi_in, config.hidden_size,
                    kernel_size=1, stride=1, padding=0,
                    groups=2, bias=False
                )
            elif cfg.USE_COMPLEX and cfg.FULL_COMPLEX:
                self.hsi_mode = 'complex'
                hsi_in = 256
                self.hsi_token_proj = ComplexConv2d(
                    hsi_in, config.hidden_size,
                    kernel_size=1, stride=1, padding=0, bias=False
                )
            else:
                self.hsi_mode = 'real'
                hsi_in = 256
                self.hsi_token_proj = nn.Conv2d(
                    hsi_in, config.hidden_size,
                    kernel_size=1, stride=1, padding=0, bias=False
                )

            # —— 初始化期形状自检（确保 16x16 -> 16x16）——
            with torch.no_grad():
                _dbg_in = torch.zeros(1, hsi_in, 16, 16)
                _dbg_out = self.hsi_token_proj(_dbg_in)
                print("[DBG] hsi_token_proj =", self.hsi_token_proj)
                with torch.no_grad():
                    _x = torch.zeros(1, 256 * 2, 16, 16, dtype=torch.float32)
                    _y = self.hsi_token_proj(_x)
                    print("[DBG] hsi_token_proj IN->OUT:", tuple(_x.shape), "->", tuple(_y.shape))
                print("[DBG] hsi_token_proj IN->OUT:", tuple(_dbg_in.shape), "->", tuple(_dbg_out.shape))
                if _dbg_out.shape[-2:] != (16, 16):
                    print(f"[WARN] hsi_token_proj produced spatial {_dbg_out.shape[-2:]}, expected (16, 16). "
                          f"Will center-crop/align in forward to enforce 16×16.")
        else:
            # no hybrid-resnet path -> simple patch embedding for HSI
            if cfg.USE_COMPLEX and not cfg.FULL_COMPLEX:
                self.hsi_mode = 'dual'
                ConvHSI, hsi_in = DualGroupConv2d, in_channels_hsi * 2
            elif cfg.USE_COMPLEX and cfg.FULL_COMPLEX:
                self.hsi_mode = 'complex'
                ConvHSI, hsi_in = ComplexConv2d, in_channels_hsi
            else:
                self.hsi_mode = 'real'
                ConvHSI, hsi_in = nn.Conv2d, in_channels_hsi
            self.patch_embeddings_hsi = ConvHSI(
                hsi_in, config.hidden_size, kernel_size=patch, stride=patch
            )

        # SE fusion (per stage)
        if self.use_shallow_fusion:
            self.se_top = SELayer(256, 256, reduction=16)
            self.se_s3  = SELayer(256, 256, reduction=16)
            self.se_s2  = SELayer(256, 256, reduction=16)
            self.se_s1  = SELayer(128, 128, reduction=16)

        # Cross-attention heads (match channels per stage)
        def _mk_ca(ch):
            return CrossAttn2D(
                channels=ch,
                heads=self.ca_heads,
                dim_head=self.ca_dim_head,
                dropout=self.ca_dropout,
                symmetric=self.ca_symmetric
            )

        self.ca_top = _mk_ca(256)
        self.ca_s3  = _mk_ca(256)
        self.ca_s2  = _mk_ca(256)
        self.ca_s1  = _mk_ca(128)

        # 1x1 conv for concat policy
        def _mk_concat(ch):
            return nn.Sequential(
                nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            )

        self.concat_top = _mk_concat(256)
        self.concat_s3  = _mk_concat(256)
        self.concat_s2  = _mk_concat(256)
        self.concat_s1  = _mk_concat(128)

        # Learnable weights for residual/replace policies
        self.alpha_top = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_s3  = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_s2  = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_s1  = nn.Parameter(torch.tensor(alpha_init))

        self.beta_top = nn.Parameter(torch.tensor(beta_init))
        self.beta_s3  = nn.Parameter(torch.tensor(beta_init))
        self.beta_s2  = nn.Parameter(torch.tensor(beta_init))
        self.beta_s1  = nn.Parameter(torch.tensor(beta_init))

        # ---- New: modules for BI-XAttn strategies ----
        def _mk_bi_post(ch, p=0.1):
            m = nn.Sequential(
                nn.BatchNorm2d(2 * ch),             # post-norm on concat
                nn.Conv2d(2 * ch, ch, 1, bias=False),
                nn.GELU(),
                nn.Dropout(p),
            )
            with torch.no_grad():
                m[1].weight.zero_()
                for i in range(ch):
                    m[1].weight[i, i, 0, 0]       = 0.5   # rgb side
                    m[1].weight[i, ch + i, 0, 0]  = 0.5   # hsi side
            return m

        ppost = max(0.1, self.ca_dropout)
        self.bi_post_top = _mk_bi_post(256, p=ppost)
        self.bi_post_s3  = _mk_bi_post(256, p=ppost)
        self.bi_post_s2  = _mk_bi_post(256, p=ppost)
        self.bi_post_s1  = _mk_bi_post(128, p=ppost)

        # Directional residual gates (start from 0 -> safe identity)
        z = 0.0
        self.gamma_rgb_top = nn.Parameter(torch.tensor(z)); self.gamma_hsi_top = nn.Parameter(torch.tensor(z))
        self.gamma_rgb_s3  = nn.Parameter(torch.tensor(z)); self.gamma_hsi_s3  = nn.Parameter(torch.tensor(z))
        self.gamma_rgb_s2  = nn.Parameter(torch.tensor(z)); self.gamma_hsi_s2  = nn.Parameter(torch.tensor(z))
        self.gamma_rgb_s1  = nn.Parameter(torch.tensor(z)); self.gamma_hsi_s1  = nn.Parameter(torch.tensor(z))

        # Output residual gates back to base (SE-sum or raw-sum)
        self.delta_top = nn.Parameter(torch.tensor(z))
        self.delta_s3  = nn.Parameter(torch.tensor(z))
        self.delta_s2  = nn.Parameter(torch.tensor(z))
        self.delta_s1  = nn.Parameter(torch.tensor(z))

        # Positional embedding for 256 tokens (no interpolation)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

        self._dbg_printed = False

    # --------------------------- helpers --------------------------- #
    def _prep_hsi_token_input(self, f256: torch.Tensor) -> torch.Tensor:
        """Prepare HSI top feature for 1x1 token projection depending on complex/dual mode."""
        if self.hsi_mode == 'dual':
            return torch.cat([f256, torch.zeros_like(f256)], dim=1)
        if self.hsi_mode == 'complex':
            return f256.to(torch.complex64)
        return f256

    @staticmethod
    def _force_16x16_map(feat: torch.Tensor) -> torch.Tensor:
        """严格把空间尺寸对齐到 16×16（优先中心裁剪，其次插值）。"""
        H, W = feat.shape[-2], feat.shape[-1]
        if (H, W) == (16, 16):
            return feat
        # 优先中心裁剪（如果更大）
        if H >= 16 and W >= 16:
            sh = (H - 16) // 2
            sw = (W - 16) // 2
            return feat[..., sh:sh+16, sw:sw+16]
        # 否则插值到 16×16（通常发生在异常情况下）
        return F.interpolate(feat, size=(16, 16), mode="bilinear", align_corners=False)

    def _fuse_stage(
        self,
        rgb: torch.Tensor,
        hsi: torch.Tensor,
        se_layer: SELayer,
        ca_layer: CrossAttn2D,
        alpha: nn.Parameter,
        beta: nn.Parameter,
        concat_head: nn.Module,
        name: str,
    ) -> torch.Tensor:
        """Per-stage fusion: SE -> Cross-Attn (according to policy)."""
        se_out, _ = se_layer(rgb, hsi)
        if self.fusion_policy == "se_only":
            return se_out

        xattn = ca_layer(rgb, hsi)  # internally aligned to HSI
        if xattn.shape[-2:] != se_out.shape[-2:]:
            xattn = F.interpolate(xattn, size=se_out.shape[-2:], mode="bilinear", align_corners=False)

        if self.fusion_policy == "xattn_residual":
            fused = se_out + alpha * xattn
        elif self.fusion_policy == "xattn_replace":
            fused = (1.0 - beta) * se_out + beta * xattn
        elif self.fusion_policy == "xattn_concat":
            fused = concat_head(torch.cat([se_out, xattn], dim=1))
        else:
            fused = se_out
        return fused

    def _fuse_stage_bi(
        self,
        rgb: torch.Tensor,
        hsi: torch.Tensor,
        ca_layer: CrossAttn2D,
        post_head: nn.Module,
        gamma_rgb: nn.Parameter,
        gamma_hsi: nn.Parameter,
        delta_gate: nn.Parameter,
        use_se: bool,
        se_layer: Optional[SELayer] = None,
        name: str = "",
    ) -> torch.Tensor:
        """Bilateral cross-attention fusion with residuals."""
        if not ca_layer.symmetric:
            raise AssertionError(f"{name}: bi-xattn requires ca_layer.symmetric=True")

        if use_se:
            assert se_layer is not None, "SE layer must be provided when use_se=True"
            se_out, _ = se_layer(rgb, hsi)
            base = se_out
        else:
            if rgb.shape[-2:] != hsi.shape[-2:]:
                hsi = F.interpolate(hsi, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            base = rgb + hsi

        out_hsi, out_rgb = ca_layer.forward_bi(feat_rgb=rgb, feat_hsi=hsi)
        if out_rgb.shape[-2:] != base.shape[-2:]:
            out_rgb = F.interpolate(out_rgb, size=base.shape[-2:], mode="bilinear", align_corners=False)
        if out_hsi.shape[-2:] != base.shape[-2:]:
            out_hsi = F.interpolate(out_hsi, size=base.shape[-2:], mode="bilinear", align_corners=False)

        r_aligned, h_aligned = rgb, hsi
        if r_aligned.shape[-2:] != base.shape[-2:]:
            r_aligned = F.interpolate(r_aligned, size=base.shape[-2:], mode="bilinear", align_corners=False)
        if h_aligned.shape[-2:] != base.shape[-2:]:
            h_aligned = F.interpolate(h_aligned, size=base.shape[-2:], mode="bilinear", align_corners=False)

        rgb_enh = r_aligned + gamma_rgb * out_rgb
        hsi_enh = h_aligned + gamma_hsi * out_hsi

        fused = post_head(torch.cat([rgb_enh, hsi_enh], dim=1))
        out = base + delta_gate * fused
        return out

    def _resize_pos_to_tokens(self, _tokens: torch.Tensor) -> torch.Tensor:
        """Fixed 256-token positional embedding (no-op)."""
        return self.pos_embed

    # --------------------------- forward --------------------------- #
    def forward(self, x_rgb: torch.Tensor, x_hsi: torch.Tensor):
        assert self.use_transpath_rgb and self.use_hybrid_resnet, \
            "Please enable both use_transpath_rgb=True and use_hybrid_resnet=True in config."

        # 统一的正则字典
        reg_dict = {"peak": torch.tensor(0.0, device=x_rgb.device),
                    "div":  torch.tensor(0.0, device=x_rgb.device)}

        # 1) Optional pseudo-RGB band selection (with regularization)
        x_rgb_in = x_rgb
        if self.band_sel is not None:
            pseudo_rgb, reg = self.band_sel(x_hsi)   # reg 是 dict {"peak":..., "div":...}
            x_rgb_in = pseudo_rgb
            # 合并 band selector 的正则
            for k in reg_dict.keys():
                if k in reg and isinstance(reg[k], torch.Tensor):
                    reg_dict[k] = reg_dict[k] + reg[k].to(reg_dict[k].device, reg_dict[k].dtype)

        # 2) RGB -> TransPath
        feats_tp = self.rgb_encoder_tp(x_rgb_in)
        tp_top, tp_s3, tp_s2, tp_s1 = feats_tp  # [256, 256, 256, 128]

        # 3) HSI -> SSFENet (可多遍) -> Hybrid-ResNet
        hsi_for_backbone = x_hsi
        if hasattr(self, "my_spectral_modules"):
            y = x_hsi
            y1 = None
            for idx, m in enumerate(self.my_spectral_modules):
                y = m(y)
                if idx == 0:
                    y1 = y  # 记录第一遍输出
            # y 即最后一遍输出（y3）
            if self.use_long_skip and (y1 is not None):
                y3 = y
                y1a = y1

                # 形状/通道对齐（仅在不一致时创建 FRFD 1×1）
                if (y1a.shape[1] != y3.shape[1]) or (y1a.shape[-2:] != y3.shape[-2:]):
                    if self.long_align is None:
                        stride_h = y3.shape[-2] // y1a.shape[-2]
                        stride_w = y3.shape[-1] // y1a.shape[-1]
                        assert stride_h == stride_w and stride_h >= 1, "long-skip: unsupported spatial ratio"

                        # 尝试不同签名，适配你工程里的 FrFDConv
                        try:
                            self.long_align = FrFDConv(
                                in_channels=y1a.shape[1],
                                out_channels=y3.shape[1],
                                kernel_size=1,
                                stride=stride_h,
                                bias=False,
                            )
                        except TypeError:
                            try:
                                self.long_align = FrFDConv(
                                    y1a.shape[1], y3.shape[1], 1, stride_h, bias=False
                                )
                            except TypeError:
                                self.long_align = FrFDConv(
                                    y1a.shape[1], y3.shape[1], kernel_size=1, stride=stride_h
                                )

                    y1a = self.long_align(y1a)

                # 可选：逐样本 RMS 等能量（避免某一路过大）
                if self.long_skip_rms_norm:
                    def _rms(t):
                        return (t.pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-6).sqrt()
                    y1a = y1a * (_rms(y3) / _rms(y1a))

                # 凸组合： (1-β)*y3 + β*y1a
                beta = torch.sigmoid(self.beta_long_raw)  # 标量 β∈(0,1)
                y = (1.0 - beta) * y3 + beta * y1a

            hsi_for_backbone = y

        elif hasattr(self, "my_spectral_module"):
            # 兼容旧的单块写法
            hsi_for_backbone = self.my_spectral_module(hsi_for_backbone)

        out_backbone = self.hybrid_model(x_rgb_in, hsi_for_backbone)

        if isinstance(out_backbone, (tuple, list)) and len(out_backbone) >= 3:
            _, reg_backbone, feats_h = out_backbone
            if reg_backbone is not None:
                # 假设 reg_backbone 只贡献到 "div"
                reg_dict["div"] = reg_dict["div"] + torch.as_tensor(
                    reg_backbone, device=x_rgb.device, dtype=reg_dict["div"].dtype
                )
        else:
            feats_h = out_backbone

        hy_top, hy_s3, hy_s2, hy_s1 = feats_h
        hy_s2 = self.hsi_proj_s2(hy_s2)   # 128 -> 256
        hy_s1 = self.hsi_proj_s1(hy_s1)   #  64 -> 128

        # 4) Stage-wise fusion
        if self.fusion_policy in ("bi_xattn_se_residual", "bi_xattn_only"):
            if not self.ca_symmetric:
                raise AssertionError("bi-xattn policies require config.cross_attn_symmetric=True")
            use_se = (self.fusion_policy == "bi_xattn_se_residual")

            fused_top = self._fuse_stage_bi(
                tp_top, hy_top, self.ca_top, self.bi_post_top,
                self.gamma_rgb_top, self.gamma_hsi_top, self.delta_top,
                use_se=use_se, se_layer=self.se_top if use_se else None, name="top"
            )
            fused_s3 = self._fuse_stage_bi(
                tp_s3, hy_s3, self.ca_s3, self.bi_post_s3,
                self.gamma_rgb_s3, self.gamma_hsi_s3, self.delta_s3,
                use_se=use_se, se_layer=self.se_s3 if use_se else None, name="s3"
            )
            fused_s2, _ = self.se_s2(tp_s2, hy_s2)  # 256
            fused_s1, _ = self.se_s1(tp_s1, hy_s1)  # 128
        else:
            fused_top = self._fuse_stage(tp_top, hy_top, self.se_top, self.ca_top,
                                         self.alpha_top, self.beta_top, self.concat_top, "top")
            fused_s3  = self._fuse_stage(tp_s3,  hy_s3,  self.se_s3,  self.ca_s3,
                                         self.alpha_s3, self.beta_s3, self.concat_s3,  "s3")
            fused_s2  = self._fuse_stage(tp_s2,  hy_s2,  self.se_s2,  self.ca_s2,
                                         self.alpha_s2, self.beta_s2, self.concat_s2,  "s2")
            fused_s1  = self._fuse_stage(tp_s1,  hy_s1,  self.se_s1,  self.ca_s1,  "s1")

        feats_out = [fused_top, fused_s3, fused_s2, fused_s1]

        # 5) Tokens (force 16x16 -> 256 tokens)
        # RGB branch
        tp_top_g = F.adaptive_avg_pool2d(tp_top, output_size=(16, 16))
        rgb_feat_map = self.rgb_token_proj(tp_top_g)
        if rgb_feat_map.shape[-2:] != (16, 16):
            rgb_feat_map = self._force_16x16_map(rgb_feat_map)
        x_rgb_tok = rgb_feat_map.flatten(2).transpose(1, 2)  # [B, 256, C]

        # HSI branch
        hy_top_g = F.adaptive_avg_pool2d(hy_top, output_size=(16, 16))
        hsi_token_in = self._prep_hsi_token_input(hy_top_g)
        hsi_feat_map = self.hsi_token_proj(hsi_token_in)
        if hsi_feat_map.shape[-2:] != (16, 16):
            hsi_feat_map = self._force_16x16_map(hsi_feat_map)
        x_hsi_tok = hsi_feat_map.flatten(2).transpose(1, 2)  # [B, 256, C]

        # Sanity check
        assert x_rgb_tok.size(1) == 256 and x_hsi_tok.size(1) == 256, \
            f"unexpected token count: rgb={x_rgb_tok.size(1)}, hsi={x_hsi_tok.size(1)} (expected 256)."

        # 6) Positional embedding + dropout
        pe_rgb = self.pos_embed
        pe_hsi = self.pos_embed
        x_rgb_tok = self.dropout(x_rgb_tok + pe_rgb)
        x_hsi_tok = self.dropout(x_hsi_tok + pe_hsi)

        return x_rgb_tok, x_hsi_tok, feats_out, reg_dict
