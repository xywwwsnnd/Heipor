#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trace_dual_branches.py — 双链条（HSI & 伪RGB）训练路径追踪脚本

目标：
- 同时追踪：HSI 原始支路、HSI→伪RGB 选带支路；
- 打印『选带权重 Top-K』、『伪RGB 张量形状』；
- 展示两支路如何送入后续（TransPath / Hybrid-ResNet）、融合、skip（严格按训练：skips=[Top,S3,S2,S1]）、Decoder 全流程；
- 每个关键模块打印：进入/输出 (B,C,H,W) + 所在脚本路径。

用法：
  python trace_dual_branches.py --device cuda:0 --img-size 256 --rgb-ch 3 --hsi-ch 60 [--use-medsam] [--lite]

说明：本脚本只做一次前向（no_grad），用于形状与数据流追踪；不会更改模型参数。
"""
import argparse
import inspect
import importlib
import os
from typing import Optional

import torch
import torch.nn as nn

# ---------------- utils ---------------- #
def loc(obj) -> str:
    mod = getattr(obj, "__module__", type(obj).__module__)
    try:
        src = inspect.getsourcefile(obj if inspect.isclass(obj) else obj.__class__)
    except Exception:
        src = None
    return f"{mod} :: {src or '<?>'}"

def fmt(x):
    if isinstance(x, torch.Tensor):
        return f"Tensor{tuple(x.shape)} {str(x.dtype).replace('torch.', '')}"
    if isinstance(x, (list, tuple)):
        return '[' + ', '.join(fmt(t) for t in x[:3]) + (' ...]' if len(x)>3 else ']')
    return str(type(x))

def import_vit_class(modname: Optional[str] = None):
    candidates = [modname,
                  "vit_model.model", "vit_model.vit_model",
                  "vit_model.vit_seg_modeling", "vit_model.vit"]
    errs = []
    for mn in [m for m in candidates if m]:
        try:
            m = importlib.import_module(mn)
            if hasattr(m, "VisionTransformer"):
                return m.VisionTransformer, m.__file__
        except Exception as e:
            errs.append((mn, repr(e)))
    raise ImportError("找不到 VisionTransformer；尝试过：\n" + "\n".join(f"- {mn}: {e}" for mn, e in errs))

# ---------------- hooks ---------------- #
class DualBranchTracer:
    """在 Embeddings 内部，尽最大努力捕获：
       - HSI 原始张量的进入/输出；
       - HSI→伪RGB 的投影/选带（记录权重 Top-K 与伪RGB形状）；
       - TransPath RGB encoder / Hybrid-ResNet HSI encoder 的输入输出形状。
    """
    def __init__(self, project_root: str, hsi_bands: int, topk: int = 6, lite: bool = False):
        self.project_root = os.path.abspath(project_root)
        self.hsi_bands = hsi_bands
        self.topk = topk
        self.lite = lite
        self.printed = set()
        self.pseudo_rgb_tensor = None
        self.pseudo_weight_vec = None  # (B, C_hsi) 或 (C_hsi,)

    def _hook(self, name):
        def fn(mod, inp, out):
            if id(mod) in self.printed:
                return
            self.printed.add(id(mod))
            in_s = fmt(inp[0]) if isinstance(inp, (list, tuple)) and len(inp)>0 else "None"
            out_s = fmt(out)
            print(f"[Hook] {name:<56} IN={in_s:<36} -> OUT={out_s:<36} @ {loc(mod)}")

            # 尝试自动识别伪RGB：输出维度 (B,3,H,W)
            if isinstance(out, torch.Tensor) and out.ndim == 4 and out.shape[1] == 3:
                src = inp[0] if isinstance(inp, (list, tuple)) and len(inp)>0 else None
                if isinstance(src, torch.Tensor) and src.ndim == 4 and src.shape[1] >= 8:
                    self.pseudo_rgb_tensor = out.detach().cpu()
                    print(f"[PseudoRGB] detected @ {name}: {tuple(out.shape)}")

            # 捕获权重向量：寻找 (B,C_hsi) 或 (C_hsi,) 的输出/输入
            cand = None
            seq = []
            if isinstance(out, torch.Tensor):
                seq = [out]
            elif isinstance(out, (list, tuple)):
                seq = list(out)
            for t in seq:
                if isinstance(t, torch.Tensor) and t.ndim in (1,2) and t.shape[-1] == self.hsi_bands:
                    cand = t
                    break
            if cand is None and isinstance(inp, (list, tuple)):
                for t in inp:
                    if isinstance(t, torch.Tensor) and t.ndim in (1,2) and t.shape[-1] == self.hsi_bands:
                        cand = t
                        break
            if cand is not None:
                self.pseudo_weight_vec = cand.detach().cpu()
                # 打印 Top-K
                with torch.no_grad():
                    v = cand.float().abs()
                    if v.ndim == 2:
                        v = v.mean(dim=0)
                    topv, topi = torch.topk(v, k=min(self.topk, v.numel()))
                    idx_list = ', '.join(f"{int(i)}:{float(val):.4f}" for val, i in zip(topv.tolist(), topi.tolist()))
                    print(f"[BandWeight] top-{self.topk}: {idx_list}")
        return fn

    def attach(self, embeddings: nn.Module):
        # 遍历 embeddings 的子模块，基于名字/类名/源码路径匹配
        for name, mod in embeddings.named_modules():
            try:
                src = inspect.getsourcefile(mod.__class__)
            except Exception:
                src = None
            inside = (src and os.path.commonpath([self.project_root, os.path.abspath(src)]) == self.project_root)
            cls = mod.__class__.__name__.lower()
            nm = name.lower()
            if not inside:
                continue

            # 关键：与伪RGB/选带/SSFE相关的模块名特征
            is_pseudo = any(k in (cls+nm) for k in ["pseudo", "rgbproj", "rgb_projector", "rgb_selector"])
            is_band   = any(k in (cls+nm) for k in ["band", "weight", "selector"])
            is_ssfe   = any(k in (cls+nm) for k in ["ssfe", "spectral", "spatio", "noise", "multiscale"])
            is_transp = "transpath" in (cls+nm)
            is_hybrid = any(k in (cls+nm) for k in ["fuseresnet", "hybrid", "se_layer", "basicblock"])

            if is_pseudo or is_band or is_ssfe or is_transp or is_hybrid:
                mod.register_forward_hook(self._hook(f"Emb.{name}"))

# 通用 hooks（Decoder/SegHead/ResNet 全层，类似 verbose 脚本）
def register_verbose_hooks(root: nn.Module, *, project_root: str, lite: bool):
    printed = set()
    def hook_for(name):
        def _fn(mod, inp, out):
            if id(mod) in printed:
                return
            printed.add(id(mod))
            in_s = fmt(inp[0]) if isinstance(inp, (list, tuple)) and len(inp)>0 else "None"
            out_s = fmt(out)
            print(f"[Hook] {name:<56} IN={in_s:<36} -> OUT={out_s:<36} @ {loc(mod)}")
        return _fn

    must_classes = {
        "SSFENetBlock", "BandWeightGenerator", "SpectralSelfAttention",
        "SpatioSpectralAttention", "SpectralNoiseSuppression", "SpatioSpectralMultiScale",
        "FrFDConv", "BasicBlock", "SELayer", "FuseResNetV2", "DecoderBlock"
    }
    verbose_types = (
        nn.Conv2d, nn.ConvTranspose2d,
        nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm,
        nn.ReLU, nn.SiLU, nn.GELU,
        nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d,
        nn.MultiheadAttention,
    )

    for name, mod in root.named_modules():
        cls = mod.__class__.__name__
        add = False
        if cls in must_classes:
            add = True
        elif not lite and isinstance(mod, verbose_types):
            add = True
        if add and not isinstance(mod, nn.MultiheadAttention):
            try:
                src = inspect.getsourcefile(mod.__class__)
            except Exception:
                src = None
            if not src or os.path.commonpath([project_root, os.path.abspath(src)]) != os.path.abspath(project_root):
                continue
        if add:
            mod.register_forward_hook(hook_for(name))

# ---------------- main ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--img-size', type=int, default=None)
    ap.add_argument('--rgb-ch', type=int, default=None)
    ap.add_argument('--hsi-ch', type=int, default=None)
    ap.add_argument('--vit-module', default=None)
    ap.add_argument('--use-medsam', action='store_true')  # 为兼容旧参数保留，但模型里未必使用
    ap.add_argument('--lite', action='store_true')
    ap.add_argument('--topk', type=int, default=6, help='打印选带 top-k 权重')
    args = ap.parse_args()

    from vit_model.config import Config
    cfg = Config()
    vit_cfg = cfg.model_config

    if args.img_size is not None:
        cfg.image_size = args.img_size
    if args.rgb_ch is not None:
        cfg.num_channels_rgb = args.rgb_ch
        vit_cfg.in_channels = args.rgb_ch
    if args.hsi_ch is not None:
        cfg.num_channels_hsi = args.hsi_ch
        vit_cfg.in_channels_hsi = args.hsi_ch
    if args.use_medsam:
        # 仅为兼容旧脚本的 flag；是否生效取决于你的模型实现
        vit_cfg.use_medsam2_encoder = True
        vit_cfg.use_hybrid_resnet = False

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    VitClass, vit_file = import_vit_class(args.vit_module)
    print("=== 模块来源 ===")
    print("VisionTransformer:", vit_file)
    vt = VitClass(vit_cfg, img_size=cfg.image_size, num_classes=cfg.num_classes, vis=False).to(device)
    vt.train()

    project_root = os.path.dirname(os.path.abspath(vit_file)).rsplit('vit_model', 1)[0] + 'vit_model'
    print("Transformer     :", loc(vt.transformer))
    from vit_model.vit_transformer import embeddings as emb_mod
    from vit_model.vit_transformer import encoder as enc_mod
    print("Embeddings (module):", f"{emb_mod.__name__} :: {emb_mod.__file__}")
    print("Encoder   (module):", f"{enc_mod.__name__} :: {enc_mod.__file__}")
    print("DecoderCup       :", loc(vt.decoder))
    print("SegmentationHead :", loc(vt.segmentation_head))

    # 专门为 dual-branch 设计的 Embeddings 内部追踪
    tracer = DualBranchTracer(project_root, hsi_bands=cfg.num_channels_hsi, topk=args.topk, lite=args.lite)
    tracer.attach(vt.transformer.embeddings)

    # 全局详细 hooks（含 Decoder/ResNet 等）
    register_verbose_hooks(vt, project_root=project_root, lite=args.lite)

    # 构造输入
    B, H, W = 1, cfg.image_size, cfg.image_size
    x_rgb = torch.randn(B, cfg.num_channels_rgb, H, W, device=device)
    x_hsi = torch.randn(B, cfg.num_channels_hsi, H, W, device=device)

    print("\n=== 双链条（HSI & 伪RGB）训练路径追踪 ===")
    print(f"[Input] RGB {fmt(x_rgb)} | HSI {fmt(x_hsi)}")

    with torch.no_grad():
        # Embeddings：内部会触发 tracer hooks，打印伪RGB 和选带信息
        emb_out = vt.transformer.embeddings(x_rgb, x_hsi)
        if not (isinstance(emb_out, (list, tuple)) and len(emb_out) == 4):
            raise RuntimeError("embeddings 返回应为 (emb_rgb, emb_hsi, features, reg_loss)")
        emb_rgb, emb_hsi, features, reg_loss = emb_out
        print("[Embeddings] emb_rgb:", fmt(emb_rgb), "emb_hsi:", fmt(emb_hsi))

        # Encoder
        enc_out = vt.transformer.encoder(emb_rgb, emb_hsi)
        enc_rgb, enc_hsi = enc_out[0], enc_out[1]
        print("[Encoder] enc_rgb:", fmt(enc_rgb), "enc_hsi:", fmt(enc_hsi))

        # 融合（训练路径用 Top 作为主输入）
        fused = enc_rgb + enc_hsi
        print("[Fusion] fused_features (Top):", fmt(fused))

        # ----------------------------- 关键改动：正确传 4 个 skip -----------------------------
        # features = [Top, S3, S2, S1]
        n_skip = int(getattr(vit_cfg, "n_skip", 0))
        skip_cfg = list(getattr(vit_cfg, "skip_channels", []))

        if not isinstance(features, (list, tuple)) or len(features) < 4:
            raise RuntimeError(f"[Skips] embeddings 应返回 4 个跳连 [Top,S3,S2,S1]，但拿到 {0 if features is None else len(features)}")

        # 取前 n_skip 个（当 n_skip=4 时即 [Top,S3,S2,S1]；当 n_skip=3 时即 [Top,S3,S2]）
        skips = list(features)[:n_skip]

        names = ["Top", "S3", "S2", "S1"]
        print(f"[Skips] use first {n_skip} from [Top,S3,S2,S1], n_skip= {n_skip}")
        for i, t in enumerate(skips):
            exp_ch = skip_cfg[i] if i < len(skip_cfg) else None
            ch_ok = (exp_ch is None) or (t.shape[1] == exp_ch)
            status = "✓" if ch_ok else "✗"
            print(f"  skip[{i}] {names[i] if i < len(names) else '?'} {fmt(t)}  ⇔  cfg={exp_ch} {status}")

        # 严格一致性检查（提前在 trace 阶段就报错，避免进入 Decoder 再爆）
        if len(skips) != n_skip:
            raise RuntimeError(f"[Skips] 数量不对：期望 {n_skip} 个，实际 {len(skips)} 个。")
        for i, (t, exp_ch) in enumerate(zip(skips, skip_cfg)):
            if exp_ch is None:
                continue
            if t.shape[1] != exp_ch:
                raise RuntimeError(f"[Skips] 通道不匹配：skip[{i}] got {t.shape[1]} but expect {exp_ch}")
        # -------------------------------------------------------------------------------------

        # Decoder & Head
        decoded = vt.decoder(fused, skips)  # 这里传入包含 Top 的 skips
        print("[Decoder] decoded:", fmt(decoded))
        logits = vt.segmentation_head(decoded)
        print("[SegHead] logits:", fmt(logits))
        if logits.ndim == 4:
            print(f"[SegHead] value range: min={logits.min().item():.4f}  max={logits.max().item():.4f}")

        # 汇总：伪RGB 与选带
        if tracer.pseudo_rgb_tensor is not None:
            pr = tracer.pseudo_rgb_tensor
            print(f"[Summary] Pseudo-RGB detected: shape={tuple(pr.shape)}")
        else:
            print("[Summary] Pseudo-RGB NOT detected via hooks（若模块名未匹配，可在 tracer.attach 内补充关键字）")
        if tracer.pseudo_weight_vec is not None:
            print(f"[Summary] Band-weight vector captured: shape={tuple(tracer.pseudo_weight_vec.shape)}")
        else:
            print("[Summary] Band-weight vector NOT captured（若未命中，可在 tracer.attach 内补充关键字）")

    print("\n=== 完成（双链条追踪） ===")

if __name__ == "__main__":
    main()
