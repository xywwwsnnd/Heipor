#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oscc_flops.py
-------------
FLOPs / Params for the OSCC fusion-only model (VisionTransformer + Config).

This script assumes the repo layout from the uploaded OSCC project.
It forces Fusion mode and constructs dummy RGB+HSI inputs based on Config.
"""

import sys, os, json
from pathlib import Path
import argparse
import torch
import torch.nn as nn

# Allow running from any cwd: add OSCC root to sys.path
THIS = Path(__file__).resolve()
ROOT = THIS.parent / "OSCC_extracted" / "OSCC"
if not ROOT.exists():
    # also support running from inside OSCC_extracted/OSCC
    maybe = THIS.parent
    if (maybe / "vit_model").exists():
        ROOT = maybe
sys.path.insert(0, str(ROOT))

from vit_model.config import Config
from vit_model.model import VisionTransformer


def human(n: float, unit: str = "", base: float = 1000.0) -> str:
    for s in ["", "K", "M", "G", "T", "P"]:
        if abs(n) < base:
            return f"{n:,.3f} {s}{unit}".strip()
        n /= base
    return f"{n:,.3f} E{unit}"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--img-size', type=int, default=None, help='Override image size')
    ap.add_argument('--hsi-ch', type=int, default=None, help='Override HSI channels')
    ap.add_argument('--rgb-ch', type=int, default=None, help='Override RGB channels (default 3)')
    args = ap.parse_args()

    # Build model from Config
    cfg = Config
    img_size = args.img_size or getattr(cfg, 'image_size', 256)
    hsi_ch = args.hsi_ch or getattr(cfg, 'num_channels_hsi', 60)
    rgb_ch = args.rgb_ch or getattr(cfg, 'num_channels_rgb', 3)
    num_cls = getattr(cfg, 'num_classes', 1)

    print("=== OSCC Fusion Model ===")
    print(f"image_size   : {img_size}")
    print(f"HSI channels : {hsi_ch}")
    print(f"RGB channels : {rgb_ch}")
    print(f"num_classes  : {num_cls}")

    # IMPORTANT: VisionTransformer expects a *model_config* for its Transformer/Embeddings
    # so we pass cfg.model_config, while still giving img_size/num_classes explicitly.
    model = VisionTransformer(cfg.model_config, img_size=img_size, num_classes=num_cls)
    model.eval().to(args.device)

    # Fusion-only dummy inputs
    rgb = torch.randn(1, rgb_ch, img_size, img_size, device=args.device)
    hsi = torch.randn(1, hsi_ch, img_size, img_size, device=args.device)

    # Dry run (try positional first, then kwargs as fallback)
    try:
        out = model(rgb, hsi)  # forward(x_rgb, x_hsi)
    except TypeError:
        out = model(rgb=rgb, hsi=hsi)  # forward(rgb=..., hsi=...)

    # Params
    total_params = sum(p.numel() for p in model.parameters())
    print("\n=== Parameters ===")
    print(f"Params: {human(total_params, ' params')}  ({total_params:,} exact)")

    # Try fvcore, then thop
    try:
        from fvcore.nn import FlopCountAnalysis
        macs = FlopCountAnalysis(model, (rgb, hsi)).total()
        print("\n=== Compute MACs/FLOPs ===")
        print("[fvcore] total_ops (≈MACs):", human(macs, 'Ops'))
        print("[fvcore] approx FLOPs     :", human(2 * macs, 'FLOPs'))
        return
    except Exception as e:
        print(f"[fvcore] skipped: {e}")

    try:
        from thop import profile
        macs, params = profile(model, inputs=(rgb, hsi), verbose=False)
        print("\n=== Compute MACs/FLOPs ===")
        print("[thop] MACs               :", human(macs, 'MACs'))
        print("[thop] approx FLOPs       :", human(2 * macs, 'FLOPs'))
        print("[thop] Params (tool)      :", human(params, ' params'))
        return
    except Exception as e:
        print(f"[thop] skipped: {e}")

    print("\n[Fallback] Install one of: pip install fvcore  OR  pip install thop")


if __name__ == "__main__":
    main()
