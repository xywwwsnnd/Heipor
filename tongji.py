# tools/compute_fg_bg_ratio.py
# -*- coding: utf-8 -*-
"""
统计训练集/测试集的 前景/背景 像素比，并给出建议的 BCE pos_weight = neg/pos
前景定义：label != 0
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ====== 你的数据路径（已按你的要求写死） ======
data_dir = "/home/bitmhsi/data_mouth"
hyperspectral_dir = f"{data_dir}/roi_tif"   # HSI: .tif, C=60
rgb_dir          = ""                       # 不使用真实RGB，留空即可
label_dir        = f"{data_dir}/roi_label"  # Label: .tif, 单通道
train_list       = f"{data_dir}/train.txt"
test_list        = f"{data_dir}/test.txt"

# ====== 你的工程内模块 ======
# - 用数据集类以保证与训练时一致的读法/预处理（但仅用 label 计数，不跑模型）
from vit_model.config import Config
from dataset import HyperspectralDataset


def _to_bin_mask(labels: torch.Tensor) -> torch.Tensor:
    """labels:[B,H,W]或[B,1,H,W] → 0/1 float32"""
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)  # [B,1,H,W]
    labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
    return (labels != 0).to(torch.float32)


@torch.inference_mode()
def count_fg_bg(ds, batch_size: int = 8, num_workers: int = 4):
    """
    遍历数据集，返回 (pos_pixels, neg_pixels, total_pixels)
    只用 CPU，仅读 label 计数。
    注意：这里使用的是 Dataset 的输出尺寸（通常等于 Config.image_size）。
    """
    ld = DataLoader(
        ds,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=False,
        persistent_workers=False,
    )

    pos_total = 0
    pix_total = 0

    for _hsi, _rgb, lab in tqdm(ld, desc="counting", leave=False):
        if not isinstance(lab, torch.Tensor):
            lab = torch.as_tensor(lab)
        mask = _to_bin_mask(lab)  # [B,1,H,W]
        pos_total += int(mask.sum().item())
        pix_total += int(mask.numel())

    neg_total = pix_total - pos_total
    return pos_total, neg_total, pix_total


def fmt_ratio(pos, neg, total):
    pos_pct = 100.0 * pos / max(1, total)
    neg_pct = 100.0 * neg / max(1, total)
    posw = neg / max(1, pos)
    return pos_pct, neg_pct, posw


def _check_paths():
    missing = []
    for p in [hyperspectral_dir, label_dir, os.path.dirname(train_list), os.path.dirname(test_list)]:
        if p and not os.path.exists(p):
            missing.append(p)
    for p in [train_list, test_list]:
        if not os.path.isfile(p):
            missing.append(p)
    if missing:
        print("[WARN] 以下路径不存在，请检查：")
        for p in missing:
            print("  -", p)
        if train_list in missing or test_list in missing:
            print("train/test list 文件缺失时，脚本无法运行。")
            sys.exit(1)


def main():
    _check_paths()

    # 和训练保持一致的 image_size / workers / batch_size
    image_size = getattr(Config, "image_size", 256)
    bs = max(1, getattr(Config, "batch_size", 4))
    nw = max(0, getattr(Config, "num_workers", 4))

    # ===== 构建数据集（与 train.py 一致） =====
    tr_ds = HyperspectralDataset(
        hyperspectral_dir, rgb_dir, label_dir, train_list,
        image_size=image_size, train=True
    )
    te_ds = HyperspectralDataset(
        hyperspectral_dir, rgb_dir, label_dir, test_list,
        image_size=image_size, train=False
    )

    # ===== 训练集 =====
    print("\n=== TRAIN SET ===")
    tr_pos, tr_neg, tr_tot = count_fg_bg(tr_ds, batch_size=bs, num_workers=nw)
    tr_pos_pct, tr_neg_pct, tr_posw = fmt_ratio(tr_pos, tr_neg, tr_tot)
    print(f"pixels total: {tr_tot:,}")
    print(f"  FG (label!=0): {tr_pos:,}  ({tr_pos_pct:.3f}%)")
    print(f"  BG (label==0): {tr_neg:,}  ({tr_neg_pct:.3f}%)")
    print(f"  suggested BCE pos_weight (neg/pos): {tr_posw:.6f}")

    # ===== 测试集 =====
    print("\n=== TEST SET ===")
    te_pos, te_neg, te_tot = count_fg_bg(te_ds, batch_size=bs, num_workers=nw)
    te_pos_pct, te_neg_pct, te_posw = fmt_ratio(te_pos, te_neg, te_tot)
    print(f"pixels total: {te_tot:,}")
    print(f"  FG (label!=0): {te_pos:,}  ({te_pos_pct:.3f}%)")
    print(f"  BG (label==0): {te_neg:,}  ({te_neg_pct:.3f}%)")
    print(f"  suggested BCE pos_weight (neg/pos): {te_posw:.6f}")

    # ===== 总体 =====
    print("\n=== OVERALL (TRAIN+TEST) ===")
    all_pos, all_neg, all_tot = tr_pos + te_pos, tr_neg + te_neg, tr_tot + te_tot
    all_pos_pct, all_neg_pct, all_posw = fmt_ratio(all_pos, all_neg, all_tot)
    print(f"pixels total: {all_tot:,}")
    print(f"  FG (label!=0): {all_pos:,}  ({all_pos_pct:.3f}%)")
    print(f"  BG (label==0): {all_neg:,}  ({all_neg_pct:.3f}%)")
    print(f"  suggested BCE pos_weight (neg/pos): {all_posw:.6f}")

    print("\nTips:")
    print(" - 训练用的 pos_weight 建议优先采用 TRAIN SET 的 neg/pos（更贴近训练分布）。")
    print(" - 在 vit_model/config.py 里： bce_pos_weight = torch.tensor(<tr_posw>)")
    print(" - 多卡训练时记得在损失里 .to(device)。")
    print(" - 本脚本按 Dataset 的输出尺寸计数（通常是 Config.image_size 的裁剪/缩放后尺寸）。")


if __name__ == "__main__":
    main()
