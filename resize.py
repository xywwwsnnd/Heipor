# resize_heipor_256.py
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from htc import DataPath

# 目标空间尺寸
TARGET_H = 256
TARGET_W = 256

# 用哪个标注者
ANNOTATOR = "annotator1"


def main():
    # 1) 读环境变量
    env_path = os.environ.get("PATH_Tivita_HeiPorSPECTRAL")
    if env_path is None:
        raise RuntimeError("请先设置环境变量 PATH_Tivita_HeiPorSPECTRAL")

    dataset_root = Path(env_path)

    # 2) 输出目录：和原数据并列放一个新文件夹
    out_root = dataset_root.parent / "HeiPor_resized_256"
    hsi_out = out_root / "hsi"
    mask_out = out_root / "mask"
    hsi_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    # 3) 用 L1 预处理目录生成 image_name 列表
    l1_dir = dataset_root / "intermediates" / "preprocessing" / "L1"
    if not l1_dir.exists():
        raise RuntimeError(f"L1 目录不存在: {l1_dir}")

    blosc_files = sorted(l1_dir.glob("*.blosc"))
    if not blosc_files:
        raise RuntimeError(f"L1 目录下没有 .blosc 文件: {l1_dir}")

    image_names = [f.stem for f in blosc_files]
    print(f"共找到 {len(image_names)} 个 L1 文件")

    num_ok = 0                 # 成功 resize 的数量
    num_skip_no_seg = 0        # 有 DataPath，但没有分割的
    num_skip_not_in_cache = 0  # DataPath 根本不认识的（不在缓存里的）

    for idx, image_name in enumerate(image_names):
        print(f"\n[{idx+1}/{len(image_names)}] 处理 {image_name} ...")

        # 有些 L1 文件在 DataPath 里没有索引（校准 / 特殊样本），这里直接跳过
        try:
            path = DataPath.from_image_name(image_name)
        except AssertionError:
            print(f"  -> {image_name} 不在 DataPath 缓存里（可能是校准/特殊样本），跳过")
            num_skip_not_in_cache += 1
            continue

        # 4) 读取 HSI 立方 (480, 640, 100)
        cube = path.read_cube().astype(np.float32)  # (H, W, C)

        # 5) 读取 segmentation（多类，uint8）
        mask = path.read_segmentation(f"polygon#{ANNOTATOR}")
        if mask is None:
            print(f"  -> 没有 {ANNOTATOR} 的分割，跳过")
            num_skip_no_seg += 1
            continue

        mask = np.asarray(mask, dtype=np.uint8)  # (H, W)

        # ---- 6) 用 PyTorch 做 resize ----
        # cube: (H, W, C) -> (1, C, H, W)
        cube_t = torch.from_numpy(cube).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        # 双线性插值，适合连续值
        cube_resized = F.interpolate(
            cube_t,
            size=(TARGET_H, TARGET_W),
            mode="bilinear",
            align_corners=False,
        )  # [1, C, 256, 256]
        cube_resized = cube_resized.squeeze(0).cpu().numpy()  # [C, 256, 256]

        # mask: (H, W) -> (1,1,H,W)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
        # 最近邻插值，保证类别 id 不被插值成小数
        mask_resized = F.interpolate(
            mask_t,
            size=(TARGET_H, TARGET_W),
            mode="nearest",
        )
        mask_resized = (
            mask_resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        )  # [256,256]

        # ---- 7) 保存成 .npy ----
        # 这里 HSI 保存为 (C, H, W)，方便后面直接 torch.from_numpy 变成 [C,H,W]
        hsi_path = hsi_out / f"{image_name}.npy"
        mask_path = mask_out / f"{image_name}.npy"

        np.save(hsi_path, cube_resized)
        np.save(mask_path, mask_resized)

        num_ok += 1

    print("\n==== 完成 resize ====")
    print("成功 resize 的图像数:", num_ok)
    print("因无分割而跳过的图像数:", num_skip_no_seg)
    print("因不在 DataPath 缓存而跳过的 L1 文件数:", num_skip_not_in_cache)
    print("输出目录:", out_root)


if __name__ == "__main__":
    main()
