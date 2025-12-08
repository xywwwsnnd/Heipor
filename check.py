# -*- coding: utf-8 -*-
"""
统计每一张标签图中，各个类别像素占比（含 0/254/255），并给出全局统计。

使用方法：
    python stat_class_ratio.py

可根据需要修改 data_root 路径。
"""

from pathlib import Path
import numpy as np


def load_list(list_path: Path):
    if not list_path.exists():
        return None
    with open(list_path, "r", encoding="utf-8") as f:
        stems = [ln.strip() for ln in f if ln.strip()]
    return stems


def main():
    # ====== 1. 数据集根目录 & 子目录 ======
    data_root = Path("/mnt/nvme1n1/bitmhsi/dataset/HeiPor_resized_256")
    mask_dir = data_root / "mask"

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"mask 目录不存在: {mask_dir}")

    # 可选：读取 train / test 列表，方便你后面按猪划分/按训练集划分统计
    train_stems = load_list(data_root / "train.txt")
    test_stems  = load_list(data_root / "test.txt")

    print("train.txt 图像数:", len(train_stems) if train_stems is not None else "N/A")
    print("test.txt  图像数:", len(test_stems)  if test_stems  is not None else "N/A")

    # ====== 2. 遍历所有 mask.npy，统计每张图的类别占比 ======
    mask_files = sorted(mask_dir.glob("*.npy"))
    if not mask_files:
        raise RuntimeError(f"在 {mask_dir} 下没有找到 .npy 掩码文件")

    print(f"共找到 {len(mask_files)} 张 mask 图像\n")

    # 全局统计：每个类的总像素数
    global_counts = {}

    for mask_path in mask_files:
        stem = mask_path.stem  # e.g. P086#2021_04_15_09_22_02

        mask = np.load(mask_path)
        if mask.ndim > 2:
            # 如果不小心变成 (1,H,W) 或 (H,W,1)，压成 (H,W)
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"{mask_path} 维度不是 2D，实际 shape={mask.shape}")

        H, W = mask.shape
        total_pixels = H * W

        # 当前图的直方图
        vals, cnts = np.unique(mask, return_counts=True)

        # 更新全局统计
        for v, c in zip(vals, cnts):
            v = int(v)
            global_counts[v] = global_counts.get(v, 0) + int(c)

        # ====== 2.1 打印当前图的类别占比 ======
        print(f"Image: {stem}  (H={H}, W={W}, total={total_pixels})")

        # 对类排序一下，0 在最前面，255、254 这些也会显示出来
        for v, c in sorted(zip(vals, cnts), key=lambda x: int(x[0])):
            v = int(v)
            ratio = c / float(total_pixels)
            if v == 0:
                name = "背景/未标注(含 0,254,255 映射后)"
            elif v == 254:
                name = "特殊重叠区(254)"
            elif v == 255:
                name = "未标注区(255)"
            else:
                name = f"类别 {v}"
            print(f"  类 {v:3d} ({name}): {int(c):7d} 像素, {ratio*100:6.3f} %")

        print("-" * 60)

    # ====== 3. 输出全局统计（整个数据集所有像素的占比） ======
    print("\n================= 全局类别统计（所有图像合计） =================")
    total_all = sum(global_counts.values())
    print(f"总像素数（所有 mask 合计）: {total_all}")

    # 按类别 ID 排序打印
    for v in sorted(global_counts.keys()):
        c = global_counts[v]
        ratio = c / float(total_all)
        if v == 0:
            name = "背景/未标注(含 0,254,255 映射后)"
        elif v == 254:
            name = "特殊重叠区(254)"
        elif v == 255:
            name = "未标注区(255)"
        else:
            name = f"类别 {v}"
        print(f"类 {v:3d} ({name}): {c:10d} 像素, {ratio*100:6.3f} %")


if __name__ == "__main__":
    main()
