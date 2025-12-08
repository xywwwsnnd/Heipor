import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import HyperspectralDataset
from config import Config


def check_dataset():
    print("=== 开始数据检查 ===")

    # 1. 强制覆盖 Config 以便调试
    # 请确保这里的路径是你真实的 .npy 路径
    hsi_dir = Config.hyperspectral_dir
    label_dir = Config.label_dir
    train_list = Config.train_list

    ds = HyperspectralDataset(
        hsi_dir=hsi_dir,
        rgb_dir="",  # 假设不用单独的RGB文件
        label_dir=label_dir,
        list_file=train_list,
        image_size=256,
        train=False,  # 关闭增强，看原始数据
        num_channels_hsi=100,
        multi_class=True,
        per_sample_minmax=False  # 先检查原始值
    )

    print(f"数据集长度: {len(ds)}")

    # 取一个样本
    idx = 0
    hsi, rgb, lab, lab_orig = ds[idx]

    print(f"\n--- 样本 {idx} 统计 ---")
    print(f"HSI Shape: {hsi.shape} (Expect: [100, 256, 256])")
    print(f"HSI Range: Min={hsi.min():.4f}, Max={hsi.max():.4f}, Mean={hsi.mean():.4f}")

    print(f"Label Shape: {lab.shape}")
    unique_labs = torch.unique(lab)
    print(f"Label Unique Values: {unique_labs.tolist()}")

    if unique_labs.max() >= Config.num_classes:
        print(f"⚠️ 警告: 标签最大值 {unique_labs.max()} >= num_classes {Config.num_classes}")
        print("   如果不做重映射，train.py 会把这些值变为 0 (背景)！")
    else:
        print("✅ 标签值在范围内。")

    if hsi.max() > 10.0:
        print("⚠️ 警告: HSI 数值过大，建议开启 per_sample_minmax 或预处理归一化。")

    # 可视化检查维度是否错乱
    # 取 HSI 第 50 个波段看是否像图
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("HSI Band 50")
    plt.imshow(hsi[50].cpu().numpy(), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("RGB (Normalized)")
    # 反归一化以便显示
    rgb_disp = rgb.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    plt.imshow(np.clip(rgb_disp, 0, 1))

    plt.subplot(1, 3, 3)
    plt.title("Label")
    plt.imshow(lab.cpu().numpy(), cmap='jet', interpolation='nearest')

    plt.savefig("check_data_vis.png")
    print("\n可视化已保存为 check_data_vis.png，请查看图像是否正常。")


if __name__ == "__main__":
    check_dataset()