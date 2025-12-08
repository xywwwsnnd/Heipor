# -*- coding: utf-8 -*-
"""
make_rgb_icons_fixed.py
从一张伪RGB图生成三张用于论文结构图的小图（R/G/B）：
- 固定输出尺寸：256x256
- 方形（非圆形），透明背景，彩色描边，角标 R/G/B
- 固定路径：
    输入： G:\HSI\伪RGB\030406-20x-roi1_patch_0.png
    输出： G:\HSI\tu
依赖：pip install pillow numpy
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops

# ======== 固定参数 ========
INPUT_PATH = r"G:\HSI\伪RGB\030406-20x-roi1_patch_0.png"
OUT_DIR    = r"G:\HSI\tu"
SIZE       = 256         # 输出尺寸 256x256
BORDER_PX  = 6           # 描边像素（方形边框）
SHAPE      = "square"    # ★ 方形（非圆形）
COL_R = (230, 40, 40)    # R/G/B 颜色
COL_G = (40, 190, 70)
COL_B = (40, 150, 230)
FONT_PATH = None         # 可选：r"C:\Windows\Fonts\arialbd.ttf"
# ==========================

def robust_rescale01(arr, p_low=2.0, p_high=98.0):
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip((arr - lo) / max(hi - lo, 1e-8), 0, 1)
    return arr

def read_rgb(path):
    img = Image.open(path).convert("RGB")  # 确保3通道
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.clip(arr, 0, 1)  # [H,W,3], 0~1

def channel_to_icon(ch, *, size=256, color=(255,0,0),
                    shape="square", border_px=6, label=None, font_path=None):
    # 归一化 + 鲁棒拉伸
    ch = robust_rescale01(ch)

    # 居中贴到方形画布再统一缩放
    H, W = ch.shape
    s = max(H, W)
    canvas = np.zeros((s, s), dtype=np.float32)
    y0 = (s - H) // 2
    x0 = (s - W) // 2
    canvas[y0:y0+H, x0:x0+W] = ch
    canvas_img = Image.fromarray((canvas * 255).astype(np.uint8), mode="L").resize((size, size), Image.BICUBIC)

    # 染色
    tint = Image.new("RGB", (size, size), color=color)
    rgb = Image.composite(tint, Image.new("RGB", (size, size), (0,0,0)), mask=canvas_img)
    rgba = rgb.convert("RGBA")

    # 形状 alpha（方形）
    alpha_shape = Image.new("L", (size, size), 0)
    draw_a = ImageDraw.Draw(alpha_shape)
    pad = max(border_px//2, 2)
    rect = [pad, pad, size - pad - 1, size - pad - 1]
    # 方形填充
    draw_a.rectangle(rect, fill=255)

    # 纹理 alpha 与形状相乘，保留数据质感
    alpha = ImageChops.multiply(alpha_shape, canvas_img)
    rgba.putalpha(alpha)

    # 方形描边
    draw = ImageDraw.Draw(rgba)
    for t in range(border_px):
        rr = [pad + t, pad + t, size - pad - 1 - t, size - pad - 1 - t]
        draw.rectangle(rr, outline=color + (210,), width=1)

    # 角标
    if label:
        try:
            font = ImageFont.truetype(font_path or "DejaVuSans-Bold.ttf", size=max(22, size//10))
        except Exception:
            font = ImageFont.load_default()
        tw = draw.textlength(label, font=font)
        th = font.size
        m = max(10, size//30)
        pos = (int(size - tw - m), int(size - th - m))
        bg = Image.new("RGBA", (int(tw)+12, th+10), (0,0,0,120))
        rgba.alpha_composite(bg, (pos[0]-6, pos[1]-5))
        draw.text(pos, label, fill=(255,255,255,230), font=font)

    # 最终保证尺寸为 256x256
    if rgba.size != (size, size):
        rgba = rgba.resize((size, size), Image.BICUBIC)
    return rgba

def main():
    inp = Path(INPUT_PATH)
    outd = Path(OUT_DIR)
    outd.mkdir(parents=True, exist_ok=True)

    arr = read_rgb(inp)       # [H,W,3] float
    r, g, b = arr[...,0], arr[...,1], arr[...,2]

    icon_R = channel_to_icon(r, size=SIZE, color=COL_R, shape=SHAPE, border_px=BORDER_PX, label="R", font_path=FONT_PATH)
    icon_G = channel_to_icon(g, size=SIZE, color=COL_G, shape=SHAPE, border_px=BORDER_PX, label="G", font_path=FONT_PATH)
    icon_B = channel_to_icon(b, size=SIZE, color=COL_B, shape=SHAPE, border_px=BORDER_PX, label="B", font_path=FONT_PATH)

    stem = inp.stem
    pR = outd / f"{stem}_R_icon.png"
    pG = outd / f"{stem}_G_icon.png"
    pB = outd / f"{stem}_B_icon.png"

    icon_R.save(pR)
    icon_G.save(pG)
    icon_B.save(pB)

    print("[OK] 已导出(256x256, 方形)：")
    print(" ", pR)
    print(" ", pG)
    print(" ", pB)

if __name__ == "__main__":
    main()
