# draw_model_diagram.py
"""
生成模型示意图的脚本。
使用 matplotlib 绘制包含波段选择、TransPath 分支、FuseResNetV2 分支、
各阶段 SE+CrossAttn 融合、Token 化、Transformer 编码器、UNet 解码器和
分割头的流程图。
生成的图像保存为 model_diagram.png。
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, text, fontsize=10, edgecolor="black", facecolor="#F0F8FF"):
    """绘制矩形并在中心写入文字的工具函数。"""
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=1.0,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )
    return rect

def draw_diagram():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    # 绘制输入和波段选择器
    hsi_box = draw_box(ax, 0.0, 5.0, 2.5, 0.8, "HSI input\n(C=60)")
    band_box = draw_box(ax, 2.8, 5.0, 3.0, 0.8, "Band Selector\n→ Pseudo RGB (3ch)")
    ax.annotate("", xy=(2.8, 5.4), xytext=(2.5, 5.4), arrowprops=dict(arrowstyle="->", lw=1))

    # TransPath 分支
    tp_box = draw_box(ax, 6.2, 5.7, 4.0, 1.4,
        "TransPath RGB encoder\nConvStem (3×3→BN→ReLU) ×2\nPatch embed + Transformer (12 layers)",
        fontsize=9
    )
    ax.annotate("", xy=(6.2, 6.7), xytext=(5.8, 5.8), arrowprops=dict(arrowstyle="->", lw=1))

    # FuseResNet 分支根部
    frn_y = 3.0
    root_box = draw_box(ax, 2.8, frn_y, 3.5, 0.8,
        "FuseResNetV2 root\nConv7×7 s4→GN→ReLU"
    )
    ax.annotate("", xy=(2.8, frn_y + 0.4), xytext=(5.8, 5.0), arrowprops=dict(arrowstyle="->", lw=1))

    # FuseResNet 各阶段
    stage1_box = draw_box(ax, 6.6, frn_y, 3.8, 0.8,
        "Stage1 (3× BasicBlock)\nConv3×3→GN→ReLU→Conv3×3→GN\n+ skip identity",
        fontsize=8
    )
    ax.annotate("", xy=(6.6, frn_y + 0.4), xytext=(6.3, frn_y + 0.4), arrowprops=dict(arrowstyle="->", lw=1))

    stage2_box = draw_box(ax, 10.6, frn_y, 3.8, 0.8,
        "Stage2 (4× BasicBlock)\n1st: stride2 conv\nrest: conv3×3→GN→ReLU→conv3×3→GN",
        fontsize=8
    )
    ax.annotate("", xy=(10.6, frn_y + 0.4), xytext=(10.4, frn_y + 0.4), arrowprops=dict(arrowstyle="->", lw=1))

    stage3_box = draw_box(ax, 14.6, frn_y, 3.8, 0.8,
        "Stage3 (6× BasicBlock)\n1st: stride2 conv\n5× conv3×3→GN→ReLU→conv3×3→GN",
        fontsize=8
    )
    ax.annotate("", xy=(14.6, frn_y + 0.4), xytext=(14.4, frn_y + 0.4), arrowprops=dict(arrowstyle="->", lw=1))

    stage4_box = draw_box(ax, 18.6, frn_y, 3.8, 0.8,
        "Stage4 (3× BasicBlock)\nConv3×3→GN→ReLU→Conv3×3→GN",
        fontsize=8
    )
    ax.annotate("", xy=(18.6, frn_y + 0.4), xytext=(18.4, frn_y + 0.4), arrowprops=dict(arrowstyle="->", lw=1))

    # 跳跃连接
    skip1_box = draw_box(ax, 7.0, frn_y - 1.2, 2.4, 0.6, "skip1\n1×1 conv→64ch", fontsize=8)
    skip2_box = draw_box(ax, 11.0, frn_y - 1.2, 2.4, 0.6, "skip2\n1×1 conv→128ch", fontsize=8)
    skip3_box = draw_box(ax, 15.0, frn_y - 1.2, 2.4, 0.6, "skip3\n1×1 conv→256ch", fontsize=8)
    ax.annotate("", xy=(7.0, frn_y - 0.6), xytext=(8.4, frn_y + 0.0), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(11.0, frn_y - 0.6), xytext=(12.4, frn_y + 0.0), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(15.0, frn_y - 0.6), xytext=(16.4, frn_y + 0.0), arrowprops=dict(arrowstyle="->", lw=1))

    # FuseResNet 顶部输出
    top_box = draw_box(ax, 22.6, frn_y, 3.8, 0.8, "fused4/top\n256ch (H/16×W/16)")
    ax.annotate("", xy=(22.6, frn_y + 0.4), xytext=(22.4, frn_y + 0.4), arrowprops=dict(arrowstyle="->", lw=1))

    # 跨模态融合
    fusion_box = draw_box(ax, 26.4, 4.4, 4.5, 2.0,
        "Fusion (per stage)\nCrossAttn2D + SE\nStages: top, s3, s2, s1",
        fontsize=9
    )
    ax.annotate("", xy=(26.4, 5.4), xytext=(10.2, 6.4), arrowprops=dict(arrowstyle="->", lw=1))  # tp_top→fusion
    ax.annotate("", xy=(26.4, 4.8), xytext=(22.6, frn_y + 0.4), arrowprops=dict(arrowstyle="->", lw=1))  # fused top→fusion

    # Token 化
    token_box = draw_box(ax, 31.4, 5.0, 3.5, 1.0, "Tokenization\nAdaptive pool→16×16→Flatten")
    ax.annotate("", xy=(31.4, 5.5), xytext=(30.9, 5.5), arrowprops=dict(arrowstyle="->", lw=1))

    # Transformer 编码器
    trans_box = draw_box(ax, 35.5, 5.0, 3.5, 1.0, "Transformer\nEncoder (6 layers)")
    ax.annotate("", xy=(35.5, 5.5), xytext=(34.9, 5.5), arrowprops=dict(arrowstyle="->", lw=1))

    # UNet 解码器
    dec_box = draw_box(ax, 39.6, 5.0, 4.5, 1.5,
        "DecoderCup (UNet)\n4 blocks: up×2 + concat skip\nChannels: 64→32→16→8",
        fontsize=9
    )
    ax.annotate("", xy=(39.6, 5.5), xytext=(39.0, 5.5), arrowprops=dict(arrowstyle="->", lw=1))

    # 跳跃连接流向 Decoder
    ax.annotate("", xy=(39.8, 6.0), xytext=(15.0 + 1.2, frn_y - 1.2 + 0.3), arrowprops=dict(arrowstyle="->", lw=1))  # skip3
    ax.annotate("", xy=(39.8, 5.8), xytext=(11.0 + 1.2, frn_y - 1.2 + 0.3), arrowprops=dict(arrowstyle="->", lw=1))  # skip2
    ax.annotate("", xy=(39.8, 5.6), xytext=(7.0 + 1.2, frn_y - 1.2 + 0.3), arrowprops=dict(arrowstyle="->", lw=1))   # skip1
    ax.annotate("", xy=(39.8, 5.4), xytext=(22.6 + 1.9, frn_y + 0.4), arrowprops=dict(arrowstyle="->", lw=1))        # fused top

    # 分割头及输出
    seg_box = draw_box(ax, 45.2, 5.2, 3.8, 0.8, "Segmentation Head\nConv3×3→BN→ReLU→Conv1×1")
    ax.annotate("", xy=(45.2, 5.6), xytext=(44.1, 5.6), arrowprops=dict(arrowstyle="->", lw=1))
    out_box = draw_box(ax, 49.6, 5.2, 2.6, 0.8, "Segmentation\nOutput (1ch)")
    ax.annotate("", xy=(49.6, 5.6), xytext=(49.0, 5.6), arrowprops=dict(arrowstyle="->", lw=1))

    # 设置视图范围
    ax.set_xlim(0, 54)
    ax.set_ylim(0, 7)
    ax.set_title("Hyperspectral Segmentation Model Architecture", fontsize=14, pad=20)

    fig.tight_layout()
    fig.savefig("model_diagram.png", dpi=300)

if __name__ == "__main__":
    draw_diagram()
