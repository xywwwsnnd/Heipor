# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from vit_model.vit_transformer.transformer import Transformer
from vit_model.decoder.decoder_cup import DecoderCup
from vit_model.decoder.segmentation_head import SegmentationHead


__all__ = ["VisionTransformer"]


class VisionTransformer(nn.Module):
    """
    Clean VisionTransformer (config-first)
    - 以 config 的 image_size / n_classes 为准（外部传参可覆盖）
    - 解码器期望 4 个 skip: [top, s3, s2, s1]
    - n_skip = 4 时，Decoder 输出已是 256×256，SegHead 不再上采样 (upsampling=1)
    - 不包含任何 MedSAM2 相关逻辑

    forward 返回:
        logits, reg_loss, attn_weights, features
    """
    def __init__(self, config, img_size=None, num_classes=None, zero_head=False, vis=False):
        super().__init__()
        # === 以 config 为真值；允许外部覆盖 ===
        self.config = config
        self.img_size = int(img_size) if img_size is not None else int(getattr(config, "image_size", 256))
        self.num_classes = int(num_classes) if num_classes is not None else int(getattr(config, "n_classes", 1))

        self.zero_head = bool(zero_head)
        self.classifier = getattr(config, "classifier", "seg")

        # === Transformer（内含 embeddings / encoder）===
        self.transformer = Transformer(config, self.img_size, vis)

        # === Decoder ===
        # 与 config.decoder_channels / skip_channels / n_skip 对齐
        # 你的配置为：n_skip=4，skip_channels=[256,256,256,128]
        self.decoder = DecoderCup(config)

        # === Segmentation Head ===
        # n_skip=4 -> 解码输出空间已回到 256×256，与标签对齐，故 upsampling=1
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=self.num_classes,
            kernel_size=3,
            upsampling=1,          # 重要：不要再放大
            with_bn=True,
            activation=None        # 训练用 BCEWithLogitsLoss 时必须为 None
        )

    def forward(self, x_rgb: torch.Tensor, x_hsi: torch.Tensor):
        # 1) Embeddings：返回 token & 多尺度特征
        #    x_rgb_tok, x_hsi_tok: [B, 256, C]
        #    features: [top, s3, s2, s1]（用于 skip）
        #    reg_loss: dict（选带/正则项）
        x_rgb_tok, x_hsi_tok, features, reg_loss = self.transformer.embeddings(x_rgb, x_hsi)

        # 2) Encoder：编码两个 token（内部可能含 cross-attn），返回 4D map
        enc_rgb, enc_hsi, attn_weights = self.transformer.encoder(x_rgb_tok, x_hsi_tok)

        # 3) 融合：两个路径的 4D map 按元素相加（与当前工程一致）
        fused_features = enc_rgb + enc_hsi

        # 4) Decoder：显式传入四个 skip；需与 config.skip_channels 一致
        assert isinstance(features, (list, tuple)) and len(features) == 4, \
            f"[VisionTransformer] 期望 4 个 skip [top,s3,s2,s1]，当前为 {0 if features is None else len(features)}"
        decoded = self.decoder(fused_features, features=features)

        # 5) Seg head：upsampling=1，输出与标签 (256×256) 对齐
        logits = self.segmentation_head(decoded)

        return logits, reg_loss, attn_weights, features
