# vit_model/vit_transformer/transformer.py
import torch.nn as nn
from .embeddings import Embeddings
from .encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis=False):
        super().__init__()
        # [修改] 显式传入 config 中的通道数，否则会使用 embeddings.py 里的默认值(60)
        self.embeddings = Embeddings(
            config,
            img_size=img_size,
            in_channels_rgb=config.in_channels,     # 通常为 3
            in_channels_hsi=config.in_channels_hsi  # 这里是你 config 设定的 100
        )
        self.encoder = Encoder(config, vis)

    def forward(self, rgb, hsi):  # 修改参数名为 rgb, hsi 以明确输入
        x_rgb, x_hsi, features, reg_loss = self.embeddings(rgb, hsi)
        x_rgb, x_hsi, attn_weights = self.encoder(x_rgb, x_hsi)
        return x_rgb, x_hsi, features, attn_weights, reg_loss