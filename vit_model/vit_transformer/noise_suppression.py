import torch
import torch.nn as nn
from vit_model.layers.frft_fdconv import FrFBM
from vit_model.config import Config

class SpectralNoiseSuppression(nn.Module):
    def __init__(self, in_channels):
        super(SpectralNoiseSuppression, self).__init__()
        # 读取 FRFT 配置（若无则用默认 0.5）
        frft_cfg   = getattr(Config.model_config, "frft", None)
        alpha_init = 0.5 if frft_cfg is None else float(getattr(frft_cfg, "alpha_init", 0.5))

        # 可学习 α（会被你的参数分组按 name.endswith("alpha") 捕获）
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # FrFBM：极坐标 + 自动 1×1 reduce，内部是残差形式（x + y）
        self.frfbm = FrFBM(in_channels, n_bands=4, alpha=self.alpha)

    def forward(self, x):
        # FrFBM 已经返回 x + y，这里直接用（残差语义）
        return self.frfbm(x)
