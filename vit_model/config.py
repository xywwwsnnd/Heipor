# vit_model/config.py
# ------------------------------------------------------------
from vit_model.vit_seg_configs import CONFIGS as VIT_CONFIGS
import torch
from copy import deepcopy as _dc
import os

# ============ ① 控制 HSI 分支模式（与原来一致） ============ #
USE_COMPLEX = True   # ➜ True ⇒ 走 dual / complex 路径
FULL_COMPLEX = False # ➜ False ⇒ dual；True 时才会全复数
# ========================================================== #

# ---- 安全的 AttrDict：支持 getattr / setattr / deepcopy ---- #
class AttrDict(dict):
    def __getattr__(self, k):
        if k not in self:
            raise AttributeError(k)
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        if k in self:
            del self[k]
        else:
            raise AttributeError(k)

    def __deepcopy__(self, memo):
        return AttrDict(_dc(dict(self), memo))

# ========================================================== #
# ---- 全局配置类 (Config) ---- #
# 注意：Config 必须定义在 AttrDict 之外
class Config:
    # ---------- 数据路径 ----------
    data_dir = "/mnt/nvme1n1/bitmhsi/dataset/HeiPor_resized_256"

    # 这里会真正被 Dataset 用到
    hyperspectral_dir = f"{data_dir}/hsi"  # 你的 HSI .npy
    rgb_dir = ""                           # 不用就空着
    label_dir = f"{data_dir}/mask"         # 你的 mask .npy

    train_list = f"{data_dir}/train.txt"   # 每行：P086#2021_04_15_09_22_02
    test_list = f"{data_dir}/test.txt"

    # ---------- 数据形状 ----------
    num_channels_hsi = 100
    num_channels_rgb = 3
    image_size = 256
    num_classes = 21

    # ---------- 训练参数（基础） ----------
    batch_size = 16
    num_epochs = 300
    learning_rate = 5e-4
    initial_lr = 2e-4
    num_workers = 32          # 若排查崩溃，可临时改为 0
    device = "cuda:3" if torch.cuda.is_available() else "cpu"

    # BCE 正负样本不均衡
    bce_pos_weight = torch.tensor([1.00])

    # ---------- 训练日志/评估频率 ----------
    print_loss_every = 20    # 训练中每隔多少 iter 打印 band selector 指标
    validation_freq  = 50    # 每多少个 epoch 存一次完整断点
    skip_metric      = 100   # 前若干个 epoch 不跑评估（老师口径）

    # ---------- 随机种子 ----------
    seed = 42

    # ---------- 训练策略增强/学习率 ----------
    warmup_epochs   = 10
    lr_max          = learning_rate
    lr_start_factor = 0.1
    min_lr          = 1e-6
    accumulation_steps = 4

    # ---------- Dice 动态权重（如需，可在 compute_total_loss 里使用） ----------
    dice_weight_start  = 0.05
    dice_weight_target = 0.15
    dice_warmup_ratio  = 0.10
    dice_ramp_ratio    = 0.30

    # EMA（如需在训练里启用）
    use_ema   = True
    per_sample_minmax = True
    ema_decay = 0.999

    # ---------- “间歇精修 band selector”训练超参（新增） ----------
    # 只更新 band selector，使用更“为 Dice 负责”的分割损失：bce_w*BCE + dice_w*SoftDice
    refine_selector_enable      = True   # 开/关
    refine_selector_every       = 1      # 主优化器每 step 触发一次精修；可改为 2/4 降低开销
    refine_selector_steps       = 3      # 每次触发做几步精修
    refine_selector_lr_mult     = 0.5    # 精修学习率 = base_lr * 该倍率
    refine_selector_bce_weight  = 1.0    # 精修时 BCE 权重
    refine_selector_dice_weight = 1.0    # 精修时 Dice 权重（↑）

    # ---------- 断点恢复 ----------
    checkpoint_dir = "/home/bitmhsi/HeiPor/checkpoints"
    log_dir = "./logs"
    resume = False
    resume_ckpt = os.path.join(checkpoint_dir, "model_ep100.pth")

    # ---------- 模型配置占位符 ----------
    model_config = None


# ========================================================== #
# ---- 初始化模型详细配置 (在类定义之外执行) ---- #
# ========================================================== #

# 1. 加载基础 ViT 配置
_base_config = _dc(VIT_CONFIGS['R50-ViT-B_16'])
Config.model_config = AttrDict(_base_config)

# 2. 应用自定义修改
Config.model_config.hidden_size = 256
Config.model_config.in_channels = Config.num_channels_rgb
Config.model_config.in_channels_hsi = Config.num_channels_hsi

# Transformer 配置
Config.model_config.transformer.update({
    "num_heads": 8,
    "num_layers": 6,
    "mlp_dim": 1024,
    "attention_dropout_rate": 0.0,
    "dropout_rate": 0.1,
})

# Patch / Grid
Config.model_config.patches.update({
    "size": (16, 16),
    "grid": (16, 16),
})

# ResNet 主干配置
Config.model_config.resnet = {
    "num_layers": [3, 4, 6, 3],
    "width_factor": 1.0,
}

# Decoder / 旁路
Config.model_config.decoder_channels = [64, 32, 16, 8]
Config.model_config.n_classes = Config.num_classes
Config.model_config.n_skip = 4
Config.model_config.skip_channels = [256, 256, 256, 128]
Config.model_config.use_ssfenet = True
Config.model_config.classifier = "seg"
Config.model_config.activation = "softmax"
Config.model_config.upsampling = 32

# 显式开关
Config.model_config.use_hybrid_resnet = True
Config.model_config.use_medsam2_encoder = False

# TransPath RGB encoder
Config.model_config.use_transpath_rgb = True
Config.model_config.transpath_variant = 'ctranspath'
Config.model_config.transpath_hidden = 96
Config.model_config.transpath_ckpt = "checkpoints/ctranspath.pth"


# 浅层 SE 融合保持开启
Config.model_config.use_shallow_fusion = True

# 逐层融合
Config.model_config.use_stagewise_hsi_fusion = False

# HSI→伪RGB 波段选择
Config.model_config.use_pseudo_rgb = True
Config.model_config.pseudo_rgb_hidden = 128
Config.model_config.pseudo_rgb_temperature = 1.0
Config.model_config.pseudo_rgb_lambda_peak = 0.02
Config.model_config.pseudo_rgb_lambda_div = 0.05

# 交叉注意力（编码端）
Config.model_config.fusion_policy = "bi_xattn_se_residual"
Config.model_config.cross_attn_symmetric = True
Config.model_config.cross_attn_heads = 4
Config.model_config.cross_attn_dim_head = 32
Config.model_config.cross_attn_dropout = 0.1
Config.model_config.alpha_init = 0.1
Config.model_config.beta_init = 0.2

# ---------- FRFT/频域相关超参（如需启用请取消注释） ----------
# Config.model_config.frft = AttrDict({
#     "alpha_init": 0.95,
#     "alpha_center": 1.00,
#     "alpha_lr_mult": 0.02,
#     "alpha_weight_decay": 0.0,
#     "alpha_shared": True,
#     "alpha_domain": "(0.90,1.10)",
#     "alpha_prior_lambda": 1e-3,
#     "alpha_freeze_epochs": 8,
#     "alpha_tiny_lr_mult": 0.01,
#     "fft_norm": "backward",
#     "rms_align": True,
#     "frfdconv_use_polar": True,
#     "polar_init": (0.95, 0.05),
#     "polar_w_lr_mult": 0.5,
#     "enable_ksm": True,
#     "enable_fbm": True,
#     "ksm_lr_mult": 0.5,
#     "fbm_lr_mult": 0.5,
#     "clip_input_rms": 80000000000000.0,
#     "clip_frft_map_rms": 80000000000000.0,
#     "clip_adaptive_spatial_rms": 80000000000000.0,
#     "clip_fbm_rms": 8000000000000.0,
#     "clip_print": True,
#     "frfdconv_scale_mult": 1e-3,
#     "fbm_global_gain": 3e-3,
# })

# ---------- SSFENet 串行级联 ----------
# Config.model_config.ssfenet_passes = 1
# Config.model_config.ssfenet_strides = (1,)
# Config.model_config.ssfenet_out_ch = 64
# Config.model_config.ssfenet_shared_alpha = True

# ---------- 末端长残差 ----------
# Config.model_config.use_long_skip = True
# Config.model_config.long_skip_beta_init = 0.95
# Config.model_config.long_skip_rms_norm = True