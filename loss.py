# === loss.py (Final Correct Version) ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

__all__ = [
    "DiceLoss", "BCEDiceLoss",
    "MultiClassDiceLoss", "CEDiceLoss"
]


# ----------------------- 二类工具 ----------------------- #
def _sanitize_inputs_binary(logits: torch.Tensor, targets: torch.Tensor, ignore_index: Optional[int]):
    """
    - 清理 logits 的 NaN/Inf
    - 生成 valid 掩膜（优先处理 ignore_index）
    - 将 targets 规范到 {0,1}（兼容 0/255 或浮点）
    """
    if not torch.isfinite(logits).all():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.to(logits.dtype)

    # [关键逻辑] 1. 优先处理 ignore_index
    # 必须先生成掩膜并把忽略区域置0，否则 255 会干扰下面的 min/max 计算
    if ignore_index is not None:
        # 必须确保比较是准确的（转为 long 再比），避免 float 精度问题
        valid = (targets.long() != int(ignore_index)).to(logits.dtype)
        targets = targets * valid  # 把忽略区域的值先洗成 0
    else:
        valid = torch.ones_like(targets)

    # [关键逻辑] 2. 然后再进行数值范围检查和二值化
    # 此时 255 已经被上面的代码置 0 了，所以 max 应该是 1 (如果只有0和1) 或者其他正数
    t_min, t_max = targets.min().item(), targets.max().item()
    if t_max > 1.0 or t_min < 0.0:
        thr = 0.5 * max(t_max, 1.0)  # 兼容 0/255
        targets = (targets > thr).to(logits.dtype)

    return logits, targets, valid


class DiceLoss(nn.Module):
    """Binary soft Dice over N,H,W (对 logits 做 sigmoid)"""

    def __init__(self, smooth: float = 1e-6, ignore_index: Optional[int] = None, skip_empty: bool = False):
        super().__init__()
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.skip_empty = bool(skip_empty)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets, valid = _sanitize_inputs_binary(logits, targets, self.ignore_index)
        if valid.sum() == 0:
            return logits.new_tensor(0.0)

        probs = torch.sigmoid(logits)
        probs = (probs * valid).flatten(1)
        targets = (targets * valid).flatten(1)

        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)

        if self.skip_empty:
            has_fg = (targets.sum(dim=1) > 0)
            if has_fg.any():
                dice = dice[has_fg]
            else:
                return logits.new_tensor(0.0)

        dice = dice.clamp(0.0, 1.0)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Binary: BCEWithLogits + λ * Dice"""

    def __init__(self, pos_weight=None, dice_weight: float = 0.5,
                 smooth: float = 1e-6, ignore_index: Optional[int] = None,
                 skip_empty: bool = False):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.dice = DiceLoss(smooth=smooth, ignore_index=ignore_index, skip_empty=skip_empty)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, t_eff, valid = _sanitize_inputs_binary(logits, targets, self.ignore_index)
        if valid.sum() == 0:
            bce_loss = logits.new_tensor(0.0)
        else:
            bce_map = self.bce(logits, t_eff) * valid
            bce_loss = bce_map.sum() / (valid.sum() + 1e-6)
        dice_loss = self.dice(logits, targets)
        total = bce_loss + self.dice_weight * dice_loss
        if not torch.isfinite(total):
            mx = torch.nan_to_num(logits.detach()).abs().max().item()
            total = torch.nan_to_num(total)
        return total


# ----------------------- 多类工具 (保持原样) ----------------------- #
def _valid_mask_mc(targets: torch.Tensor, ignore_index: Optional[int]) -> torch.Tensor:
    """
    多类有效掩膜：targets [B,H,W] (long)，返回 [B,1,H,W] ∈ {0,1}
    """
    if ignore_index is None:
        return torch.ones((targets.shape[0], 1, targets.shape[1], targets.shape[2]),
                          dtype=torch.float32, device=targets.device)
    else:
        return (targets != int(ignore_index)).unsqueeze(1).to(torch.float32)


def _one_hot_mc(targets: torch.Tensor, num_classes: int, ignore_index: Optional[int]) -> torch.Tensor:
    """
    将 [B,H,W] (long) → [B,C,H,W] (float)，忽略像素位置置零
    """
    B, H, W = targets.shape
    C = int(num_classes)
    oh = torch.zeros((B, C, H, W), dtype=torch.float32, device=targets.device)
    if ignore_index is not None:
        valid = (targets != int(ignore_index))
        # 将 ignore_index 处的值暂时置 0，避免 one_hot 抛错，然后再用 valid 掩膜
        t = targets.clone()
        t[~valid] = 0
        oh.scatter_(1, t.unsqueeze(1), 1.0)
        oh = oh * valid.unsqueeze(1).to(torch.float32)
    else:
        oh.scatter_(1, targets.unsqueeze(1), 1.0)
    return oh


class MultiClassDiceLoss(nn.Module):
    """
    多类 Dice（对 logits 做 softmax），默认忽略背景0类。
    """

    def __init__(self,
                 num_classes: int,
                 smooth: float = 1e-6,
                 ignore_index: Optional[int] = None,
                 include_background: bool = False,
                 class_weights: Optional[Sequence[float]] = None,
                 skip_empty: bool = True):
        super().__init__()
        self.C = int(num_classes)
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.include_bg = bool(include_background)
        self.skip_empty = bool(skip_empty)
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            assert w.numel() == self.C, "class_weights 长度必须等于 num_classes"
            self.register_buffer("class_weights", w, persistent=False)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.dim() == 4 and logits.shape[1] == self.C
        assert targets.dim() == 3

        probs = F.softmax(logits, dim=1)  # [B,C,H,W]
        valid = _valid_mask_mc(targets, self.ignore_index)  # [B,1,H,W]
        tgt_oh = _one_hot_mc(targets, self.C, self.ignore_index)  # [B,C,H,W]

        probs = probs * valid
        tgt_oh = tgt_oh * valid

        start_c = 0 if self.include_bg else 1
        probs = probs[:, start_c:, :, :]
        tgt_oh = tgt_oh[:, start_c:, :, :]

        probs_f = probs.flatten(2)
        tgt_f = tgt_oh.flatten(2)

        inter = (probs_f * tgt_f).sum(dim=2)  # [B,C]
        denom = probs_f.sum(dim=2) + tgt_f.sum(dim=2)  # [B,C]
        dice = (2 * inter + self.smooth) / (denom + self.smooth)  # [B,C]

        if self.skip_empty:
            has_fg = (tgt_f.sum(dim=2) > 0)  # [B,C]
            mask = has_fg.to(dice.dtype)
            dice = (dice * mask + (1 - mask))
            denom_cls = mask.sum(dim=1).clamp_min(1.0)
            dice_mean = (dice.sum(dim=1) / denom_cls)  # [B]
        else:
            dice_mean = dice.mean(dim=1)

        if self.class_weights is not None:
            w = self.class_weights[start_c:]  # [C_eff]
            w = w / (w.sum() + 1e-8)
            dice_w = (dice * w.unsqueeze(0)).sum(dim=1)  # [B]
            loss = 1.0 - dice_w.mean()
        else:
            loss = 1.0 - dice_mean.mean()

        return loss


class CEDiceLoss(nn.Module):
    """
    多类组合损失：CE + λ * Dice
    """

    def __init__(self,
                 num_classes: int,
                 ce_weight: float = 1.0,
                 dice_weight: float = 0.3,
                 ignore_index: Optional[int] = 0,
                 label_smoothing: float = 0.0,
                 class_weights: Optional[Sequence[float]] = None,
                 include_background_in_dice: bool = False,
                 dice_smooth: float = 1e-6):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)

        if class_weights is not None:
            ce_w = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("ce_class_weights", ce_w, persistent=False)
        else:
            self.ce_class_weights = None

        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index if ignore_index is not None else -100,
            weight=self.ce_class_weights,
            label_smoothing=float(label_smoothing)
        )

        self.dice = MultiClassDiceLoss(
            num_classes=num_classes,
            smooth=dice_smooth,
            ignore_index=ignore_index,
            include_background=include_background_in_dice,
            class_weights=class_weights,
            skip_empty=True
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice