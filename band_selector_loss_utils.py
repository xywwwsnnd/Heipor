# band_selector_loss_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math, torch
import torch.nn.functional as F
from dataclasses import dataclass, fields
from typing import Optional, Dict, Tuple

__all__ = ["compute_total_loss"]

# ==== 读取全局 Config 的 Dice 权重日程参数（若不可用则回退默认） ====
try:
    from vit_model.config import Config as _GlobalCfg

    _DICE_W0 = float(getattr(_GlobalCfg, "dice_weight_start", 0.1))
    _DICE_W1 = float(getattr(_GlobalCfg, "dice_weight_target", 0.5))
    _DICE_WARM = float(getattr(_GlobalCfg, "dice_warmup_ratio", 0.10))
    _DICE_RAMP = float(getattr(_GlobalCfg, "dice_ramp_ratio", 0.30))
    _POS_W = getattr(_GlobalCfg, "bce_pos_weight", None)
except Exception:
    _DICE_W0, _DICE_W1, _DICE_WARM, _DICE_RAMP = 0.1, 0.5, 0.10, 0.30
    _POS_W = None


# ------------------------- 自适应约束配置 ------------------------- #
@dataclass
class Cfg:
    epoch: int = 0
    mode: str = "auto_lambda"  # "dual_constraints" | "auto_lambda"
    ema: float = 0.98

    # 约束目标（仅对 dual_constraints 模式生效）
    tau_peak_init: float = 0.90
    tau_div_init: float = 0.03
    tau_peak_ceiling: float = 1.20
    tau_div_floor: float = 0.10
    tau_warmup_epochs: int = 30
    dual_lr: float = 0.05
    mu_max: float = 1.0

    normalize_entropy: bool = True

    # 自动 λ 的控制参数
    target_reward_ratio: float = 0.08
    peak_share: float = 0.40
    min_lambda: float = 0.0
    max_lambda: float = 0.15


class _EMA:
    def __init__(self, beta=0.98):
        self.beta = float(beta);
        self.init = False;
        self.x = None

    def update(self, v: torch.Tensor):
        v = v.detach()
        if not self.init:
            self.x = v;
            self.init = True
        else:
            self.x = self.beta * self.x + (1 - self.beta) * v

    @property
    def val(self):
        return self.x


_STATE = {}


def _state(dev: torch.device, cfg: Cfg):
    key = str(dev)
    s = _STATE.get(key)
    if s is None:
        s = dict(
            ema_seg=_EMA(cfg.ema), ema_peak=_EMA(cfg.ema), ema_div=_EMA(cfg.ema),
            mu_p=torch.tensor(0.0, device=dev), mu_d=torch.tensor(0.0, device=dev),
            epoch=torch.tensor(0, device=dev, dtype=torch.long),
            tau_peak=torch.tensor(cfg.tau_peak_init, device=dev),
            tau_div=torch.tensor(cfg.tau_div_init, device=dev),
            lambda_peak=torch.tensor(0.02, device=dev),
            lambda_div=torch.tensor(0.05, device=dev),
        )
        _STATE[key] = s
    return s


def _entropy_max_from_reg(reg_dict) -> float | None:
    if isinstance(reg_dict, dict):
        if "entropy_max" in reg_dict and reg_dict["entropy_max"]:
            try:
                return float(reg_dict["entropy_max"])
            except Exception:
                pass
        for k in ("K", "num_bands", "bands"):
            if k in reg_dict and reg_dict[k]:
                try:
                    return math.log(float(reg_dict[k]))
                except Exception:
                    pass
    return None


# ------------------------- Dice 权重日程 + 分离 BCE/Dice ------------------------- #

def _to_bin_mask(targets: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    生成二分类前景掩膜：
    - 背景(0) -> 0
    - 忽略(255) -> 0 (重要修改：排除255)
    - 前景(>0) -> 1
    """
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    # [修改] 只有大于0且不等于ignore_index的才是前景
    return ((targets > 0) & (targets != ignore_index)).to(torch.float32)


def _bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, pos_weight: Optional[torch.Tensor],
                     ignore_index: int = 255):
    """
    带 Valid Mask 的 BCE Loss，忽略 ignore_index 区域
    """
    # 1. 生成前景 mask (0/1)
    bin_targets = _to_bin_mask(targets, ignore_index=ignore_index)

    # 2. 生成有效区域 mask (valid)
    if targets.dim() == 3: targets = targets.unsqueeze(1)
    valid_mask = (targets != ignore_index).to(logits.dtype)

    # 3. 对齐尺寸 (logits 对齐到 target)
    if logits.shape[-2:] != bin_targets.shape[-2:]:
        logits = F.interpolate(logits, size=bin_targets.shape[-2:], mode="bilinear", align_corners=False)
        # valid_mask 使用 nearest 插值保持 0/1 性质
        valid_mask = F.interpolate(valid_mask, size=bin_targets.shape[-2:], mode="nearest")

    # 4. 计算 Weighted BCE (reduction='sum' 后手动求 mean)
    # weight=valid_mask 会让无效区域的 loss 变为 0
    loss = F.binary_cross_entropy_with_logits(logits, bin_targets, pos_weight=pos_weight, weight=valid_mask,
                                              reduction='sum')

    # 5. 归一化
    valid_sum = valid_mask.sum()
    return loss / (valid_sum + 1e-6)


def _dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6, ignore_index: int = 255):
    """
    带 Valid Mask 的 Dice Loss，忽略 ignore_index 区域
    """
    probs = torch.sigmoid(logits)

    # 1. 生成前景 mask
    bin_targets = _to_bin_mask(targets, ignore_index=ignore_index)

    # 2. 生成有效区域 mask
    if targets.dim() == 3: targets = targets.unsqueeze(1)
    valid_mask = (targets != ignore_index).to(probs.dtype)

    # 3. 对齐尺寸
    if probs.shape[-2:] != bin_targets.shape[-2:]:
        probs = F.interpolate(probs, size=bin_targets.shape[-2:], mode="bilinear", align_corners=False)
        valid_mask = F.interpolate(valid_mask, size=bin_targets.shape[-2:], mode="nearest")

    # 4. 应用 mask：无效区域 prob 和 target 都置 0
    probs = probs * valid_mask
    bin_targets = bin_targets * valid_mask

    inter = torch.sum(probs * bin_targets)
    denom = torch.sum(probs) + torch.sum(bin_targets)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice


def _dice_weight_schedule(global_step: Optional[int], total_steps: Optional[int],
                          w0: float = _DICE_W0, w1: float = _DICE_W1,
                          warmup_ratio: float = _DICE_WARM, ramp_ratio: float = _DICE_RAMP) -> float:
    if (global_step is None) or (total_steps is None) or (total_steps <= 0):
        return float(w1)
    p = max(0.0, min(1.0, float(global_step) / float(total_steps)))
    warm_end = float(warmup_ratio)
    ramp_end = float(warmup_ratio + ramp_ratio)
    if p <= warm_end:
        w = w0
    elif p <= ramp_end and ramp_end > warm_end:
        t = (p - warm_end) / max(1e-8, (ramp_end - warm_end))
        w = w0 + (w1 - w0) * t
    else:
        w = w1
    return float(w)


def _coerce_cfg(adapt_cfg) -> Cfg:
    if adapt_cfg is None:
        return Cfg()
    if isinstance(adapt_cfg, Cfg):
        return adapt_cfg
    if isinstance(adapt_cfg, dict):
        allowed = {f.name for f in fields(Cfg)}
        clean = {k: v for k, v in adapt_cfg.items() if k in allowed}
        return Cfg(**clean)
    return Cfg()


# ------------------------------- 主入口：总损失 ------------------------------- #
def compute_total_loss(
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
        reg_dict: Dict[str, torch.Tensor | float] | None,
        criterion=None,
        lambda_peak="auto",
        lambda_div="auto",
        adapt_cfg: dict | Cfg | None = None,
        *,
        global_step: Optional[int] = None,
        total_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    dev, dtype = pred_logits.device, pred_logits.dtype
    cfg = _coerce_cfg(adapt_cfg)
    st = _state(dev, cfg)

    try:
        st["epoch"] = torch.tensor(int(cfg.epoch), device=dev, dtype=torch.long)
    except Exception:
        pass

    # ===== 分割项：动态加权的 BCE + Dice (显式传递 ignore_index=255) =====
    try:
        pos_w = _POS_W.to(dev, dtype) if _POS_W is not None else None
    except Exception:
        pos_w = None

    # [修改] 传入 ignore_index=255，修复 CUDA device assert 错误
    bce = _bce_with_logits(pred_logits, targets, pos_weight=pos_w, ignore_index=255)
    dice = _dice_loss_from_logits(pred_logits, targets, ignore_index=255)

    w_dice = _dice_weight_schedule(global_step, total_steps)
    w_bce = 1.0 - w_dice
    seg_loss = w_bce * bce + w_dice * dice

    # ===== 选带正则 =====
    peak = 0.0
    div = 0.0
    if isinstance(reg_dict, dict):
        peak = reg_dict.get("peak", 0.0)
        div = reg_dict.get("div", 0.0)
    peak = peak.to(dev, dtype) if isinstance(peak, torch.Tensor) else torch.tensor(float(peak), device=dev, dtype=dtype)
    div = div.to(dev, dtype) if isinstance(div, torch.Tensor) else torch.tensor(float(div), device=dev, dtype=dtype)

    if cfg.normalize_entropy:
        em = _entropy_max_from_reg(reg_dict)
        if (em is not None) and (em > 0):
            div_used = div / float(em)
            ent_used_info = ("norm", float(em))
        else:
            div_used = div
            ent_used_info = ("raw", None)
    else:
        div_used = div
        ent_used_info = ("raw", None)

    st["ema_seg"].update(seg_loss.detach().abs())
    st["ema_peak"].update(peak.detach().abs())
    st["ema_div"].update(div_used.detach().abs())

    # ===== 模式一：对偶约束 (Dual Constraints) =====
    if cfg.mode == "dual_constraints":
        ep = int(st["epoch"])
        if ep < cfg.tau_warmup_epochs:
            with torch.no_grad():
                warm = (ep + 1) / max(1, cfg.tau_warmup_epochs)
                st["tau_peak"] = (1 - warm) * st["ema_peak"].val + warm * torch.tensor(cfg.tau_peak_init, device=dev)
                st["tau_div"] = (1 - warm) * st["ema_div"].val + warm * torch.tensor(cfg.tau_div_init, device=dev)

        tau_peak = torch.clamp(st["tau_peak"], max=cfg.tau_peak_ceiling)
        tau_div = torch.clamp(st["tau_div"], min=cfg.tau_div_floor)

        pen_peak = torch.relu(tau_peak - peak)
        pen_div = torch.relu(div_used - tau_div)

        mu_p = torch.clamp(st["mu_p"], 0.0, cfg.mu_max)
        mu_d = torch.clamp(st["mu_d"], 0.0, cfg.mu_max)

        reg_loss = mu_p * pen_peak + mu_d * pen_div
        total = seg_loss + reg_loss

        with torch.no_grad():
            st["mu_p"] = torch.clamp(mu_p + cfg.dual_lr * (tau_peak - st["ema_peak"].val), 0.0, cfg.mu_max)
            st["mu_d"] = torch.clamp(mu_d + cfg.dual_lr * (st["ema_div"].val - tau_div), 0.0, cfg.mu_max)

        loss_dict = {
            "total": float(total.detach().cpu()),
            "seg": float(seg_loss.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "dice": float(dice.detach().cpu()),
            "w_dice": float(w_dice),
            "reg_loss": float(reg_loss.detach().cpu()),
            "peak": float(peak.detach().cpu()),
            "div": float(div.detach().cpu()),
            "div_used": float(div_used.detach().cpu()),
            "tau_peak": float(tau_peak.detach().cpu()),
            "tau_div": float(tau_div.detach().cpu()),
            "mu_p": float(st["mu_p"].detach().cpu()),
            "mu_d": float(st["mu_d"].detach().cpu()),
            "entropy_used": ent_used_info[0],
            "entropy_max": ent_used_info[1],
            "mode": "dual_constraints",
        }
        return total, loss_dict

    # ===== 模式二：自动 λ (Auto Lambda) =====
    lam_p = torch.as_tensor(lambda_peak if lambda_peak != "auto" else float(st["lambda_peak"]), device=dev, dtype=dtype)
    lam_d = torch.as_tensor(lambda_div if lambda_div != "auto" else float(st["lambda_div"]), device=dev, dtype=dtype)
    if (lambda_peak == "auto") or (lambda_div == "auto"):
        seg_ref = (st["ema_seg"].val + 1e-8)
        reward_target = cfg.target_reward_ratio * seg_ref
        rew_p = cfg.peak_share * reward_target
        rew_d = (1 - cfg.peak_share) * reward_target
        epv = (st["ema_peak"].val + 1e-8);
        edv = (st["ema_div"].val + 1e-8)
        lam_p_new = torch.clamp(rew_p / epv, min=cfg.min_lambda, max=cfg.max_lambda)
        lam_d_new = torch.clamp(rew_d / edv, min=cfg.min_lambda, max=cfg.max_lambda)
        a = 0.9
        lam_p = a * lam_p + (1 - a) * lam_p_new
        lam_d = a * lam_d + (1 - a) * lam_d_new
        st["lambda_peak"] = lam_p.detach().float().cpu()
        st["lambda_div"] = lam_d.detach().float().cpu()

    reg_loss = -lam_p * peak + lam_d * div_used
    total = seg_loss + reg_loss

    loss_dict = {
        "total": float(total.detach().cpu()),
        "seg": float(seg_loss.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "w_dice": float(w_dice),
        "reg_loss": float(reg_loss.detach().cpu()),
        "peak": float(peak.detach().cpu()),
        "div": float(div.detach().cpu()),
        "div_used": float(div_used.detach().cpu()),
        "lambda_peak": float(lam_p.detach().cpu()),
        "lambda_div": float(lam_d.detach().cpu()),
        "entropy_used": ent_used_info[0],
        "entropy_max": ent_used_info[1],
        "mode": "auto_lambda",
    }
    return total, loss_dict