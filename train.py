# -*- coding: utf-8 -*-
# train.py (Fixed & Refine Logic Updated)
import os, sys, signal, faulthandler, time, json, traceback, platform, multiprocessing as mp, math, random, shutil
import psutil, torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from band_selector_loss_utils import compute_total_loss

from vit_model.model import VisionTransformer
from vit_model.config import Config
from dataset import HyperspectralDataset
from loss import BCEDiceLoss

# ==== 评估：严格使用老师的口径（硬阈值 0.5；global Dice） ====
import torch.nn.functional as F
from metrics import MyMetrics  # 保留

USE_CHANNELS_LAST = True
PRINT_LOSS_EVERY = getattr(Config, "print_loss_every", 20)

os.environ.update({"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"})
torch.set_num_threads(8)
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HALT_ON_NAN = True
DEBUG_NAN = True

mp.set_start_method("spawn", force=True)
os.makedirs(Config.log_dir, exist_ok=True)
ERR_LOG = os.path.join(Config.log_dir, "error_log.txt")
faulthandler.enable(all_threads=True, file=open(ERR_LOG, "w"))
for _sig in (signal.SIGSEGV, signal.SIGABRT):
    signal.signal(_sig, lambda s, f_: faulthandler.dump_traceback(file=open(ERR_LOG, "a")))

LOG_FILE = os.path.join(Config.log_dir, "train_log.txt")


def log(msg: str, end="\n"):
    print(msg, end=end);
    sys.stdout.flush()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + end)


def dump_mem(tag=""):
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    gpu = torch.cuda.memory_allocated() / 1e9
    ram = psutil.Process().memory_info().rss / 1e9
    log(f"[MEM] {tag} | GPU: {gpu:.2f} GB | RAM: {ram:.2f} GB")


def set_seed(s=42):
    random.seed(s);
    np.random.seed(s)
    torch.manual_seed(s);
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


set_seed(getattr(Config, "seed", 42))


def env_report():
    p = psutil.Process()
    info = {
        "hostname": platform.node(),
        "cpu_logical": psutil.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "total_RAM_GB": round(psutil.virtual_memory().total / 1e9, 1),
        "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "process_threads": p.num_threads(),
    }
    log("ENV → " + json.dumps(info, ensure_ascii=False))


def build_step_scheduler(optimizer, steps_per_epoch, num_epochs,
                         warmup_epochs=5, start_factor=0.1, min_lr=1e-6):
    total_updates = steps_per_epoch * num_epochs
    warmup_updates = int(warmup_epochs * steps_per_epoch)

    lambdas = []
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    for base_lr in base_lrs:
        min_lr_ratio = float(min_lr) / float(base_lr) if base_lr > 0 else 0.0

        def lr_lambda(current_update, mrr=min_lr_ratio):
            if total_updates <= 0:
                return 1.0
            if current_update < warmup_updates:
                return float(start_factor) + (1.0 - float(start_factor)) * (current_update / max(1, warmup_updates))
            t = (current_update - warmup_updates) / max(1, (total_updates - warmup_updates))
            cosine = 0.5 * (1 + math.cos(math.pi * t))
            return mrr + (1.0 - mrr) * cosine

        lambdas.append(lr_lambda)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    return scheduler, total_updates


# ---------- 标签整理：把任意来自 Dataset 的标签整理为 [B,H,W] int64 ----------
def _as_mc_target(lab: torch.Tensor) -> torch.Tensor:
    """
    - 若是 [B,1,H,W]，压成 [B,H,W]
    - 若是浮点/0-1/0-255，四舍五入到整数并 clamp>=0
    - 最终返回 long 型 [B,H,W]（CE 的期望）
    """
    if lab.dim() == 4 and lab.size(1) == 1:
        lab = lab[:, 0, ...]
    if lab.dtype.is_floating_point:
        lab = torch.round(lab).clamp_min(0).to(torch.int64)
    else:
        lab = lab.to(torch.int64)
    return lab


@torch.inference_mode()
def _unpack_batch_for_eval(batch):
    """
    兼容 train=True（三元组）与 train=False（四元组）两种返回格式：
      - (hsi, rgb, lab_256) 或
      - (hsi, rgb, lab_256, lab_orig)
    """
    if isinstance(batch, (list, tuple)) and len(batch) == 4:
        hsi, rgb, lab_256, lab_orig = batch
        return hsi, rgb, lab_256, lab_orig
    elif isinstance(batch, (list, tuple)) and len(batch) == 3:
        hsi, rgb, lab_256 = batch
        return hsi, rgb, lab_256, None
    else:
        raise RuntimeError(
            f"Unexpected batch structure: type={type(batch)}, len={len(batch) if hasattr(batch, '__len__') else 'NA'}")


@torch.inference_mode()
def eval_model_teacher(net, loader, dev, desc="EVAL", prefer_orig=False):
    """
    老师口径（二类）：把多类预测映射成前景/背景，硬阈值0.5做 global Dice。
    [修正] 强制忽略 255 区域（不仅在 GT 中忽略，在 Pred 中也将其 mask 掉，防止 False Positive）
    """
    net.eval()
    inter_sum = 0.0
    pred_sum = 0.0
    gt_sum = 0.0
    for batch in tqdm(loader, desc=desc, leave=False):
        hsi, rgb, lab_256, lab_orig = _unpack_batch_for_eval(batch)

        hsi = hsi.to(dev, non_blocking=True)
        rgb = rgb.to(dev, non_blocking=True)

        # 目标分辨率选择
        target_lab = lab_orig if (prefer_orig and lab_orig is not None) else lab_256
        lab = _as_mc_target(target_lab.to(dev, non_blocking=True))  # [B,H,W]

        # [修正] 制作二值 Mask：排除背景(0) 和 忽略区域(255)
        # 1. 忽略区域 Mask (用于过滤预测)
        mask_valid = (lab != 255)

        # 2. GT 二值化 (0=背景/忽略, 1=前景)
        # 注意：这里 lab>0 且 lab!=255 才是 1。
        lab_bin = ((lab > 0) & mask_valid).to(torch.float32)

        logits, *_ = net(rgb, hsi)  # [B,C,h,w] 或 [B,1,h,w]
        if logits.shape[-2:] != lab_bin.shape[-2:]:
            logits = F.interpolate(logits, size=lab_bin.shape[-2:], mode="bilinear", align_corners=False)

        # 多类 → 二类 logit：fg - bg
        if logits.shape[1] >= 2:
            bg = logits[:, :1]
            fg = torch.logsumexp(logits[:, 1:], dim=1, keepdim=True)
            bin_logit = fg - bg
        else:
            bin_logit = logits

        prob = torch.sigmoid(bin_logit)
        pred = (prob >= 0.5).to(torch.float32)  # [B,1,H,W]
        gt = lab_bin.unsqueeze(1)  # [B,1,H,W]

        # [关键] 应用 Valid Mask，确保 255 区域不计入 pred_sum (防止虚假 False Positive)
        mask_valid = mask_valid.unsqueeze(1).to(torch.float32)
        pred = pred * mask_valid

        pred = pred.detach().cpu()
        gt = gt.detach().cpu()

        inter_sum += float((pred * gt).sum())
        pred_sum += float(pred.sum())
        gt_sum += float(gt.sum())

    dice = 1.0 if (pred_sum == 0.0 and gt_sum == 0.0) else (2.0 * inter_sum / max(1e-8, (pred_sum + gt_sum)))
    return {1: {"dice": round(dice, 4), "iou": 0.0, "recall": 0.0, "precision": 0.0, "specificity": 0.0}}


@torch.inference_mode()
def eval_model_multiclass(net, loader, dev, num_classes, desc="EVAL-MC", prefer_orig=False):
    """
    多类 mDice/mIoU（忽略背景0；仅统计出现在GT中的类）
    [修正] 强制忽略 255 (忽略区域不计入 pred_sum，也不计入 gt_sum)
    """
    net.eval()
    K = int(num_classes)
    inter = torch.zeros(K, dtype=torch.float64)
    predA = torch.zeros(K, dtype=torch.float64)
    gtA = torch.zeros(K, dtype=torch.float64)

    for batch in tqdm(loader, desc=desc, leave=False):
        hsi, rgb, lab_256, lab_orig = _unpack_batch_for_eval(batch)

        hsi = hsi.to(dev, non_blocking=True)
        rgb = rgb.to(dev, non_blocking=True)

        target_lab = lab_orig if (prefer_orig and lab_orig is not None) else lab_256
        lab = _as_mc_target(target_lab.to(dev, non_blocking=True))  # [B,H,W]

        # [修正] 在 Eval 时也要保护 255，不要把它变成 0
        mask_ignore = (lab == 255)
        lab[lab >= K] = 0
        lab[mask_ignore] = 255  # 恢复 255

        logits, *_ = net(rgb, hsi)  # [B,C,h,w]
        if logits.shape[-2:] != lab.shape[-2:]:
            logits = F.interpolate(logits, size=lab.shape[-2:], mode="bilinear", align_corners=False)

        pred = torch.argmax(logits, dim=1)  # [B,H,W]

        # 构造有效掩码 (lab != 255)
        valid_mask = (lab != 255)

        for c in range(1, K):  # 忽略背景0
            # 只在 valid 区域计算
            # p: 预测为 c 且 不是忽略区域
            # g: 真值为 c (lab为255时 g必为False)
            p = (pred == c) & valid_mask
            g = (lab == c)

            # 修复：加 .item() 把结果取回 CPU
            inter[c] += (p & g).sum().item()
            predA[c] += p.sum().item()
            gtA[c] += g.sum().item()

    eps = 1e-8
    union = predA + gtA - inter
    iou_c = inter / (union + eps)
    dice_c = (2 * inter) / (predA + gtA + eps)

    valid = (gtA > 0)
    mIoU = (iou_c[valid]).mean().item() if valid.any() else 0.0
    mDice = (dice_c[valid]).mean().item() if valid.any() else 0.0
    return {"miou": round(mIoU, 4), "mdice": round(mDice, 4)}


def _add_nan_hook(module, name):
    def _hook(_m, _inp, out):
        t = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (tuple, list)) else None)
        if t is not None and not torch.isfinite(t).all():
            mx = torch.nan_to_num(t.detach()).abs().max().item()
            print(f"❌ NaN after [{name}] | max|out|={mx:.2e}")
            raise SystemExit

    module.register_forward_hook(_hook)


def _pack_rng():
    return {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _unpack_rng(state):
    try:
        if not state: return
        random.setstate(state["py_random"])
        np.random.set_state(state["np_random"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda", None) is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as e:
        log(f"[CKPT] RNG restore failed: {e}")


def save_ckpt(path, net, opt, sched, epoch, global_update, best_val_mdice, extra=None):  # ★ rename
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {
        "model": net.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "global_update": global_update,
        "best_val_dice": float(best_val_mdice),  # 字段名沿用旧键，值存多类 mDice
        "rng_state": _pack_rng(),
        "config": {k: getattr(Config, k) for k in dir(Config) if not k.startswith("__")},
    }
    if extra: to_save.update(extra)
    torch.save(to_save, path)
    log(f"[CKPT] Saved full checkpoint → {path}")


def load_ckpt(path, net, opt=None, sched=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)

    def _safe_partial_load(module, sd):
        model_sd = module.state_dict()
        keep, skipped = {}, []
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                keep[k] = v
            else:
                skipped.append(k)
        missing, unexpected = module.load_state_dict(keep, strict=False)
        if skipped:
            log(f"[CKPT] skipped (missing/renamed): {len(skipped)} → e.g. {skipped[:3]}")
        if missing:
            log(f"[CKPT] missing in ckpt: {len(missing)} → e.g. {missing[:3]}")
        if unexpected:
            log(f"[CKPT] unexpected in ckpt: {len(unexpected)} → e.g. {unexpected[:3]}")

    if isinstance(state, dict) and "model" in state:
        try:
            net.load_state_dict(state["model"], strict=True)
        except RuntimeError as e:
            log(f"[CKPT] strict load failed → fallback to shape-matched partial load. Reason: {e}")
            _safe_partial_load(net, state["model"])

        if opt is not None and "optimizer" in state and state["optimizer"] is not None:
            try:
                opt.load_state_dict(state["optimizer"])
            except Exception as e:
                log(f"[CKPT] optimizer.load_state_dict failed (ignored): {e}")

        if sched is not None and "scheduler" in state and state["scheduler"] is not None:
            try:
                sched.load_state_dict(state["scheduler"])
            except Exception as e:
                log(f"[CKPT] scheduler.load_state_dict failed (ignored): {e}")

        _unpack_rng(state.get("rng_state", {}))
        return {
            "epoch": int(state.get("epoch", 0)),
            "global_update": int(state.get("global_update", 0)),
            "best_val_dice": float(state.get("best_val_dice", -1.0)),
        }
    else:
        try:
            net.load_state_dict(state, strict=True)
        except RuntimeError as e:
            log(f"[CKPT] strict load (pure sd) failed → fallback: {e}")
            _safe_partial_load(net, state)
        return {"epoch": 0, "global_update": 0, "best_val_dice": -1.0}


def _fmt(v, d=4):
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return str(v)


def _maybe_log_band_metrics(loss_dict, prefix="[Loss] "):
    keys_order = [
        ("total", "total"), ("seg", "seg"), ("bce", "bce"), ("dice", "dice"),
        ("w_dice", "w_dice"), ("reg_loss", "reg"),
        ("peak", "peak"), ("div", "div"), ("div_used", "div_used"),
        ("tau_peak", "tau_p"), ("tau_div", "tau_d"), ("mu_p", "mu_p"), ("mu_d", "mu_d"),
    ]
    parts = []
    for k, alias in keys_order:
        if k in loss_dict:
            parts.append(f"{alias}={_fmt(loss_dict[k])}")
    if parts:
        log(prefix + ", ".join(parts))


# [修改] 增加 mask 参数支持
def _soft_dice_loss(logits, target, mask=None, eps=1e-6):
    if target.dim() == 3: target = target.unsqueeze(1)

    p = torch.sigmoid(logits)

    # 如果有 mask，先过滤 (255 区域)
    if mask is not None:
        if mask.dim() == 3: mask = mask.unsqueeze(1)
        # 对齐 mask 尺寸
        if mask.shape[-2:] != logits.shape[-2:]:
            mask = F.interpolate(mask, size=logits.shape[-2:], mode="nearest")
        p = p * mask
        target = target * mask

    inter = (p * target).sum(dim=(2, 3))
    denom = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = 1.0 - (2.0 * inter + eps) / denom
    return dice.mean()


# [修改] 增加 mask 参数支持
def _seg_criterion_refine(logits, targets, pos_weight, mask=None, bce_w=0.5, dice_w=2.0):
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    # 处理 BCE weight
    weight = None
    if mask is not None:
        if mask.dim() == 3: mask = mask.unsqueeze(1)
        if mask.shape[-2:] != logits.shape[-2:]:
            mask = F.interpolate(mask, size=logits.shape[-2:], mode="nearest")
        weight = mask

    # 1. BCE: 传入 weight=mask
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, weight=weight)(logits, targets)

    # 2. Dice: 传入 mask
    dice = _soft_dice_loss(logits, targets, mask=weight)

    return bce_w * bce + dice_w * dice


def _get_band_selector_params(net):
    try:
        bs = net.transformer.embeddings.band_sel
        return list(bs.parameters()) if bs is not None else []
    except Exception:
        return []


def train():
    dev = torch.device(Config.device)

    # 数据集
    # [修正] 强制开启 per_sample_minmax=True，确保 HSI 数据在 [0,1] 范围内
    tr_ds = HyperspectralDataset(
        Config.hyperspectral_dir,
        Config.rgb_dir,
        Config.label_dir,
        Config.train_list,
        image_size=Config.image_size,
        train=True,
        num_channels_hsi=Config.num_channels_hsi,
        multi_class=True,
        per_sample_minmax=True  # ★ FORCE TRUE
    )

    va_ds = HyperspectralDataset(
        Config.hyperspectral_dir,
        Config.rgb_dir,
        Config.label_dir,
        Config.test_list,
        image_size=Config.image_size,
        train=False,
        num_channels_hsi=Config.num_channels_hsi,
        multi_class=True,
        per_sample_minmax=True  # ★ FORCE TRUE
    )

    # DataLoader
    use_pin = (dev.type == "cuda")
    num_w = max(4, getattr(Config, 'num_workers', 4))
    dl_kwargs_tr = dict(batch_size=Config.batch_size, shuffle=True,
                        num_workers=num_w, pin_memory=use_pin, persistent_workers=(num_w > 0),
                        drop_last=True)
    if num_w > 0:
        dl_kwargs_tr["prefetch_factor"] = 2
    tr_ld = DataLoader(tr_ds, **dl_kwargs_tr)

    dl_kwargs_va = dict(batch_size=Config.batch_size, shuffle=False,
                        num_workers=max(2, num_w // 2), pin_memory=use_pin, persistent_workers=(num_w > 0))
    if max(2, num_w // 2) > 0:
        dl_kwargs_va["prefetch_factor"] = 2
    va_ld = DataLoader(va_ds, **dl_kwargs_va)

    # ★ 模型：多类输出
    num_classes = int(getattr(Config, "num_classes", 21))  # 设为数据集类别数（含背景0）
    net = VisionTransformer(Config.model_config, img_size=Config.image_size, num_classes=num_classes).to(dev)
    net = net.to(torch.float32)
    if USE_CHANNELS_LAST and dev.type == "cuda":
        net = net.to(memory_format=torch.channels_last)

    if any(p.device.type != 'cuda' for p in net.parameters()):
        log("[WARN] some parameters on CPU, check model definition!")
    else:
        log("[OK] Params on GPU")

    # ★ 损失：多类 CE +（二类Dice/BCE + 选带正则）
    ce_weight = float(getattr(Config, "seg_ce_weight", 1.0))
    bs_weight = float(getattr(Config, "bs_bin_weight", 1.0))

    # [关键] 设置 ignore_index=255，确保 CE Loss 忽略无效区域
    ce_criterion = nn.CrossEntropyLoss(ignore_index=255)

    pos_w = getattr(Config, "bce_pos_weight", None)
    pos_w = pos_w.to(dev) if pos_w is not None else None
    crit = BCEDiceLoss(pos_weight=pos_w)  # 保留给 refine 用

    base_lr = Config.learning_rate
    wd = getattr(Config, "weight_decay", 1e-4)

    frft_cfg = getattr(Config.model_config, "frft", None)
    alpha_lr_mult = getattr(frft_cfg, "alpha_lr_mult", 0.2) if frft_cfg else 0.2
    polar_lr_mult = getattr(frft_cfg, "polar_w_lr_mult", 0.5) if frft_cfg else 0.5
    alpha_wd = getattr(frft_cfg, "alpha_weight_decay", 0.0) if frft_cfg else 0.0

    params_rest, params_alpha, params_polar = [], [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("alpha"):
            params_alpha.append(p)
        elif name.endswith("polar_w"):
            params_polar.append(p)
        else:
            params_rest.append(p)

    opt = optim.AdamW(
        [
            {"params": params_rest, "lr": base_lr, "weight_decay": wd},
            {"params": params_alpha, "lr": base_lr * alpha_lr_mult, "weight_decay": alpha_wd},
            {"params": params_polar, "lr": base_lr * polar_lr_mult, "weight_decay": 0.0},
        ],
        lr=base_lr, weight_decay=wd
    )

    log(f"[OPT] groups → rest={sum(p.numel() for p in params_rest)}, "
        f"alpha={sum(p.numel() for p in params_alpha)} (lr×{alpha_lr_mult}, wd={alpha_wd}), "
        f"polar_w={sum(p.numel() for p in params_polar)} (lr×{polar_lr_mult})")

    refine_enable = bool(getattr(Config, "refine_selector_enable", True))
    refine_every = int(getattr(Config, "refine_selector_every", 1))
    refine_steps = int(getattr(Config, "refine_selector_steps", 1))
    refine_lr_mult = float(getattr(Config, "refine_selector_lr_mult", 0.5))
    refine_bce_w = float(getattr(Config, "refine_selector_bce_weight", 0.5))
    refine_dice_w = float(getattr(Config, "refine_selector_dice_weight", 2.0))

    band_params = _get_band_selector_params(net)
    if refine_enable and band_params:
        opt_sel = optim.AdamW(band_params, lr=base_lr * refine_lr_mult, weight_decay=0.0)
        log(f"[REFINE] Enabled | every={refine_every} | steps={refine_steps} | lr_mult={refine_lr_mult} | "
            f"w(BCE,Dice)=({refine_bce_w},{refine_dice_w}) | band_params={sum(p.numel() for p in band_params)}")
    else:
        opt_sel = None
        if refine_enable:
            log("[REFINE] Enabled but band selector not found -> skip refine.")

    if DEBUG_NAN:
        for n, m in net.named_modules():
            if any(k in n for k in ["band_sel", "ca_", "se_", "hsi_proj", "rgb_token_proj", "decoder"]):
                _add_nan_hook(m, n)

    accumulation_steps = max(1, getattr(Config, "accumulation_steps", 1))
    batches_per_epoch = len(tr_ld)
    updates_per_epoch = math.ceil(batches_per_epoch / accumulation_steps)
    warmup_epochs = getattr(Config, "warmup_epochs", 5)
    min_lr = getattr(Config, "min_lr", 1e-6)
    start_factor = getattr(Config, "lr_start_factor", 0.1)

    sched, TOTAL_UPDATES = build_step_scheduler(
        opt, updates_per_epoch, Config.num_epochs,
        warmup_epochs=warmup_epochs, start_factor=start_factor, min_lr=min_lr
    )

    RESUME = getattr(Config, "resume", True)
    RESUME_CKPT = getattr(Config, "resume_ckpt", os.path.join(Config.checkpoint_dir, "model_latest.pth"))
    start_ep, best_score, global_update = 0, -1.0, 0
    if RESUME and os.path.isfile(RESUME_CKPT):
        info = load_ckpt(RESUME_CKPT, net, opt, sched, map_location=dev)
        start_ep = info["epoch"]
        best_score = info["best_val_dice"]
        global_update = info["global_update"]
        log(f"[RESUME] Loaded full ckpt @ {RESUME_CKPT} | epoch={start_ep} | best_val(mDice)={best_score:.4f}")
    else:
        log("[RESUME] Disabled or ckpt not found → start from scratch.")

    skip_metric = getattr(Config, "skip_metric", 100)

    env_report()
    log("Ep | LR | Tr-mDice | Va-mDice | Tr-binDice | Va-binDice | Loss | d(ms) | f(ms) | b(ms) | CPU% | Thr | GPU% | RAM | VRAM")

    try:
        import pynvml;
        pynvml.nvmlInit();
        _NVML_OK = True
        _cuda_idx = int(str(Config.device).split(":")[1]) if ":" in str(Config.device) else 0
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(_cuda_idx)
    except Exception:
        _NVML_OK = False
        _nvml_handle = None
        _cuda_idx = 0

    p = psutil.Process();
    p.cpu_percent(None)
    os.makedirs(Config.checkpoint_dir, exist_ok=True)

    def _new_band_epoch_stat():
        return dict(cnt=0, seg=0.0, reg=0.0, peak=0.0, div=0.0, div_used=0.0,
                    tau_p=0.0, tau_d=0.0, mu_p=0.0, mu_d=0.0, total=0.0)

    for ep in range(start_ep, Config.num_epochs):
        net.train();
        run_loss = 0.0;
        t_data = t_fwd = t_bwd = 0.0
        opt.zero_grad(set_to_none=True)
        band_stat = _new_band_epoch_stat()

        for i, (hsi, rgb, lab) in enumerate(tqdm(tr_ld, desc=f"E{ep + 1}/{Config.num_epochs}", leave=False)):
            t0 = time.perf_counter()
            hsi = hsi.to(dev, non_blocking=True).float()
            rgb = rgb.to(dev, non_blocking=True).float()
            lab = _as_mc_target(lab.to(dev, non_blocking=True))  # ★ 统一成 [B,H,W] int64

            # [修正] 安全钳制，但必须保护 255 (ignore_index)
            # 1. 先把 255 保护起来 (Mask Out)
            mask_ignore = (lab == 255)
            # 2. 把其他大于 num_classes 的异常值(例如 22, 30 等)变成 0
            # 注意：如果 mask_ignore 里的位置也符合 >= num_classes，它们会被暂时设为0
            lab[lab >= num_classes] = 0
            # 3. 恢复 255 (把“VIP”放回来)
            lab[mask_ignore] = 255

            if USE_CHANNELS_LAST and dev.type == "cuda":
                hsi = hsi.to(memory_format=torch.channels_last)
                rgb = rgb.to(memory_format=torch.channels_last)

            if not torch.isfinite(rgb).all(): rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1e3, neginf=-1e3)
            if not torch.isfinite(hsi).all(): hsi = torch.nan_to_num(hsi, nan=0.0, posinf=1e3, neginf=-1e3)

            t_data += time.perf_counter() - t0

            if i == 0:
                log(f"[TRAIN debug] devs→ hsi:{hsi.device} rgb:{rgb.device} lab:{lab.device} shape_lab={tuple(lab.shape)}")

            t1 = time.perf_counter()
            out, reg, *_ = net(rgb, hsi)  # out:[B,C,H',W']

            # 对齐到标签尺度
            if out.shape[-2:] != lab.shape[-2:]:
                out = F.interpolate(out, size=lab.shape[-2:], mode="bilinear", align_corners=True)

            # ★ 1) 多类 CE（真正监督）- 此处 lab 含有 255，CE 会自动忽略
            loss_ce = ce_criterion(out, lab)

            # ★ 2) 把多类 logits 映射成“前景 vs 背景”二类 logit → 交给 compute_total_loss
            # compute_total_loss 内部已修复 logic，会正确忽略 255
            if out.shape[1] >= 2:
                bg = out[:, :1, :, :]
                fg = torch.logsumexp(out[:, 1:, :, :], dim=1, keepdim=True)
                bin_logit = fg - bg
            else:
                bin_logit = out  # 兼容只有1通道的情况

            loss_bs, loss_dict = compute_total_loss(
                pred_logits=bin_logit,
                targets=lab,  # 内部会 ((targets>0) & (targets!=255))
                reg_dict={**reg, "K": Config.num_channels_hsi},
                adapt_cfg={"epoch": ep, "mode": "auto_lambda"},
                global_step=global_update,
                total_steps=TOTAL_UPDATES,
            )

            # ★ 总损失 = 多类CE + 二类(BCE/Dice+正则)
            loss = ce_weight * loss_ce + bs_weight * loss_bs

            if (i % PRINT_LOSS_EVERY) == 0:
                _maybe_log_band_metrics(loss_dict, prefix="[Loss] ")

            band_stat["cnt"] += 1
            for k_src, k_dst in [
                ("total", "total"), ("seg", "seg"), ("reg_loss", "reg"),
                ("peak", "peak"), ("div", "div"), ("div_used", "div_used"),
                ("tau_peak", "tau_p"), ("tau_div", "tau_d"), ("mu_p", "mu_p"), ("mu_d", "mu_d")
            ]:
                if k_src in loss_dict:
                    band_stat[k_dst] += float(loss_dict[k_src])

            if not torch.isfinite(loss):
                mx = torch.nan_to_num(out.detach()).abs().max().item()
                lr_now_dbg = opt.param_groups[0]['lr']
                log(f"[NaN] loss non-finite @ ep{ep + 1} it{i + 1} | lr={lr_now_dbg:.2e} | max|logit|={mx:.2e}")
                if HALT_ON_NAN: raise SystemExit
                loss = torch.nan_to_num(loss)

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                t2 = time.perf_counter()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_update += 1
                sched.step()
                t_bwd += time.perf_counter() - t2

                # REFINE：仅 band selector 用二类目标微调
                if refine_enable and opt_sel is not None and (global_update % refine_every == 0):
                    for r in range(refine_steps):
                        opt_sel.zero_grad(set_to_none=True)
                        out_ref, _, *_ = net(rgb, hsi)
                        if out_ref.shape[-2:] != lab.shape[-2:]:
                            out_ref = F.interpolate(out_ref, size=lab.shape[-2:], mode="bilinear", align_corners=True)

                        # [修正] Refine 时也排除 255
                        # 生成 Valid Mask (255 为 0，其他为 1)
                        valid_mask = (lab != 255).float()
                        # 生成二值标签 (255 暂时变 0，但会被 mask 盖住)
                        lab_bin = ((lab > 0) & (lab != 255)).to(torch.float32)

                        bin_ref = (torch.logsumexp(out_ref[:, 1:], 1, True) - out_ref[:, :1]) if out_ref.shape[
                                                                                                     1] > 1 else out_ref

                        # [修正] 传入 mask
                        loss_ref = _seg_criterion_refine(
                            bin_ref,
                            lab_bin,
                            pos_weight=pos_w,
                            mask=valid_mask,  # <--- 新增参数
                            bce_w=refine_bce_w,
                            dice_w=refine_dice_w
                        )

                        loss_ref.backward()
                        nn.utils.clip_grad_norm_(band_params, 1.0)
                        opt_sel.step()
                    if (i % PRINT_LOSS_EVERY) == 0:
                        log(f"[Refine] it={i + 1} upd={global_update} | steps={refine_steps} | w(BCE,Dice)=({refine_bce_w},{refine_dice_w})")

            t_fwd += time.perf_counter() - t1
            run_loss += float(loss.item())

        if (len(tr_ld) % accumulation_steps) != 0:
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step();
            opt.zero_grad(set_to_none=True)
            global_update += 1;
            sched.step()

        avg = run_loss / len(tr_ld)

        if band_stat["cnt"] > 0:
            cnt = max(1, band_stat["cnt"])
            log("[Band/EpochAvg] " +
                ", ".join([
                    f"total={band_stat['total'] / cnt:.4f}",
                    f"seg={band_stat['seg'] / cnt:.4f}",
                    f"reg={band_stat['reg'] / cnt:.4f}",
                    f"peak={band_stat['peak'] / cnt:.4f}",
                    f"div={band_stat['div'] / cnt:.4f}",
                    f"div_used={band_stat['div_used'] / cnt:.4f}",
                    f"tau_p={band_stat['tau_p'] / cnt:.4f}",
                    f"tau_d={band_stat['tau_d'] / cnt:.4f}",
                    f"mu_p={band_stat['mu_p'] / cnt:.4f}",
                    f"mu_d={band_stat['mu_d'] / cnt:.4f}",
                ]))

        # --------- 评估（多类 + 老师口径二类） ---------
        if ep + 1 > skip_metric:
            # prefer_orig=True：若 batch 提供 label_orig（验证集），则在原始分辨率评估
            trM_bin = eval_model_teacher(net, tr_ld, dev, desc=f"TRN-EVAL(bin) {ep + 1}/{Config.num_epochs}",
                                         prefer_orig=True)
            vaM_bin = eval_model_teacher(net, va_ld, dev, desc=f"VAL-EVAL(bin) {ep + 1}/{Config.num_epochs}",
                                         prefer_orig=True)

            trM_mc = eval_model_multiclass(net, tr_ld, dev, num_classes,
                                           desc=f"TRN-EVAL(mc) {ep + 1}/{Config.num_epochs}", prefer_orig=True)
            vaM_mc = eval_model_multiclass(net, va_ld, dev, num_classes,
                                           desc=f"VAL-EVAL(mc) {ep + 1}/{Config.num_epochs}", prefer_orig=True)

            # 以 Val mDice（多类，忽略背景）作为最佳模型判据
            if vaM_mc['mdice'] > (best_score if 'best_score' in locals() else -1.0):
                best_score = vaM_mc['mdice']
                best_ckpt = os.path.join(Config.checkpoint_dir, "model_best.pth")
                save_ckpt(best_ckpt, net, opt, sched, epoch=ep + 1,
                          global_update=global_update, best_val_mdice=best_score,
                          extra={"tag": "best"})
                latest = os.path.join(Config.checkpoint_dir, "model_latest.pth")
                try:
                    shutil.copy2(best_ckpt, latest)
                except Exception:
                    torch.save(torch.load(best_ckpt, map_location="cpu"), latest)
                log(f"[BEST] Saved & updated latest: {best_ckpt}")
        else:
            trM_bin = vaM_bin = {1: {'dice': 0, 'iou': 0, 'recall': 0, 'specificity': 0, 'precision': 0}}
            trM_mc = vaM_mc = {'mdice': 0.0, 'miou': 0.0}

        lr_now = opt.param_groups[0]['lr']
        cpu = p.cpu_percent(0.1);
        thr = p.num_threads();
        ram = psutil.virtual_memory().used / 1e9
        if ('pynvml' in sys.modules) and _NVML_OK and (dev.type == "cuda"):
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle).gpu
                vram = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle).used / 1e9
            except Exception:
                util = -1
                vram = torch.cuda.memory_allocated(device=dev) / 1e9
        else:
            util = -1
            vram = torch.cuda.memory_allocated(device=dev) / 1e9

        log(f"{ep + 1:3d}/{Config.num_epochs} | {lr_now:.2e} | {trM_mc['mdice']:.4f} | {vaM_mc['mdice']:.4f} | "
            f"{trM_bin[1]['dice']:.4f} | {vaM_bin[1]['dice']:.4f} | {avg:.4f} | "
            f"{t_data * 1e3 / len(tr_ld):6.1f} | {t_fwd * 1e3 / len(tr_ld):6.1f} | {t_bwd * 1e3 / len(tr_ld):6.1f} | "
            f"{cpu:5.1f}% | {thr:3d} | {util if util >= 0 else 'NA':>3} | {ram:5.1f} | {vram:5.2f}")
        log("-" * 80)

        if (ep + 1) % getattr(Config, "validation_freq", 50) == 0:
            ck = os.path.join(Config.checkpoint_dir, f"model_ep{ep + 1}.pth")
            save_ckpt(ck, net, opt, sched, epoch=ep + 1,
                      global_update=global_update, best_val_mdice=(best_score if 'best_score' in locals() else -1.0),
                      extra={"tag": "periodic"})
            latest = os.path.join(Config.checkpoint_dir, "model_latest.pth")
            try:
                shutil.copy2(ck, latest)
            except Exception:
                torch.save(torch.load(ck, map_location="cpu"), latest)
            log(f"[CKPT] Updated latest -> {latest}")

    log(f"Training done. Best Val mDice={(best_score if 'best_score' in locals() else -1.0):.4f}")


if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            _dev_str = str(Config.device)
            _cuda_idx = int(_dev_str.split(":")[1]) if ":" in _dev_str else 0
            torch.cuda.set_per_process_memory_fraction(0.8, _cuda_idx)
    except RuntimeError:
        pass

    try:
        train()
    except Exception:
        log("[FATAL] exception caught, dumping traceback …")
        traceback.print_exc(file=open(ERR_LOG, "a"))
        dump_mem("FATAL")
        torch.cuda.empty_cache()