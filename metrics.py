# -*- coding: utf-8 -*-
# @Time     : 2023/11/13  15:23
# @Author   : Geng Qin (modified)
# @File     : metrics.py
# @Software : Vscode
import numpy as np
from medpy import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


# ------------------------- 通用小工具 ------------------------- #
def segmentation_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """逐像素准确率（预测/标签同形状，任意整型/布尔）"""
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    return float((y_true_flat == y_pred_flat).sum()) / float(y_true_flat.size)


def _to_numpy_long(x: torch.Tensor) -> np.ndarray:
    """(B,H,W) Long numpy"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().long().numpy()
    return x


def _safe_hd(mask_pred: np.ndarray, mask_gt: np.ndarray) -> Tuple[float, float]:
    """
    计算二值掩码的 HD 和 HD95；若任一全空则返回 0（或你也可以返回 np.nan）
    medpy 的 hd/hd95 对空掩码会报错，这里做了防护。
    """
    if mask_pred.sum() == 0 and mask_gt.sum() == 0:
        return 0.0, 0.0
    if mask_pred.sum() == 0 or mask_gt.sum() == 0:
        # 只要一边全空，HD 可以理解为“极大”；给一个稳妥的兜底
        # 你也可以选择返回 np.nan 并在外部忽略
        return 0.0, 0.0
    hd = float(metric.binary.hd(mask_pred, mask_gt, voxelspacing=None, connectivity=1))
    hd95 = float(metric.binary.hd95(mask_pred, mask_gt, voxelspacing=None, connectivity=1))
    return hd, hd95


# ------------------------- 多类累计器（宏平均） ------------------------- #
class _MCStats:
    """
    维护多类统计：每类 intersection / pred / gt 的全局累加，
    支持按“在 GT 出现过的类”做宏平均（忽略 ignore_index）。
    """
    def __init__(self, num_classes: int, ignore_index: Optional[int] = 0):
        self.K = int(num_classes)
        self.ignore_index = ignore_index
        self.inter = np.zeros(self.K, dtype=np.float64)
        self.pA   = np.zeros(self.K, dtype=np.float64)  # predicted area
        self.gA   = np.zeros(self.K, dtype=np.float64)  # gt area

        # 用于可选的 HD/HD95 统计
        self._hd_sum   = np.zeros(self.K, dtype=np.float64)
        self._hd95_sum = np.zeros(self.K, dtype=np.float64)
        self._hd_cnt   = np.zeros(self.K, dtype=np.int64)  # 该类累积了多少次有效 HD

    def add(self, pred_np: np.ndarray, gt_np: np.ndarray, compute_hd: bool = False):
        """
        pred_np, gt_np: (B,H,W) int
        """
        assert pred_np.shape == gt_np.shape
        B = pred_np.shape[0]

        for b in range(B):
            p = pred_np[b]
            g = gt_np[b]

            # 局部掩掉 ignore_index（只掩 GT）
            if self.ignore_index is not None:
                valid = (g != self.ignore_index)
                p = p[valid]
                g = g[valid]

            # 统计 per-class 面积/交集
            for c in range(self.K):
                if self.ignore_index is not None and c == self.ignore_index:
                    continue
                pc = (p == c)
                gc = (g == c)
                self.pA[c] += pc.sum()
                self.gA[c] += gc.sum()
                self.inter[c] += (pc & gc).sum()

                # 可选 HD/HD95：仅当该类在 GT 或 Pred 出现过才统计
                if compute_hd and (pc.any() or gc.any()):
                    hd, hd95 = _safe_hd(pc.astype(np.uint8), gc.astype(np.uint8))
                    self._hd_sum[c] += hd
                    self._hd95_sum[c] += hd95
                    self._hd_cnt[c] += 1

    def summarize(self) -> Dict[str, object]:
        eps = 1e-8
        union = self.pA + self.gA - self.inter
        dice_c = (2.0 * self.inter) / (self.pA + self.gA + eps)
        iou_c  = self.inter / (union + eps)

        # 仅对“在 GT 出现过的类”做宏平均；忽略 ignore_index
        valid = (self.gA > 0)
        if self.ignore_index is not None and 0 <= self.ignore_index < self.K:
            valid[self.ignore_index] = False

        if valid.any():
            mDice = float(dice_c[valid].mean())
            mIoU  = float(iou_c[valid].mean())
        else:
            mDice = 0.0
            mIoU  = 0.0

        return {
            "dice_per_class": dice_c.astype(np.float32),
            "iou_per_class":  iou_c.astype(np.float32),
            "valid_mask":     valid,
            "mDice":          round(mDice, 4),
            "mIoU":           round(mIoU, 4),
            # HD/HD95（若没开启统计，cnt全0，外部可忽略）
            "hd_per_class":   np.divide(self._hd_sum,   np.maximum(self._hd_cnt, 1), where=(self._hd_cnt>0)).astype(np.float32),
            "hd95_per_class": np.divide(self._hd95_sum, np.maximum(self._hd_cnt, 1), where=(self._hd_cnt>0)).astype(np.float32),
            "hd_cnt":         self._hd_cnt.copy(),
        }


# ------------------------- 旧接口：二类批量指标 ------------------------- #
def calculate_metrics_compose(prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """
    二类整体指标（与旧版兼容）：DSC / IOU / HD / HD95 / Recall / Precision / ACC / ASD
    prediction, ground_truth: 二值 {0,1} numpy 数组（同形状）
    """
    Dsc = metric.binary.dc(prediction, ground_truth)
    Jaccard = metric.binary.jc(prediction, ground_truth)
    hous_dis = metric.binary.hd(prediction, ground_truth, voxelspacing=None, connectivity=1)
    hous_dis95 = metric.binary.hd95(prediction, ground_truth, voxelspacing=None, connectivity=1)
    Recall = metric.binary.recall(prediction, ground_truth)
    Prec = metric.binary.precision(prediction, ground_truth)
    asd = metric.binary.asd(prediction, ground_truth)
    acc = segmentation_accuracy(prediction, ground_truth)

    return {
        'DICE': Dsc,
        'IOU': Jaccard,
        'ACC': acc,
        'Precision': Prec,
        'HD': hous_dis,
        'HD95': hous_dis95,
        'Recall': Recall,
        'asd': asd
    }


# ------------------------- 你的旧类（保留，轻改） ------------------------- #
class MyMetrics_test:
    """
    轻量 IoU 统计（可用于快速对比）。
    当 n_classes>1 时，忽略背景0，返回最后一类的 Dice（保持你原始行为）。
    """
    def __init__(self, max_save=False, n_classes=1, **kwargs):
        self.metrics = {'DICE': 0.}
        self._max = 0. if max_save else None
        self._n_classes = n_classes
        if self._n_classes == 1:
            self.sigmoid = nn.Sigmoid()
        self._intersection = np.zeros(self._n_classes, dtype=np.int64)
        self._prediction_num = np.zeros(self._n_classes, dtype=np.int64)
        self._target_num = np.zeros(self._n_classes, dtype=np.int64)

    def reset_zero(self):
        if self._max is not None:
            improved = self._max < self.metrics['DICE']
            self._max = max(self._max, self.metrics['DICE'])
        self.metrics = {key: 0. for key in self.metrics.keys()}
        self._intersection[:] = 0
        self._prediction_num[:] = 0
        self._target_num[:] = 0
        if self._max is not None:
            return improved

    def _calculate_iou(self, prediction, ground_truth, K, ignored_index=None):
        assert prediction.shape == ground_truth.shape
        pred_flat = prediction.reshape(-1)
        gt_flat = ground_truth.reshape(-1)
        if ignored_index is not None:
            valid = (gt_flat != ignored_index)
            pred_flat = pred_flat[valid]
            gt_flat = gt_flat[valid]

        area_intersection = np.zeros(K, dtype=np.int64)
        area_output = np.zeros(K, dtype=np.int64)
        area_target = np.zeros(K, dtype=np.int64)

        for k in range(K):
            area_output[k] = np.sum(pred_flat == k)
            area_target[k] = np.sum(gt_flat == k)

        correct_mask = (pred_flat == gt_flat)
        for k in range(K):
            class_mask = (gt_flat == k)
            area_intersection[k] = np.sum(correct_mask & class_mask)

        area_union = area_output + area_target - area_intersection
        return area_intersection, area_union, area_output, area_target

    def add_metrics(self, prediction, ground_truth):
        prediction = prediction.to(ground_truth.device)
        prediction = F.interpolate(prediction, ground_truth.shape[-2:], mode="bilinear", align_corners=True)
        if self._n_classes == 1:
            prediction = (torch.sigmoid(prediction).squeeze(1) >= 0.5).long().cpu().numpy()
        else:
            prediction = prediction.argmax(dim=1).cpu().numpy()
        ground_truth = ground_truth.cpu().long().numpy()

        inter, _, outA, gtA = self._calculate_iou(prediction, ground_truth, self._n_classes, ignored_index=0)
        self._intersection += inter
        self._prediction_num += outA
        self._target_num += gtA

    def get_metrics(self, _len):
        # 仅返回“最后一类”的 Dice（保留你的原逻辑）
        eps = 1e-8
        c = self._n_classes - 1
        dice = (2 * self._intersection[c]) / (self._prediction_num[c] + self._target_num[c] + eps)
        self.metrics['DICE'] += float(dice)
        return self.metrics


# ------------------------- 建议使用：统一多/二类指标 ------------------------- #
class MyMetrics:
    """
    统一的训练期指标收集器：
      - n_classes == 1: 计算二类总体指标（与旧版保持一致）
      - n_classes  > 1: 计算多类 per-class Dice/IoU 与 mDice/mIoU（忽略背景/ignore_index）
                        可选 compute_hd=True 时，再加每类 HD/HD95 的宏平均
    """
    def __init__(self,
                 max_save: bool = False,
                 n_classes: int = 1,
                 ignore_index: Optional[int] = 0,
                 compute_hd: bool = False,
                 **kwargs):
        self._n_classes = int(n_classes)
        self._ignore_index = ignore_index
        self._compute_hd = bool(compute_hd)

        # 输出字典（与旧接口字段名保持一部分一致）
        if self._n_classes == 1:
            self.metrics = {
                'DICE': 0., 'IOU': 0., 'ACC': 0., 'Precision': 0.,
                'HD': 0., 'HD95': 0., 'Recall': 0.,
            }
            self._preds_bin: List[np.ndarray] = []
            self._gts_bin:   List[np.ndarray] = []
            self._sigmoid = nn.Sigmoid()
        else:
            self.metrics = {
                'mDice': 0., 'mIoU': 0.,
                # 下面三个键是“可选/补充信息”，你可以不使用
                'dice_per_class': None,
                'iou_per_class':  None,
                'hd95_per_class': None,
            }
            self._mc = _MCStats(self._n_classes, ignore_index=self._ignore_index)

        self._max = 0. if max_save else None

    def reset_zero(self):
        improved = None
        if self._max is not None:
            key = 'DICE' if self._n_classes == 1 else 'mDice'
            improved = self._max < self.metrics[key]
            self._max = max(self._max, self.metrics[key])

        if self._n_classes == 1:
            for k in self.metrics: self.metrics[k] = 0.
            self._preds_bin.clear()
            self._gts_bin.clear()
        else:
            for k in self.metrics: self.metrics[k] = 0. if k in ('mDice','mIoU') else None
            self._mc = _MCStats(self._n_classes, ignore_index=self._ignore_index)

        if self._max is not None:
            return improved

    def add_metrics(self, prediction: torch.Tensor, ground_truth: torch.Tensor):
        """
        prediction: [B, C, h, w]（logits）
        ground_truth:
          - 二类: [B, H, W] 或 [B,1,H,W]，元素 {0,1}
          - 多类: [B, H, W]（Long），元素 {0..K-1}
        """
        # 对齐到标签大小
        prediction = F.interpolate(prediction, ground_truth.shape[-2:], mode="bilinear", align_corners=True)

        if self._n_classes == 1:
            # 二类：Sigmoid→0/1
            prob = torch.sigmoid(prediction).squeeze(1)
            pred_np = (prob >= 0.5).to(torch.int64).cpu().numpy()
            gt = ground_truth
            if gt.dim() == 4: gt = gt.squeeze(1)
            gt_np = gt.detach().cpu().to(torch.int64).numpy()
            # 统一到 {0,1}
            if gt_np.max() > 1:
                gt_np = (gt_np > (0.5*gt_np.max())).astype(np.int64)
            self._preds_bin.append(pred_np)
            self._gts_bin.append(gt_np)
        else:
            # 多类：argmax
            pred_np = prediction.argmax(dim=1).cpu().long().numpy()
            gt_np   = ground_truth.cpu().long().numpy()
            self._mc.add(pred_np, gt_np, compute_hd=self._compute_hd)

    def get_metrics(self, _len) -> Dict[str, float]:
        if self._n_classes == 1:
            # 聚合二类
            pred = np.concatenate(self._preds_bin, axis=0) if self._preds_bin else np.zeros((0,))
            gt   = np.concatenate(self._gts_bin,   axis=0) if self._gts_bin   else np.zeros((0,))
            if pred.size == 0 or gt.size == 0:
                return self.metrics

            self.metrics['DICE']      += metric.binary.dc(pred, gt)
            self.metrics['IOU']       += metric.binary.jc(pred, gt)
            self.metrics['HD']        += metric.binary.hd(pred, gt, voxelspacing=None, connectivity=1)
            self.metrics['HD95']      += metric.binary.hd95(pred, gt, voxelspacing=None, connectivity=1)
            self.metrics['Recall']    += metric.binary.recall(pred, gt)
            self.metrics['Precision'] += metric.binary.precision(pred, gt)
            self.metrics['ACC']       += segmentation_accuracy(pred, gt)
            return self.metrics
        else:
            # 聚合多类（忽略背景/ignore_index，只对在 GT 出现过的类求均值）
            s = self._mc.summarize()
            self.metrics['mDice'] = s['mDice']
            self.metrics['mIoU']  = s['mIoU']
            # 可选：把 per-class 信息也吐出去（便于外部记录）
            self.metrics['dice_per_class'] = s['dice_per_class']
            self.metrics['iou_per_class']  = s['iou_per_class']
            # 若统计过 HD95，这里提供（否则 None）
            if self._compute_hd and s['hd_cnt'].sum() > 0:
                # 只对有效类取均值
                valid = (s['hd_cnt'] > 0)
                if self._ignore_index is not None and 0 <= self._ignore_index < self._n_classes:
                    valid[self._ignore_index] = False
                if valid.any():
                    self.metrics['hd95_per_class'] = s['hd95_per_class']
            return self.metrics


# ------------------------- 简单 IoU 直方图工具（可选） ------------------------- #
def calculate_iou(prediction: np.ndarray, ground_truth: np.ndarray, K: int, ignored_index: Optional[int] = None):
    """
    兼容你原来的接口（仅供需要时使用）。
    返回各类的 (intersection, union, pred_area, gt_area)
    """
    assert prediction.shape == ground_truth.shape
    pred_flat = prediction.reshape(-1)
    gt_flat = ground_truth.reshape(-1)

    if ignored_index is not None:
        valid = (gt_flat != ignored_index)
        pred_flat = pred_flat[valid]
        gt_flat   = gt_flat[valid]

    area_intersection, _ = np.histogram(pred_flat[ pred_flat == gt_flat ], bins=np.arange(K+1))
    area_output, _       = np.histogram(pred_flat, bins=np.arange(K+1))
    area_target, _       = np.histogram(gt_flat,   bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_output, area_target


# ------------------------- 自检 ------------------------- #
if __name__ == "__main__":
    # 简单自检：二类
    B, H, W = 4, 64, 64
    logits_bin = torch.randn(B, 1, H, W)
    gt_bin = torch.randint(0, 2, (B, H, W))
    mm = MyMetrics(n_classes=1)
    mm.add_metrics(logits_bin, gt_bin)
    print("[Binary]", mm.get_metrics(1))

    # 多类
    K = 5
    logits_mc = torch.randn(B, K, H, W)
    gt_mc = torch.randint(0, K, (B, H, W))
    mm2 = MyMetrics(n_classes=K, ignore_index=0, compute_hd=False)
    mm2.add_metrics(logits_mc, gt_mc)
    out = mm2.get_metrics(1)
    print("[Multi]", {k: (out[k] if isinstance(out[k], (float,int)) else "array") for k in out})
