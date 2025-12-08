# -*- coding: utf-8 -*-
# dataset.py (Fixed Normalization)
from pathlib import Path
from typing import Optional, Tuple, Union, Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import pickle, gzip

# 可选依赖
try:
    from bloscpack import unpack_ndarray_from_file as _unpack_ndarray_from_file

    _HAS_BLOSC = True
except Exception:
    _HAS_BLOSC = False

try:
    import tifffile as tiff

    _HAS_TIFF = True
except Exception:
    _HAS_TIFF = False

# ---------- 文件格式嗅探 ----------
_MAGIC_NUMPY = b"\x93NUMPY"  # .npy
_MAGIC_ZIP = b"PK\x03\x04"  # .npz
_MAGIC_PICKLE = b"\x80"  # pickle
_MAGIC_BLOSC = b"blpk"  # bloscpack
_MAGIC_GZIP = b"\x1f\x8b"  # .gz
_MAGIC_TIFF_II = b"II*\x00"
_MAGIC_TIFF_MM = b"MM\x00*"


def _sniff_magic(path: Path, n: int = 8) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _find_with_ext(root: Path, stem: str, exts=None) -> Path:
    if exts is None:
        exts = (".blosc", ".blp", ".pkl", ".pkl.gz", ".npy", ".npz", ".tif", ".tiff", ".gz")
    for ext in exts:
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    g = list(root.glob(stem + ".*"))
    if g:
        return g[0]
    raise FileNotFoundError(f"未在 {root} 下找到 {stem}（尝试扩展：{exts}）")


# ---------- 解包辅助 ----------
def _unwrap_scalar_like(x: Any) -> Any:
    """0 维 ndarray / 标量包装对象 → .item() 解包；否则原样返回。"""
    if isinstance(x, np.ndarray) and x.ndim == 0:
        try:
            return x.item()
        except Exception:
            return x
    return x


# ---------- 从对象里“挑一个 ndarray” ----------
def _iter_ndarrays(obj: Any) -> Iterable[np.ndarray]:
    """在 dict/list/tuple/混合结构里，递归产出 ndarray 候选；支持 dtype=object 的数组向下展开。"""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for it in obj.flat:
                yield from _iter_ndarrays(it)
        else:
            yield obj
    elif isinstance(obj, (list, tuple)):
        for it in obj:
            yield from _iter_ndarrays(it)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_ndarrays(v)
    else:
        # 尝试 array 化：若变成 object 数组也继续向下展开
        try:
            arr = np.asarray(obj)
            if isinstance(arr, np.ndarray):
                if arr.dtype == object:
                    for it in arr.flat:
                        yield from _iter_ndarrays(it)
                elif arr.size > 0:
                    yield arr
        except Exception:
            pass


def _pick_array(obj: Any, role: str, expect_C: Optional[int]) -> np.ndarray:
    """
    role: 'hsi' 或 'label'
    选择策略：
      - 对 HSI：优先 3D；若有多个，优先满足 C==expect_C 或 size 最大者
      - 对 Label：优先 2D；若 3D 且最后一维为1则 squeeze；若是 one-hot，取 argmax
    允许输入为 dict/list/tuple 的嵌套结构
    """
    # 先把所有候选抓出来
    cands = list(_iter_ndarrays(obj))
    if not cands:
        raise ValueError("对象中未找到可用的 ndarray")

    def _score(a: np.ndarray) -> Tuple[int, int, int]:
        # 返回排序 key：匹配度高的排前
        if role == "hsi":
            dim_pref = 2 if a.ndim == 3 else (1 if a.ndim == 2 else 0)
            match_c = 1 if (expect_C is not None and (
                    (a.ndim == 3 and a.shape[2] == expect_C) or
                    (a.ndim == 3 and a.shape[0] == expect_C))) else 0
            size = a.size
            return (dim_pref, match_c, size)
        else:
            dim_pref = 2 if a.ndim == 2 else (1 if (a.ndim == 3 and (1 in a.shape)) else 0)
            size = a.size
            return (dim_pref, 0, size)

    # 选分最高的
    best = max(cands, key=_score)

    # 角色后处理
    if role == "hsi":
        a = best
        # (C,H,W) -> (H,W,C)（若 expect_C 可用，用它判断）
        if a.ndim == 3:
            HWC_like = (expect_C is None) or (a.shape[2] == expect_C)
            CHW_like = (expect_C is not None and a.shape[0] == expect_C)
            if CHW_like and not HWC_like:
                a = np.moveaxis(a, 0, -1)
        elif a.ndim == 2:
            a = a[..., None]
        return a.astype(np.float32, copy=False)

    else:  # label
        a = best
        if a.ndim == 3:
            # (H,W,1) → squeeze；(H,W,K) one-hot → argmax
            if a.shape[-1] == 1:
                a = a[..., 0]
            else:
                a = np.argmax(a, axis=-1)
        # 若仍不是 2D，尝试压掉为1的轴
        if a.ndim != 2:
            for axis in range(a.ndim):
                if a.shape[axis] == 1:
                    a = np.squeeze(a, axis=axis)
                    if a.ndim == 2:
                        break
            if a.ndim != 2:
                # 兜底：取最大投影
                a = np.argmax(a.reshape(a.shape[0], -1), axis=1) if a.ndim >= 2 else a
        return a.astype(np.int64, copy=False)


# ---------- 通用读取 ----------
def _read_array_any(path: Path, role: str, expect_C: Optional[int]) -> np.ndarray:
    """
    role='hsi' or 'label'
    识别格式 + 解析 dict/list/tuple，选出合适的 ndarray
    """
    head = _sniff_magic(path, 8)

    # bloscpack
    if head.startswith(_MAGIC_BLOSC):
        if not _HAS_BLOSC:
            raise ImportError("检测到 bloscpack，但未安装 bloscpack。pip install bloscpack")
        obj = _unpack_ndarray_from_file(str(path))
        obj = _unwrap_scalar_like(obj)
        return _pick_array(obj, role, expect_C)

    # .npy
    if head.startswith(_MAGIC_NUMPY):
        obj = np.load(str(path), allow_pickle=True)  # 允许 object
        obj = _unwrap_scalar_like(obj)
        return _pick_array(obj, role, expect_C)

    # .npz
    if head.startswith(_MAGIC_ZIP):
        with np.load(str(path), allow_pickle=True) as z:
            # 先按常见 key 选
            pref = (["hsi", "cube", "data", "image", "img", "X", "spectral", "arr_0", "array", "arr"]
                    if role == "hsi"
                    else ["label", "mask", "seg", "gt", "y", "lbl", "lab", "arr_0", "array", "arr"])
            key = None
            for k in pref:
                if k in z.files:
                    key = k;
                    break
            if key is None:
                key = max(z.files, key=lambda k: z[k].size)
            obj = z[key]
        obj = _unwrap_scalar_like(obj)
        return _pick_array(obj, role, expect_C)

    # gzip（常见 .pkl.gz）
    if head.startswith(_MAGIC_GZIP):
        with gzip.open(str(path), "rb") as f:
            obj = pickle.load(f)
        obj = _unwrap_scalar_like(obj)
        return _pick_array(obj, role, expect_C)

    # 纯 pickle
    if head.startswith(_MAGIC_PICKLE):
        with open(str(path), "rb") as f:
            obj = pickle.load(f)
        obj = _unwrap_scalar_like(obj)
        return _pick_array(obj, role, expect_C)

    # tiff
    if head.startswith(_MAGIC_TIFF_II) or head.startswith(_MAGIC_TIFF_MM):
        if not _HAS_TIFF:
            raise ImportError("检测到 tiff，但未安装 tifffile。pip install tifffile")
        obj = tiff.imread(str(path))
        obj = _unwrap_scalar_like(obj)
        return _pick_array(obj, role, expect_C)

    # 兜底：先 np.load，失败再 pickle
    try:
        obj = np.load(str(path), allow_pickle=True)
        if isinstance(obj, np.lib.npyio.NpzFile):
            obj = {k: obj[k] for k in obj.files}
        obj = _unwrap_scalar_like(obj)
        return _pick_array(obj, role, expect_C)
    except Exception:
        try:
            with open(str(path), "rb") as f:
                obj = pickle.load(f)
            obj = _unwrap_scalar_like(obj)
            return _pick_array(obj, role, expect_C)
        except Exception as e2:
            raise ValueError(f"无法识别或读取文件格式：{path}") from e2


# ---------- Dataset ----------
class HyperspectralDataset(Dataset):
    """
    读取成对的 HSI 立方体与分割标签（支持多种格式、能从 dict/list 里挑 ndarray）：
      - HSI: 返回 torch.float32 [C, H, W]（内部自动把 (C,H,W)/(H,W,C) 统一成 [C,H,W]）
      - Label: np.ndarray (H, W) 多类标签，0=背景

    列表文件每行一个 stem（不带扩展名）
    约定：
      - train=True  → (hsi_256, rgb_256, label_256)
      - train=False → (hsi_256, rgb_256, label_256, label_orig)
    """

    def __init__(
            self,
            hsi_dir: str,
            rgb_dir: str,  # 兼容旧签名，不使用
            label_dir: str,
            list_file: str,
            image_size: int = 256,
            transform=None,
            train: bool = True,
            use_soft_label: bool = False,  # 仅用于二类时；多类下忽略
            soft_kernel: int = 5,
            num_channels_hsi: Optional[int] = None,  # 若提供则做一致性检查/转置判断
            segm_key: str = "",  # 兼容旧签名，忽略
            multi_class: bool = True,  # 默认多类
            per_sample_minmax: bool = False,  # 是否对每个样本做 [0,1] 归一化
    ):
        self.hsi_root = Path(hsi_dir).resolve()
        self.lab_root = Path(label_dir).resolve()
        self.image_size = int(image_size)
        self.transform = transform
        self.train = bool(train)
        self.use_soft = bool(use_soft_label)
        self.soft_k = int(soft_kernel)
        self.expect_C = int(num_channels_hsi) if num_channels_hsi is not None else None
        self.multi_class = bool(multi_class)
        self.per_sample_minmax = bool(per_sample_minmax)

        if not self.hsi_root.is_dir() or not self.lab_root.is_dir():
            raise FileNotFoundError(f"未找到目录：{self.hsi_root} 或 {self.lab_root}")

        with open(list_file, "r", encoding="utf-8") as f:
            self.stems = [ln.strip() for ln in f if ln.strip()]
        if not self.stems:
            raise RuntimeError(f"清单为空：{list_file}")

        # 同步增广参数
        self._rrc_scale = (0.8, 1.0)
        self._rrc_ratio = (0.75, 1.33)

    def __len__(self):
        return len(self.stems)

    def _soften_label(self, lab: torch.Tensor) -> torch.Tensor:
        if self.soft_k <= 1:
            return lab
        k = self.soft_k
        x = lab.unsqueeze(0).unsqueeze(0).float()
        ker = torch.ones((1, 1, k, k), dtype=torch.float32) / (k * k)
        x = F.conv2d(x, ker, padding=k // 2)
        return x.squeeze(0).squeeze(0).clamp_(0, 1)

    def _maybe_minmax(self, t: torch.Tensor) -> torch.Tensor:
        """
        [修正] 全局归一化，保留光谱曲线形状
        """
        if not self.per_sample_minmax:
            return t

        vmin = t.min()
        vmax = t.max()
        denom = vmax - vmin

        # 防止除以零
        if denom < 1e-6:
            return torch.zeros_like(t)

        return (t - vmin) / denom

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        stem = Path(self.stems[idx]).stem

        # ---- 路径 ----
        hsi_path = _find_with_ext(self.hsi_root, stem)
        lab_path = _find_with_ext(self.lab_root, stem)

        # ---- 读取 HSI ----
        cube = _read_array_any(hsi_path, role="hsi", expect_C=self.expect_C)  # (H,W,C) or (H,W,1+)
        if cube.ndim != 3:
            raise ValueError(f"HSI 立方体应为3维 (H,W,C)，实际 {cube.shape} @ {hsi_path}")

        H, W, C = cube.shape
        if self.expect_C is not None and C != self.expect_C:
            raise ValueError(
                f"通道数不匹配：文件C={C} vs 期望C={self.expect_C}。"
                f"请在 Config.num_channels_hsi 中设为 {C}，或在 Dataset 中实现投影。"
            )

        # [C,H,W] float32
        hsi = torch.from_numpy(np.moveaxis(cube, -1, 0)).contiguous().float()

        # ---- 读取 Label：2D，多类 ----
        lab_np = _read_array_any(lab_path, role="label", expect_C=None)
        if lab_np.ndim != 2:
            raise ValueError(f"标签应为二维 (H,W)，实际 {lab_np.shape} @ {lab_path}")

        # [关键修复] 显式转为 uint8 以兼容 PIL (int64 会导致 TypeError)
        # 只要 max(label) <= 255，转 uint8 是安全的
        lab_np = lab_np.astype(np.uint8)

        # [可选] 不再强制将 255/254 转为 0，以便 train.py 可以使用 ignore_index=255
        # 如果你确定想忽略 255，就保持下面注释掉的状态
        # lab_np[lab_np == 255] = 0
        # lab_np[lab_np == 254] = 0

        # lab_orig: 原始标签，用于 eval 可视化/统计
        lab_orig = lab_np.copy()

        # 占位 RGB
        rgb = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8), mode="RGB")

        # ---- 同步增广（仅训练）----
        if self.train:
            # 水平翻转
            if torch.rand(1) > 0.5:
                rgb = TF.hflip(rgb)
                hsi = TF.hflip(hsi)
                lab_np = np.ascontiguousarray(np.fliplr(lab_np))
            # 垂直翻转
            if torch.rand(1) > 0.5:
                rgb = TF.vflip(rgb)
                hsi = TF.vflip(hsi)
                lab_np = np.ascontiguousarray(np.flipud(lab_np))

            # 随机旋转
            angle = float(torch.empty(1).uniform_(-15, 15).item())
            rgb = TF.rotate(rgb, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            hsi = TF.rotate(hsi, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)

            # 使用 'nearest' 旋转标签，保持 int 类别
            lab_img = Image.fromarray(lab_np)
            lab_img = TF.rotate(lab_img, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
            lab_np = np.array(lab_img)

            # RandomResizedCrop
            i, j, h_crop, w_crop = T.RandomResizedCrop.get_params(
                rgb, scale=self._rrc_scale, ratio=self._rrc_ratio
            )
            rgb = TF.crop(rgb, i, j, h_crop, w_crop)
            hsi = hsi[:, i:i + h_crop, j:j + h_crop]
            lab_np = lab_np[i:i + h_crop, j:j + h_crop]

        # ---- 统一 Resize 到网络输入尺寸 ----
        size = (self.image_size, self.image_size)
        rgb = TF.resize(rgb, size, interpolation=TF.InterpolationMode.BILINEAR)
        hsi = TF.resize(hsi, size, interpolation=TF.InterpolationMode.BILINEAR)

        # 掩码最近邻 (此时 lab_np 已经是 uint8，PIL 可以处理)
        lab_256 = np.array(
            Image.fromarray(lab_np).resize(size[::-1], Image.NEAREST)
        ).astype(np.int64, copy=False)

        # ---- 组装张量 ----
        hsi = self._maybe_minmax(hsi)

        rgb_t = TF.to_tensor(rgb)
        rgb_t = TF.normalize(rgb_t, mean=[0.5] * 3, std=[0.5] * 3)

        if self.multi_class:
            label_256_t = torch.from_numpy(lab_256)  # [H,W] int64 (0=背景/未标注，1..K=器官类)
        else:
            bin_np = (lab_256 > 0).astype(np.float32)
            label_256_t = torch.from_numpy(bin_np).unsqueeze(0)
            if self.train and self.use_soft:
                label_256_t = self._soften_label(label_256_t.squeeze(0)).unsqueeze(0)

        # 首个样本打印调试信息
        if idx == 0:
            head = _sniff_magic(hsi_path, 4)
            fmt = "blosc" if head.startswith(_MAGIC_BLOSC) else \
                "npy" if head.startswith(_MAGIC_NUMPY) else \
                    "npz/zip" if head.startswith(_MAGIC_ZIP) else \
                        "pickle(.gz)" if head.startswith(_MAGIC_GZIP) or head.startswith(_MAGIC_PICKLE) else \
                            "tiff" if (head.startswith(_MAGIC_TIFF_II) or head.startswith(
                                _MAGIC_TIFF_MM)) else "unknown"
            print(f"[{stem}] fmt={fmt} HSI {tuple(hsi.shape)} | label256 uniq={np.unique(lab_256)[:20]}")

        if self.train:
            return hsi, rgb_t, label_256_t
        else:
            label_orig_t = torch.from_numpy(lab_orig.astype(np.int64))
            return hsi, rgb_t, label_256_t, label_orig_t