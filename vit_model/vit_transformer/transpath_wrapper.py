# vit_model/vit_transformer/transpath_wrapper.py
from __future__ import annotations
import math
import importlib
import importlib.util
from typing import Optional, Tuple, List
from pathlib import Path
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransPathRGBEncoder(nn.Module):
    """
    使用 TransPath ViT 或 CTransPath 对 RGB 编码，产出多尺度特征：
      - 优先走骨干的 forward_features（若可用）
      - 若 forward_features 不返回空间特征，回退到 patch_embed→blocks→norm
      - 输出侧用 1×1 conv 做通道适配到解码器期望的通道数
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        hidden: int = 384,
        variant: str = "vit_small",
        out_channels_top: int = 256,
        skip_channels: Tuple[int, int, int] = (256, 256, 128),
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.variant = variant
        self.pretrained_path = pretrained_path
        self._pretrained_loaded = False
        self._is_ctrans = variant.startswith("ctrans")

        # 1) 构建骨干
        self.vit = self._build_backbone(variant, patch_size)

        # 2) 以骨干的 num_features（若无则 embed_dim）为“骨干输出通道”
        self.hidden = int(getattr(self.vit, "num_features", getattr(self.vit, "embed_dim", hidden)))

        # 3) 多尺度投影（输出侧 1×1 适配器）
        self.proj_top = nn.Conv2d(self.hidden, out_channels_top, 1)
        self.proj_s3  = nn.Conv2d(out_channels_top, skip_channels[0], 1)
        self.proj_s2  = nn.Conv2d(out_channels_top, skip_channels[1], 1)
        self.proj_s1  = nn.Conv2d(out_channels_top, skip_channels[2], 1)

        # 若骨干返回 fmap 通道 != self.hidden，首次前向时懒加载 1×1 适配
        self.fmap_adapter: Optional[nn.Conv2d] = None

    # ------------------------ 构建 & 工具 ------------------------
    def _import_module(self, name: str):
        """
        仅修复“路径问题”，保持功能不变：
        1) 先正常 import
        2) 失败则从本文件锚点查找 vit_model/external/transpath/{vits,ctran} 的真实 .py 并按文件路径加载
        3) 允许用环境变量显式指定：TRANSPATH_CTRAN / TRANSPATH_VITS
        """
        # 1) 标准导入
        try:
            return importlib.import_module(name)
        except Exception:
            pass

        # 2) 环境变量直指
        leaf = name.split(".")[-1].lower()
        if leaf in ("ctran", "vits"):
            env_key = "TRANSPATH_CTRAN" if leaf == "ctran" else "TRANSPATH_VITS"
            env_path = os.getenv(env_key, "").strip()
            if env_path:
                p = Path(env_path).expanduser().resolve()
                if p.is_file():
                    mod = self._load_module_from_path(f"transpath_{leaf}_env", p)
                    if mod:
                        return mod

        # 3) 相对本文件 __file__ 的标准位置 + 兼容大小写/别名
        here = Path(__file__).resolve()
        base = here.parents[1] / "external" / "transpath"
        candidates = []
        if leaf == "ctran":
            candidates = [
                base / "ctran.py",
                base / "CTran.py",
                base / "ctrans.py",
                base / "ctranspath.py",
            ]
        elif leaf == "vits":
            candidates = [
                base / "vits.py",
                base / "ViTs.py",
            ]
        for p in candidates:
            if p.is_file():
                mod = self._load_module_from_path(f"transpath_{leaf}_local", p)
                if mod:
                    return mod

        # 4) 兜底：向上找到包含 vit_model 的工程根，再搜一遍；最后全局搜 transpath 目录
        proj_root = None
        for par in here.parents:
            if (par / "vit_model").is_dir():
                proj_root = par
                break
        if proj_root:
            base2 = proj_root / "vit_model" / "external" / "transpath"
            patterns = []
            if leaf == "ctran":
                patterns = ["ctran.py", "CTran.py", "ctrans.py", "ctranspath.py"]
            elif leaf == "vits":
                patterns = ["vits.py", "ViTs.py"]
            for name2 in patterns:
                p = base2 / name2
                if p.is_file():
                    mod = self._load_module_from_path(f"transpath_{leaf}_root", p)
                    if mod:
                        return mod
            # transpath 下所有 .py 里挑包含 'tran' 的名称（尽量窄化搜索范围）
            g = list(base2.glob("*.py")) if base2.is_dir() else []
            g += list(proj_root.glob("**/transpath/*.py"))
            g = [x for x in g if "tran" in x.name.lower()]
            for p in g:
                if p.is_file():
                    mod = self._load_module_from_path(f"transpath_{leaf}_scan", p)
                    if mod and ((leaf == "vits" and "vit" in p.name.lower()) or (leaf == "ctran" and "tran" in p.name.lower())):
                        return mod

        # 保持原行为：返回 None，由调用方决定是否抛错
        return None

    def _load_module_from_path(self, module_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            return mod
        return None

    def _fix_imgsize_and_grid(self, vit):
        """仅同步 img_size / patch_size；ViT 可更新 grid，CTransPath 不改 grid。"""
        size = (self.img_size, self.img_size)
        ps = self.patch_size
        pe = getattr(vit, "patch_embed", None)

        if pe is not None:
            if hasattr(pe, "img_size"):
                pe.img_size = size
            if not self._is_ctrans:
                # 只有 ViT/DeiT 这类平铺式 patch embed 才安全地改 grid/num_patches
                if hasattr(pe, "grid_size"):
                    pe.grid_size = (size[0] // ps, size[1] // ps)
                if hasattr(pe, "num_patches"):
                    pe.num_patches = (size[0] // ps) * (size[1] // ps)

        if hasattr(vit, "img_size"):
            vit.img_size = size
        if hasattr(vit, "patch_size"):
            vit.patch_size = ps
        if not self._is_ctrans and hasattr(vit, "num_patches") and pe is not None and hasattr(pe, "num_patches"):
            vit.num_patches = pe.num_patches

        if hasattr(vit, "build_2d_sincos_position_embedding"):
            try:
                vit.build_2d_sincos_position_embedding()
            except Exception:
                pass

    def _build_backbone(self, variant: str, patch_size: int):
        vit_mod   = self._import_module("vit_model.external.transpath.vits")
        ctran_mod = self._import_module("vit_model.external.transpath.ctran")

        if vit_mod:
            print(f"[TransPath] vits module: {getattr(vit_mod, '__file__', '<?>')}")
        if ctran_mod:
            print(f"[TransPath] ctran module: {getattr(ctran_mod, '__file__', '<?>')}")

        if variant.startswith("vit"):
            synonyms = {"vit_small": ["vit_small", "deit_small_patch16_224"],
                        "vit_base" : ["vit_base",  "deit_base_patch16_224"]}
            candidates = synonyms.get(variant, [variant])
            if vit_mod is None:
                raise ImportError("vits.py 未找到：请放到 vit_model/external/transpath/vits.py")
            builders = [k for k in dir(vit_mod) if callable(getattr(vit_mod, k)) and k.startswith("vit")]
            if builders:
                print(f"[TransPath] vits builders detected: {builders}")
            last_err = None
            for name in candidates:
                if hasattr(vit_mod, name):
                    fn = getattr(vit_mod, name)
                    try:
                        model = fn(img_size=self.img_size, patch_size=patch_size)
                    except TypeError:
                        try:
                            model = fn(patch_size=patch_size)
                        except TypeError:
                            try:
                                model = fn()
                            except Exception as e:
                                last_err = e; continue
                    self._fix_imgsize_and_grid(model)
                    return model
            raise ImportError(f"找不到 ViT 构造：{candidates}；last_err={last_err}")

        if variant.startswith("ctrans"):
            if ctran_mod is None:
                raise ImportError("ctran.py 未找到：请放到 vit_model/external/transpath/ctran.py")
            candidates = [variant, variant.replace("ctrans_", "ctranspath_"),
                          "ctranspath", "build_ctranspath", "build_model"]
            detected = [k for k in dir(ctran_mod) if callable(getattr(ctran_mod, k)) and ("ctrans" in k or "build" in k)]
            if detected:
                print(f"[TransPath] ctran builders detected: {detected}")
            last_err = None
            for name in candidates:
                if hasattr(ctran_mod, name):
                    fn = getattr(ctran_mod, name)
                    try:
                        model = fn(img_size=self.img_size, patch_size=patch_size)
                    except TypeError:
                        try:
                            model = fn(patch_size=patch_size)
                        except TypeError:
                            try:
                                model = fn()
                            except Exception as e:
                                last_err = e; continue
                    self._fix_imgsize_and_grid(model)
                    return model
            raise ImportError(f"找不到 CTrans 构造：{candidates}；last_err={last_err}")

        if vit_mod and hasattr(vit_mod, "vit_small"):
            model = getattr(vit_mod, "vit_small")(patch_size=patch_size)
            self._fix_imgsize_and_grid(model)
            return model
        raise ImportError("既没有可用的 vits.py 也没有 ctran.py 构造器。")

    def _resize_pos_embed(self, pe: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if pe.ndim != 3:
            return pe
        _, N, C = pe.shape
        h_new, w_new = H // self.patch_size, W // self.patch_size
        has_cls = (N == h_new * w_new + 1) or (N > h_new * w_new)
        cls, grid = (pe[:, :1, :], pe[:, 1:, :]) if has_cls else (None, pe)
        N_grid = grid.shape[1]
        s = int(math.sqrt(N_grid)) if int(math.sqrt(N_grid))**2 == N_grid else None
        if s is None:
            return pe
        grid_map = grid.transpose(1, 2).reshape(1, C, s, s)
        grid_map = F.interpolate(grid_map, size=(h_new, w_new), mode="bilinear", align_corners=False)
        grid_new = grid_map.flatten(2).transpose(1, 2)
        return torch.cat([cls, grid_new], dim=1) if cls is not None else grid_new

    def _maybe_load_pretrained(self, H: int, W: int, device: torch.device):
        if self._pretrained_loaded or not self.pretrained_path:
            return
        sd = torch.load(self.pretrained_path, map_location="cpu")
        if isinstance(sd, dict):
            for k in ["state_dict", "model", "net", "backbone"]:
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]; break
        if not isinstance(sd, dict):
            raise RuntimeError("Checkpoint format not supported")

        # strip prefix
        clean = {}
        for k, v in sd.items():
            kk = k
            for p in ("module.", "backbone.", "encoder.", "model."):
                if kk.startswith(p):
                    kk = kk[len(p):]
            clean[kk] = v

        # drop heads
        for k in list(clean.keys()):
            if any(x in k for x in ("head", "fc", "classifier", "mlp_head", "proj_out")):
                clean.pop(k, None)

        model_sd = self.vit.state_dict()
        if "pos_embed" in clean and "pos_embed" in model_sd and clean["pos_embed"].ndim == 3:
            clean["pos_embed"] = self._resize_pos_embed(clean["pos_embed"], H, W)

        loadable = {k: v for k, v in clean.items() if k in model_sd and v.shape == model_sd[k].shape}
        missing = [k for k in model_sd.keys() if k not in loadable]
        unexpected = [k for k in clean.keys() if k not in model_sd]
        print(f"[TransPath] load: {len(loadable)} ok, {len(missing)} missing, {len(unexpected)} unexpected")
        self.vit.load_state_dict(loadable, strict=False)
        self._pretrained_loaded = True
        self.to(device)

    # ------------------------ 前向 ------------------------
    def _forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        优先用 vit.forward_features 获取**完整骨干输出**：
          - 若返回 [B,C,h,w]：直接转 tokens
          - 若返回 [B,N,C] ：用 patch 网格还原 h,w（若不匹配则用 sqrt(N) 兜底）
          - 若返回 [B,C]   ：说明被全局池化，回退到 patch_embed 路径
        回退路径：patch_embed → (cls/pos/drop) → blocks(*) → norm(*)  (*仅当存在时)
        """
        vit = self.vit
        H0, W0 = x.shape[-2:]

        # 1) 尝试骨干的 forward_features
        if hasattr(vit, "forward_features"):
            try:
                y = vit.forward_features(x)
                if isinstance(y, (tuple, list)):
                    y = next((t for t in y if isinstance(t, torch.Tensor)), y[0])
                if isinstance(y, torch.Tensor):
                    if y.dim() == 4:
                        B, C, h, w = y.shape
                        tokens = y.flatten(2).transpose(1, 2)
                        return tokens, h, w
                    if y.dim() == 3:
                        tokens = y
                        N = tokens.shape[1]
                        # 先用 grid_size，如果不匹配再回退 sqrt(N)
                        if hasattr(vit, "patch_embed") and hasattr(vit.patch_embed, "grid_size"):
                            gh, gw = vit.patch_embed.grid_size
                        else:
                            gh, gw = H0 // self.patch_size, W0 // self.patch_size
                        if gh * gw != N:
                            s = int(round(N ** 0.5))
                            gh = gw = s
                        return tokens, gh, gw
            except Exception:
                pass  # 出错走回退路径

        # 2) 回退：patch_embed → blocks → norm
        out = vit.patch_embed(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 4:
            B, C, h, w = out.shape
            tokens = out.flatten(2).transpose(1, 2)
        elif out.dim() == 3:
            tokens = out
            N = tokens.shape[1]
            if hasattr(vit, "patch_embed") and hasattr(vit.patch_embed, "grid_size"):
                h, w = vit.patch_embed.grid_size
            else:
                s = int(round(N ** 0.5))
                h = w = s
            # ★ 关键修正：若 h*w 与 N 不一致，改用 sqrt(N)
            if h * w != N:
                s = int(round(N ** 0.5))
                h = w = s
        else:
            raise ValueError(f"Unsupported patch_embed output dim={out.dim()}")

        for blk in getattr(vit, "blocks", []):
            tokens = blk(tokens)

        if hasattr(vit, "norm") and getattr(vit, "norm") is not None:
            last_dim = tokens.shape[-1]
            ns = getattr(vit, "norm", None)
            ns_val = getattr(ns, "normalized_shape", None)
            match = False
            if isinstance(ns_val, (tuple, list)) and len(ns_val) > 0:
                match = (ns_val[-1] == last_dim)
            elif isinstance(ns_val, int):
                match = (ns_val == last_dim)
            if match:
                tokens = vit.norm(tokens)
            else:
                tokens = F.layer_norm(tokens, (last_dim,), eps=1e-6)

        return tokens, h, w

    def forward(self, x_rgb: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = x_rgb.shape
        if (H % self.patch_size) or (W % self.patch_size):
            raise ValueError(f"Input {H}x{W} must be divisible by patch_size={self.patch_size}")

        # 加载预训练（在已知 H,W 后）
        self._maybe_load_pretrained(H, W, x_rgb.device)

        # tokens → fmap
        tokens, h, w = self._forward_tokens(x_rgb)     # [B, N, hidden?]
        fmap = tokens.transpose(1, 2).reshape(B, tokens.shape[-1], h, w)

        # 若骨干返回通道 != self.hidden（num_features），首次懒加载一个 1×1 adapter 修正
        if fmap.shape[1] != self.hidden:
            if self.fmap_adapter is None:
                self.fmap_adapter = nn.Conv2d(fmap.shape[1], self.hidden, kernel_size=1, bias=False).to(fmap.device)
                nn.init.kaiming_normal_(self.fmap_adapter.weight, mode="fan_out", nonlinearity="relu")
            fmap = self.fmap_adapter(fmap)

        # 顶层 + 三个 skip（上采样得到更高分辨率特征）
        top = self.proj_top(fmap)  # [B,256,h,w]
        s3  = self.proj_s3(top)
        s2  = self.proj_s2(F.interpolate(top, scale_factor=2, mode="bilinear", align_corners=False))
        s1  = self.proj_s1(F.interpolate(top, scale_factor=4, mode="bilinear", align_corners=False))
        return [top, s3, s2, s1]
