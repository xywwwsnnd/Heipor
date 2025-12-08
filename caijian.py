# caijian.py
# -*- coding: utf-8 -*-
import argparse, pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Iterable, Union
import numpy as np

# ---------------- blosc helpers ----------------
def blosc_decompress(buf: bytes) -> Optional[bytes]:
    try:
        import blosc2
        return blosc2.decompress(buf)
    except Exception:
        try:
            import blosc
            return blosc.decompress(buf)
        except Exception:
            return None

def blosc_compress(buf: bytes, typesize: int = 1) -> bytes:
    try:
        import blosc2
        return blosc2.compress(buf)  # zstd
    except Exception:
        try:
            import blosc
            return blosc.compress(buf, cname="zstd", clevel=7, typesize=typesize)
        except Exception:
            return buf

# -------------- debug helpers ------------------
def _echo_tree(tag: str, obj: Any, max_depth: int = 2, prefix: str = ""):
    def _shape_of(x): return getattr(x, "shape", None)
    if max_depth < 0: return
    t = type(obj).__name__
    sh = _shape_of(obj)
    info = f"{prefix}{tag}: type={t}"
    if sh is not None: info += f", shape={tuple(sh)}, dtype={getattr(obj,'dtype',None)}"
    print(info)
    if isinstance(obj, dict) and max_depth > 0:
        for k, v in list(obj.items())[:12]:
            _echo_tree(f"[{k!s}]", v, max_depth-1, prefix + "  ")
    if isinstance(obj, (list, tuple)) and max_depth > 0:
        for i, v in enumerate(list(obj)[:12]):
            _echo_tree(f"[{i}]", v, max_depth-1, prefix + "  ")

# ---------- read/write: two-part .blosc ----------
def _read_blosc_header_payload(path: Path, echo_header: bool = False) -> Tuple[Tuple[int, ...], np.dtype, bytes]:
    import numpy.lib.format as npfmt
    with open(path, "rb") as f:
        magic = f.read(6); f.seek(0)
        if magic.startswith(b"\x93NUMPY"):
            # np.save(header_tuple) + payload
            header_obj = npfmt.read_array(f, allow_pickle=True)
            try: header = header_obj.item()
            except Exception: header = header_obj
        else:
            # pickle header + payload
            header = pickle.load(f)

        # 解析 header（可能并不是数组头，而是标注字典；那会在上层兜底）
        shape, dtype = None, None
        if isinstance(header, dict):
            shape = header.get("shape") or header.get("dims") or header.get("sh")
            dt    = header.get("dtype") or header.get("dt")
            dtype = (np.dtype(dt) if dt is not None else None)
        elif isinstance(header, (list, tuple)) and len(header) >= 2:
            shape, dt = header[0], header[1]
            try: dtype = np.dtype(dt)
            except Exception: dtype = None

        if echo_header:
            print(f"[BL-HEAD] {path.name}: shape={shape}, dtype={dtype}")

        payload = f.read()
        if shape is None or dtype is None:
            # 不是数组头（多半是标注字典头），交给上层特殊处理
            raise ValueError("Header has no (shape,dtype)")
        shape = tuple(int(x) for x in (list(shape) if not isinstance(shape, tuple) else shape))
        return shape, dtype, payload

def _reconstruct_from_payload(shape: Tuple[int, ...], dtype: np.dtype, payload: bytes) -> np.ndarray:
    raw = blosc_decompress(payload) or payload
    need = int(np.prod(shape)) * dtype.itemsize
    if len(raw) < need:
        raise ValueError(f"payload too small: {len(raw)} < {need}")
    return np.frombuffer(raw[:need], dtype=dtype).reshape(shape)

def write_blosc_array_2part(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        header = (arr.shape, arr.dtype)
        pickle.dump(header, f, protocol=4)
        payload = blosc_compress(arr.tobytes(order="C"), typesize=arr.dtype.itemsize)
        f.write(payload)

# ----------------- generic readers -----------------
def read_hsi_file(path: Path, echo_struct: bool = False) -> np.ndarray:
    # .blosc 优先走两段式
    if path.suffix.lower() == ".blosc":
        try:
            shape, dtype, payload = _read_blosc_header_payload(path, echo_header=echo_struct)
            arr = _reconstruct_from_payload(shape, dtype, payload)
            if arr.ndim == 2: arr = arr[..., None]
            return arr
        except Exception:
            # 退路：整个文件为（压缩的）pickle对象
            data = path.read_bytes()
            obj = None
            try:
                obj = pickle.loads(data)
            except Exception:
                dec = blosc_decompress(data)
                if dec is None: raise
                obj = pickle.loads(dec)
            if echo_struct:
                print(f"[HSI-STRUCT] {path.name}"); _echo_tree("root", obj, 2)
            if isinstance(obj, np.ndarray):
                return obj if obj.ndim == 3 else obj[..., None]
            # 尝试从容器重建
            raise ValueError(f"{path} contains no usable HSI array; type={type(obj)}.")

    suf = path.suffix.lower()
    if suf == ".npz":
        z = np.load(path, allow_pickle=True)
        key = "cube" if "cube" in z.files else z.files[0]
        arr = z[key];  return arr if arr.ndim == 3 else arr[..., None]
    if suf == ".npy":
        obj = np.load(path, allow_pickle=True)
        arr = obj if isinstance(obj, np.ndarray) else np.asarray(obj)
        return arr if arr.ndim == 3 else arr[..., None]
    if suf in (".tif", ".tiff"):
        import tifffile as tiff
        arr = tiff.imread(str(path))
        return arr if arr.ndim == 3 else arr[..., None]
    if suf == ".dat" or path.name.endswith("_SpecCube.dat"):
        return np.fromfile(path, np.float32).reshape(480, 640, -1)
    raise ValueError(f"Unsupported HSI file: {path}")

# ---------------- polygon → mask ----------------
def _guess_xy(points: np.ndarray, H: int, W: int) -> np.ndarray:
    """把多边形顶点统一成 (x,y)（列,行）。"""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("polygon points must be Nx2")
    max0, max1 = pts[:,0].max(), pts[:,1].max()
    # 如果第一列像是在宽度范围内而第二列像是在高度范围内，按 (x,y)
    xy_ok  = (max0 <= W+1) and (max1 <= H+1)
    yx_ok  = (max0 <= H+1) and (max1 <= W+1)
    if xy_ok and not yx_ok:
        return pts
    if yx_ok and not xy_ok:
        return pts[:, ::-1]
    # 都OK或都不OK：优先按 (x,y)
    return pts

def _rasterize_with_pil(polys: List[Tuple[np.ndarray, int]], H: int, W: int, keep_raw: bool) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        raise RuntimeError("PIL (Pillow) is required for polygon rasterization") from e
    mode = "I" if keep_raw else "I"  # 32-bit int
    mask = Image.new(mode, (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for pts, cls_id in polys:
        if len(pts) < 3: continue
        xy = _guess_xy(pts, H, W)
        # 裁剪到图像范围
        xy[:,0] = np.clip(xy[:,0], 0, W-1)
        xy[:,1] = np.clip(xy[:,1], 0, H-1)
        draw.polygon(list(map(tuple, xy.tolist())), outline=int(cls_id), fill=int(cls_id))
    return np.asarray(mask, dtype=np.int32)

def _flatten_polygons(geom: Any) -> List[np.ndarray]:
    """把各种嵌套(list/tuple/ndarray)的几何收集为多个 Nx2 的多边形列表（不处理洞）。"""
    out: List[np.ndarray] = []
    def rec(x):
        if isinstance(x, np.ndarray) and x.ndim==2 and x.shape[1]==2:
            out.append(x.astype(float))
            return
        if isinstance(x, (list, tuple)):
            # 一层层展开
            if len(x)>0 and isinstance(x[0], (list, tuple, np.ndarray)):
                for y in x: rec(y)
                return
            # 可能是扁平坐标 [x0,y0,x1,y1,...]
            arr = np.asarray(x, dtype=float)
            if arr.ndim==1 and arr.size%2==0:
                out.append(arr.reshape(-1,2))
            elif arr.ndim==2 and arr.shape[1]==2:
                out.append(arr.astype(float))
    rec(geom)
    return out

def _polygon_dict_to_mask(obj: Dict[str, Any], H: int, W: int,
                          annotator: Optional[str], binary: bool, echo: bool) -> np.ndarray:
    """
    期望的结构（从你的打印看）：
      { 'polygon#annotator1': (geom, dtype_like_uint8, class_or_meta),
        'polygon#annotator2': (...), ... }
    我们只取一个标注者，默认 annotator1；若指定找不到，就取第一个 'polygon#' 键。
    """
    # 选 key
    cand_keys = [k for k in obj.keys() if isinstance(k, str) and k.startswith("polygon#")]
    if not cand_keys:
        raise ValueError("No polygon#annotator* keys in label dict.")
    key_pick = None
    if annotator:
        target = f"polygon#{annotator}"
        for k in cand_keys:
            if k == target:
                key_pick = k; break
    if key_pick is None:
        key_pick = cand_keys[0]
    val = obj[key_pick]
    if echo:
        print(f"[Label] use '{key_pick}'")

    # 解析为 (polygon_pts, class_id) 列表
    polys: List[Tuple[np.ndarray, int]] = []
    cls_global = 1  # 缺省类=1（二分类）
    if isinstance(val, (list, tuple)):
        # 最常见： (geom, something, maybe_class)
        geom = val[0] if len(val)>=1 else None
        maybe_cls = val[2] if len(val)>=3 else None
        if isinstance(maybe_cls, (int, np.integer)):
            cls_global = int(maybe_cls)
        if geom is None:
            raise ValueError("polygon tuple has no geometry")
        for p in _flatten_polygons(geom):
            cls_id = (1 if binary else cls_global if cls_global>0 else 0)
            polys.append((p, cls_id))
    elif isinstance(val, dict):
        # 另一种：每个子键是一个 polygon
        for _, item in val.items():
            geom = None; cls_id = cls_global
            if isinstance(item, (list, tuple)):
                geom = item[0] if len(item)>=1 else None
                maybe_cls = item[2] if len(item)>=3 else None
                if isinstance(maybe_cls, (int, np.integer)):
                    cls_id = int(maybe_cls)
            else:
                geom = item
            for p in _flatten_polygons(geom):
                polys.append((p, (1 if binary else (cls_id if cls_id>0 else 0))))
    else:
        # 直接就是 geom
        for p in _flatten_polygons(val):
            polys.append((p, 1 if binary else cls_global))

    if not polys:
        # 没多边形就全0
        return np.zeros((H, W), dtype=np.int32)

    return _rasterize_with_pil(polys, H, W, keep_raw=not binary)

# ---------------- label reader ----------------
def read_label_file(path: Path,
                    hw: Tuple[int,int],
                    label_key: Optional[str] = None,
                    annotator: Optional[str] = None,
                    binary: bool = True,
                    echo_keys: bool = False) -> np.ndarray:
    H, W = hw
    if path.suffix.lower() == ".blosc":
        # 1) 先尝试两段式数组（若是直接掩码）
        try:
            shape, dt, payload = _read_blosc_header_payload(path, echo_header=echo_keys)
            arr = _reconstruct_from_payload(shape, dt, payload)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[...,0]
            if arr.ndim != 2:
                raise ValueError("label array must be 2D")
            return np.asarray(arr)
        except Exception:
            pass
        # 2) 退路：整个文件为（压缩的）pickle对象（标注字典）
        data = path.read_bytes()
        obj = None
        try:
            obj = pickle.loads(data)
        except Exception:
            dec = blosc_decompress(data)
            if dec is None: raise
            obj = pickle.loads(dec)
        if echo_keys:
            print(f"[Label-STRUCT] {path.name}"); _echo_tree("root", obj, 2)

        # 若是字典且带 polygon#annotator*，转成掩码
        if isinstance(obj, dict) and any(isinstance(k,str) and k.startswith("polygon#") for k in obj.keys()):
            return _polygon_dict_to_mask(obj, H, W, annotator, binary, echo_keys)

        # 若是字典里就有现成掩码
        if isinstance(obj, dict):
            if label_key and label_key in obj:
                return np.asarray(obj[label_key])
            # 优先找整型二维数组
            for v in obj.values():
                if isinstance(v, np.ndarray) and v.ndim==2 and v.dtype.kind in ("i","u","b"):
                    return np.asarray(v)
            for v in obj.values():
                if isinstance(v, np.ndarray) and v.ndim==2:
                    return np.asarray(v)

        # 其他结构：找不到可用掩码
        raise ValueError("No usable label array in .blosc (expect polygon dict or 2D mask).")

    suf = path.suffix.lower()
    if suf == ".npz":
        z = np.load(path, allow_pickle=True)
        key = "label" if "label" in z.files else z.files[0]
        return np.asarray(z[key])
    if suf == ".npy":
        return np.asarray(np.load(path, allow_pickle=True))
    if suf in (".tif", ".tiff"):
        import tifffile as tiff
        return np.asarray(tiff.imread(str(path)))
    if suf in (".png", ".jpg", ".jpeg", ".bmp"):
        from PIL import Image
        return np.asarray(Image.open(path))
    raise ValueError(f"Unsupported label file: {path}")

# ---------------- finding & tiling ----------------
def _find(root: Path, token: str, suffix: str = "", strict_suffix: bool = False,
          postfixes: Tuple[str, ...] = tuple()) -> Path:
    cand: List[Path] = []
    name = token + (suffix or "")
    cand.append(root / name)
    for pf in postfixes:
        cand.append(root / (token + pf + (suffix or "")))
    if not strict_suffix and not suffix:
        COMMON = (".blosc", ".npz", ".npy", ".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg")
        for e in COMMON:
            cand.append(root / f"{token}{e}")
            for pf in postfixes:
                cand.append(root / f"{token}{pf}{e}")
        cand.extend((root).glob(token + ".*"))
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find file for token='{token}' under root='{root}'. "
        f"Tried default_suffix='{suffix}' (strict={strict_suffix}) and postfixes={postfixes}"
    )

def axis_positions(L: int, patch: int, stride: int) -> List[int]:
    pos = {0, max(0, L - patch)}
    x = 0
    while x <= L - patch:
        pos.add(x); x += stride
    return sorted(pos)

def process_split(
    split: str,
    stems: List[str],
    hsi_dir: Path,
    lab_dir: Path,
    out_root: Path,
    patch: int,
    stride: int,
    hsi_suffix: str,
    label_suffix: str,
    label_postfixes: Tuple[str, ...],
    strict_suffix: bool,
    label_key: Optional[str],
    annotator: Optional[str],
    binary_mask: bool,
    echo_first_n_keys: int = 0,
    echo_hsi_struct_n: int = 0
):
    out_hsi = out_root / split / "hsi"
    out_lab = out_root / split / "label"
    out_hsi.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)
    patch_list = []

    for i, s in enumerate(stems, 1):
        tok = Path(s).stem
        hsi_path = _find(hsi_dir, tok, hsi_suffix, strict_suffix, tuple())
        lab_path = _find(lab_dir, tok, label_suffix, strict_suffix, label_postfixes)

        echo_hsi = echo_hsi_struct_n > 0 and i <= echo_hsi_struct_n
        echo_lab = echo_first_n_keys > 0 and i <= echo_first_n_keys

        arr = read_hsi_file(hsi_path, echo_struct=echo_hsi)  # (H,W,C)
        # (C,H,W) -> (H,W,C)
        if arr.ndim == 3 and arr.shape[0] <= 64 and arr.shape[-1] > 64:
            arr = np.moveaxis(arr, 0, -1)
        H, W = int(arr.shape[0]), int(arr.shape[1])

        lab = read_label_file(lab_path, (H, W), label_key=label_key, annotator=annotator,
                              binary=binary_mask, echo_keys=echo_lab)

        if lab.shape[:2] != (H, W):
            raise ValueError(f"[SizeMismatch] {tok}: HSI {arr.shape[:2]} vs LABEL {lab.shape[:2]}")

        ys = axis_positions(H, patch, stride)
        xs = axis_positions(W, patch, stride)
        for y in ys:
            for x in xs:
                hsi_tile = arr[y:y+patch, x:x+patch, :]
                lab_tile = lab[y:y+patch, x:x+patch]
                tile_stem = f"{tok}_y{y:04d}x{x:04d}"
                write_blosc_array_2part(out_hsi / f"{tile_stem}.blosc",
                                        hsi_tile.astype(arr.dtype, copy=False))
                write_blosc_array_2part(out_lab / f"{tile_stem}.blosc",
                                        lab_tile.astype(np.int32, copy=False))
                patch_list.append(tile_stem)

        if i % 50 == 0 or i == len(stems):
            print(f"[{split}] {i}/{len(stems)} processed…")

    with open(out_root / f"{split}_patches.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(patch_list))
    print(f"[{split}] done → {(out_root / f'{split}_patches.txt')} ({len(patch_list)} patches)")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Make NxN patches for HeiPorSpectral (.blosc; polygon labels → masks).")
    ap.add_argument("--hsi-dir", required=True)
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--test-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=224)

    ap.add_argument("--hsi-suffix", default=".blosc")
    ap.add_argument("--label-suffix", default=".blosc")
    ap.add_argument("--label-postfix", nargs="*", default=["_mask", "_label", "_gt", "_seg"])
    ap.add_argument("--strict-suffix", action="store_true")

    ap.add_argument("--label-key", default="", help="若 .blosc 里是dict且已含2D掩码，则指定键名；留空自动选择")
    ap.add_argument("--label-annotator", default="annotator1",
                    help="选哪个标注者：annotator1/annotator2/annotator3/first")
    ap.add_argument("--label-binary", action="store_true",
                    help="输出二分类掩码（前景=1, 背景=0）。默认开启。用 --no-label-binary 关闭。")
    ap.add_argument("--no-label-binary", dest="label_binary", action="store_false")

    ap.add_argument("--echo-first-n-keys", type=int, default=3, help="前N个样本打印 标签结构/头信息")
    ap.add_argument("--echo-hsi-struct", type=int, default=3, help="前N个样本打印 HSI 头信息")
    args = ap.parse_args()

    hsi_dir = Path(args.hsi_dir); lab_dir = Path(args.label_dir); out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    def load_list(p: str) -> List[str]:
        with open(p, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    train_stems = load_list(args.train_list)
    test_stems  = load_list(args.test_list)

    print(f"[CFG] hsi_dir={hsi_dir}")
    print(f"[CFG] lab_dir={lab_dir}")
    print(f"[CFG] out={out_root}")
    print(f"[CFG] patch={args.patch} stride={args.stride}")
    print(f"[CFG] train={len(train_stems)} items, test={len(test_stems)} items")
    print(f"[CFG] hsi_suffix={args.hsi_suffix}  label_suffix={args.label_suffix}")
    print(f"[CFG] label_postfixes={tuple(args.label_postfix)} strict_suffix={args.strict_suffix}")
    print(f"[CFG] annotator={args.label_annotator} binary={args.label_binary}")
    if args.label_key:
        print(f"[CFG] label_key='{args.label_key}'")

    # annotator 选择
    anno = None if args.label_annotator=="first" else args.label_annotator

    process_split(
        "train", train_stems, hsi_dir, lab_dir, out_root, args.patch, args.stride,
        args.hsi_suffix, args.label_suffix, tuple(args.label_postfix),
        args.strict_suffix, args.label_key or None, anno, args.label_binary,
        args.echo_first_n_keys, args.echo_hsi_struct
    )
    process_split(
        "test", test_stems, hsi_dir, lab_dir, out_root, args.patch, args.stride,
        args.hsi_suffix, args.label_suffix, tuple(args.label_postfix),
        args.strict_suffix, args.label_key or None, anno, args.label_binary,
        args.echo_first_n_keys, args.echo_hsi_struct
    )

if __name__ == "__main__":
    main()
