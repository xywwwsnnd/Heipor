# detect_hsi_format.py
import sys, os, struct, io, traceback
from pathlib import Path

# 可选依赖：若不存在也不报错，后面会降级探测
try:
    import numpy as np
except Exception:
    np = None

try:
    import zipfile
except Exception:
    zipfile = None

def read_head(path, n=32):
    with open(path, "rb") as f:
        return f.read(n)

def is_npy(head):
    # .npy 魔数: 0x93 'NUMPY'
    return head.startswith(b"\x93NUMPY")

def npy_header(path):
    """尽量不加载大数组，仅解析头部。"""
    if np is None:
        return None
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            return None
        v_major, v_minor = f.read(1)[0], f.read(1)[0]
        if (v_major, v_minor) == (1, 0):
            hlen = struct.unpack("<H", f.read(2))[0]
        else:
            # v2.0 / v3.0
            hlen = struct.unpack("<I", f.read(4))[0]
        header = f.read(hlen).decode("latin1")
        # 粗解析（官方是 eval-like 的 dict 字串）
        # 兼容 dtype/fortran_order/shape 三键
        info = {"version": f"{v_major}.{v_minor}", "raw": header}
        for k in ("descr", "fortran_order", "shape"):
            i = header.find(k)
            if i >= 0:
                j = header.find(",", i)
                info[k] = header[i:j].strip()
        return info

def is_zip(head):
    # ZIP/NPZ 魔数: 'PK\x03\x04'
    return head.startswith(b"PK\x03\x04")

def probe_npz(path):
    if zipfile is None:
        return {"kind": "zip/npz (zipfile unavailable)"}
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        out = {"kind": "npz" if any(n.endswith(".npy") for n in names) else "zip",
               "members": names[:10] + (["..."] if len(names) > 10 else [])}
        # 尝试读第一个 .npy 的头
        if any(n.endswith(".npy") for n in names) and "numpy" in sys.modules:
            for n in names:
                if n.endswith(".npy"):
                    with zf.open(n) as f:
                        head = f.read(32)
                    if is_npy(head):
                        info = npy_header_from_bytes(path, n)
                        if info:
                            out["first_npy_header"] = info
                    break
        return out

def npy_header_from_bytes(zip_path, member):
    """从 zip/npz 内部成员快速解析 npy 头（非必需）。"""
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(member) as f:
                # 复制到 BytesIO 解析
                data = f.read()
        # 直接用 numpy 加载 header 信息（会加载数据，体积小通常可接受）
        import numpy as np
        with io.BytesIO(data) as bio:
            arr = np.load(bio, allow_pickle=False, mmap_mode=None)
        return {"shape": tuple(arr.shape), "dtype": str(arr.dtype)}
    except Exception:
        return None

def try_blosc2(path):
    try:
        import blosc2
    except Exception:
        return {"kind": "blosc2", "ok": False, "reason": "blosc2 not installed"}
    try:
        s = blosc2.open(path, mode="r")
        # 轻取少量元素确认可读
        obj = s[:]
        kind = "blosc2"
        # 有些 blosc2 文件其实存的是对象（例如 tuple）
        shape = getattr(obj, "shape", None)
        if shape is None and "numpy" in sys.modules:
            arr = extract_first_array(obj)
            shape = getattr(arr, "shape", None)
            dt = str(getattr(arr, "dtype", ""))
            return {"kind": kind, "ok": True, "container": type(obj).__name__, "shape": shape, "dtype": dt}
        dt = str(getattr(obj, "dtype", ""))
        return {"kind": kind, "ok": True, "shape": shape, "dtype": dt}
    except Exception as e:
        return {"kind": "blosc2", "ok": False, "reason": str(e)}

def try_bloscpack(path):
    try:
        from bloscpack import unpack_ndarray_from_file
    except Exception:
        return {"kind": "bloscpack", "ok": False, "reason": "bloscpack not installed"}
    try:
        arr = unpack_ndarray_from_file(path)
        return {"kind": "bloscpack", "ok": True, "shape": tuple(arr.shape), "dtype": str(arr.dtype)}
    except Exception as e:
        return {"kind": "bloscpack", "ok": False, "reason": str(e)}

def extract_first_array(obj):
    """从 tuple/list/dict/0维 object 中递归取出第一个 numpy.ndarray。"""
    if np is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, (list, tuple)):
        for it in obj:
            a = extract_first_array(it)
            if a is not None:
                return a
        return None
    if isinstance(obj, dict):
        for k in ("cube", "hsi", "data", "array", "X", "x"):
            if k in obj:
                a = extract_first_array(obj[k])
                if a is not None:
                    return a
        for v in obj.values():
            a = extract_first_array(v)
            if a is not None:
                return a
        return None
    if getattr(obj, "dtype", None) == object and getattr(obj, "ndim", None) == 0:
        try:
            return extract_first_array(obj.item())
        except Exception:
            return None
    return None

def try_numpy_load_object(path):
    if np is None:
        return {"kind": "numpy_object", "ok": False, "reason": "numpy not installed"}
    try:
        obj = np.load(path, allow_pickle=True)
        info = {"kind": "numpy_object", "container": type(obj).__name__}
        # 直接 ndarray
        if isinstance(obj, np.ndarray) and obj.dtype != object:
            info.update({"ok": True, "shape": tuple(obj.shape), "dtype": str(obj.dtype)})
            return info
        # tuple/dict/0维 object -> 抽数组
        arr = extract_first_array(obj)
        if arr is not None:
            info.update({"ok": True, "shape": tuple(arr.shape), "dtype": str(arr.dtype), "note": "extracted first ndarray from container"})
        else:
            info.update({"ok": False, "reason": "no ndarray found inside container"})
        return info
    except Exception as e:
        return {"kind": "numpy_object", "ok": False, "reason": str(e)}

def is_tiff(head):
    return head.startswith(b"II*\x00") or head.startswith(b"MM\x00*")

def try_tiff(path):
    try:
        import tifffile as tiff
    except Exception:
        return {"kind": "tiff", "ok": False, "reason": "tifffile not installed"}
    try:
        arr = tiff.imread(path)
        return {"kind": "tiff", "ok": True, "shape": tuple(getattr(arr, "shape", ())), "dtype": str(getattr(arr, "dtype", ""))}
    except Exception as e:
        return {"kind": "tiff", "ok": False, "reason": str(e)}

def is_hdf5(head):
    return head.startswith(b"\x89HDF\r\n\x1a\n")

def human_size(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def probe_one(path):
    p = Path(path)
    out = {"path": str(p), "exists": p.exists()}
    if not p.exists() or not p.is_file():
        out["note"] = "not a file"
        return out
    try:
        size = p.stat().st_size
        out["size"] = f"{size} ({human_size(size)})"
        head = read_head(path, 64)
        out["ext"] = p.suffix.lower()

        # 1) NPY
        if is_npy(head):
            info = {"kind": "npy"}
            hdr = npy_header(path)
            if hdr:
                info.update(hdr)
            return {**out, **info}

        # 2) NPZ/ZIP
        if is_zip(head):
            return {**out, **probe_npz(path)}

        # 3) Blosc2
        b2 = try_blosc2(path)
        if b2.get("ok"):
            return {**out, **b2}
        # 4) Bloscpack
        bp = try_bloscpack(path)
        if bp.get("ok"):
            return {**out, **bp}

        # 5) numpy 对象容器（常见：.blosc 其实是 np.save 了 tuple/dict）
        npobj = try_numpy_load_object(path)
        if npobj.get("ok") or (npobj.get("kind") == "numpy_object" and "reason" in npobj):
            return {**out, **npobj}

        # 6) TIFF
        if is_tiff(head):
            return {**out, **try_tiff(path)}

        # 7) HDF5
        if is_hdf5(head):
            return {**out, "kind": "hdf5"}

        # 8) 其它常见图片（仅标识）
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return {**out, "kind": "png"}
        if head.startswith(b"\xff\xd8"):
            return {**out, "kind": "jpeg"}

        # 9) 未知
        return {**out, "kind": "unknown", "head_hex": head[:16].hex()}
    except Exception as e:
        return {**out, "kind": "error", "error": str(e), "trace": traceback.format_exc(limit=1)}

def scan_dir(root):
    roots = []
    for ext in (".blosc",".npy",".npz",".tif",".tiff",".h5",".hdf5",".png",".jpg",".jpeg"):
        roots += list(Path(root).rglob(f"*{ext}"))
    if not roots:
        roots = list(Path(root).rglob("*"))
        roots = [p for p in roots if p.is_file()]
    for p in roots:
        yield probe_one(str(p))

def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_hsi_format.py <file_or_dir>")
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        for info in scan_dir(target):
            print("="*80)
            for k,v in info.items():
                print(f"{k}: {v}")
    else:
        info = probe_one(target)
        for k,v in info.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
