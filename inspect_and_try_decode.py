# inspect_and_try_decode.py
import sys, os, textwrap
from pathlib import Path

def human(n):
    u = ["B","KB","MB","GB","TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(u)-1:
        x /= 1024.0; i += 1
    return f"{x:.1f}{u[i]}"

def safe_preview_bytes(b, limit=32):
    if not isinstance(b,(bytes,bytearray,memoryview)): return None
    bb = bytes(b)
    s = bb[:limit].hex()
    suffix = "..." if len(bb) > limit else ""
    return f"{len(bb)} bytes | head(hex)={s}{suffix}"

def short(v, maxlen=120):
    s = repr(v)
    return s if len(s) <= maxlen else s[:maxlen] + "..."

def is_bytes_like(x):
    return isinstance(x,(bytes,bytearray,memoryview))

def tree(obj, depth=0, max_depth=3, visited=None, keyname=None):
    """Pretty-print container structure without huge dumps."""
    import numpy as np
    if visited is None: visited = set()
    prefix = "  " * depth
    tag = f"[{keyname}]" if keyname is not None else ""
    oid = id(obj)
    if oid in visited:
        print(f"{prefix}{tag}<visited {type(obj).__name__}>")
        return
    visited.add(oid)

    if obj is None:
        print(f"{prefix}{tag}None")
        return
    if isinstance(obj, (int,float,str,bool)):
        ss = short(obj)
        print(f"{prefix}{tag}{type(obj).__name__}: {ss}")
        return
    if is_bytes_like(obj):
        info = safe_preview_bytes(obj)
        print(f"{prefix}{tag}{type(obj).__name__}: {info}")
        return
    if np and isinstance(obj, np.ndarray):
        print(f"{prefix}{tag}ndarray: shape={obj.shape}, dtype={obj.dtype}, ndim={obj.ndim}")
        if obj.dtype == object and obj.ndim == 0 and depth < max_depth:
            try:
                inner = obj.item()
                tree(inner, depth+1, max_depth, visited, keyname="item()")
            except Exception as e:
                print(f"{prefix}  (0-dim object item() failed: {e})")
        return
    if depth >= max_depth:
        print(f"{prefix}{tag}{type(obj).__name__}: (max_depth reached)")
        return

    if isinstance(obj, (list,tuple)):
        print(f"{prefix}{tag}{type(obj).__name__}: len={len(obj)}")
        for i, it in enumerate(obj[:16]):  # 最多看前 16 个
            tree(it, depth+1, max_depth, visited, keyname=f"{i}")
        if len(obj) > 16:
            print(f"{prefix}  ... ({len(obj)-16} more)")
        return
    if isinstance(obj, dict):
        print(f"{prefix}{tag}dict: keys={list(obj.keys())[:16]}")
        for k in list(obj.keys())[:16]:
            tree(obj[k], depth+1, max_depth, visited, keyname=str(k))
        if len(obj) > 16:
            print(f"{prefix}  ... ({len(obj)-16} more keys)")
        return

    # Fallback
    print(f"{prefix}{tag}{type(obj).__name__}: {short(obj)}")


def try_decode(obj):
    """
    Try to reconstruct an ndarray from common 'container' patterns:
    1) (bytes_like, meta{shape, dtype[, order]})
    2) {data: bytes_like, shape: ..., dtype: ...}
    3) chunked: {chunks: [bytes...], shape:..., dtype:...} or ( [bytes...], meta )
    4) plain ndarray inside container
    """
    import numpy as np

    # 0) If it's already an ndarray (non-object), return it
    if isinstance(obj, np.ndarray) and obj.dtype != object:
        return obj

    # 1) object array scalar -> unwrap and retry
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        try:
            return try_decode(obj.item())
        except Exception:
            pass

    # 2) tuple/list patterns
    if isinstance(obj, (list, tuple)):
        # 2.1) direct ndarray somewhere inside
        for it in obj:
            arr = try_decode(it)
            if isinstance(arr, np.ndarray) and arr.dtype != object:
                return arr
        # 2.2) (bytes..., meta)
        if len(obj) >= 2:
            meta = obj[-1]
            data_parts = obj[:-1]
            if isinstance(meta, dict):
                arr = try_decode_from_bytes_meta(data_parts, meta)
                if arr is not None:
                    return arr

    # 3) dict pattern
    if isinstance(obj, dict):
        # a) direct ndarray under common keys
        for k in ("cube","hsi","data","array","X","x"):
            if k in obj:
                arr = try_decode(obj[k])
                if isinstance(arr, np.ndarray) and arr.dtype != object:
                    return arr
        # b) bytes + meta
        arr = try_decode_from_bytes_meta(obj.get("data", None), obj)
        if arr is not None:
            return arr
        # c) chunks + meta
        if "chunks" in obj:
            arr = try_decode_from_bytes_meta(obj.get("chunks"), obj)
            if arr is not None:
                return arr

        # d) try any value
        for v in obj.values():
            arr = try_decode(v)
            if isinstance(arr, np.ndarray) and arr.dtype != object:
                return arr

    return None


def try_decode_from_bytes_meta(data_parts, meta):
    """Decode array from (bytes/chunks, meta dict) if possible."""
    import numpy as np

    if data_parts is None:
        return None

    # unify to list of bytes
    if isinstance(data_parts, (bytes, bytearray, memoryview)):
        parts = [bytes(data_parts)]
    elif isinstance(data_parts, (list, tuple)):
        parts = []
        for p in data_parts:
            if isinstance(p, (bytes, bytearray, memoryview)):
                parts.append(bytes(p))
            # nested list of bytes
            elif isinstance(p, (list, tuple)) and all(isinstance(x,(bytes,bytearray,memoryview)) for x in p):
                parts.extend([bytes(x) for x in p])
    else:
        return None
    if not parts:
        return None
    blob = b"".join(parts)

    shape = meta.get("shape") or meta.get("Shape") or meta.get("shp")
    dtype = meta.get("dtype") or meta.get("DType") or meta.get("dt")
    order = meta.get("order", "C")
    compressed = bool(meta.get("compressed") or meta.get("is_compressed") or meta.get("blosc", False) or meta.get("codec") in ("blosc","blosc2"))

    if shape is None or dtype is None:
        # Try to parse shape if given as list-of-int in string
        return None

    # Try decompression if marked compressed
    raw = blob
    if compressed:
        # try blosc2 then blosc
        raw = None
        try:
            import blosc2
            raw = blosc2.decompress(blob)  # may work on many blosc2 payloads
        except Exception:
            try:
                import blosc
                raw = blosc.decompress(blob)
            except Exception:
                pass
        if raw is None:
            # maybe already uncompressed or unknown codec
            raw = blob

    # Build array from buffer
    try:
        arr = np.frombuffer(raw, dtype=np.dtype(dtype))
        arr = arr.reshape(tuple(shape), order=order if order in ("C","F") else "C")
        return arr
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_and_try_decode.py <file>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print("Not a file:", path)
        sys.exit(1)

    print("path:", path)
    print("size:", human(os.path.getsize(path)))
    print("ext :", Path(path).suffix.lower())

    # 先尝试 np.load(allow_pickle=True) 读取容器
    import numpy as np
    obj = None
    try:
        obj = np.load(path, allow_pickle=True)
        # 如果是 NpzFile，会当作 dict-like
        if hasattr(obj, "files"):
            print("\n== NPZ members:", obj.files)
            if obj.files:
                first = obj.files[0]
                arr = obj[first]
                print(f"-> First member '{first}': shape={arr.shape}, dtype={arr.dtype}")
                return
    except Exception as e:
        print("np.load failed:", e)
        return

    print("\n== Container structure (up to depth=3) ==")
    tree(obj, max_depth=3)

    print("\n== Try to decode an ndarray from container ==")
    arr = try_decode(obj)
    if arr is not None:
        print(f"DECODED ndarray: shape={arr.shape}, dtype={arr.dtype}, ndim={arr.ndim}")
    else:
        print("No ndarray could be decoded automatically.")
        print(textwrap.dedent("""
        提示：
        1) 容器里若有 bytes 和一个 meta(dict) 同时出现，请把 meta 里是否有 'shape' / 'dtype' / 'order' / 'compressed' / 'codec' 贴出来；
        2) 如果 bytes 很大（几十 MB），很可能就是压缩后的数组；可尝试安装 blosc/blosc2 以便自动解压；
        3) 若容器里只有路径字符串或很小的 dict，需要查看它生成数据的脚本，确认写入格式再定制解码。
        """).strip())

if __name__ == "__main__":
    main()
