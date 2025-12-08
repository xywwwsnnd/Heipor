#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, pickle, sys
from pathlib import Path
from typing import Any
import numpy as np

try:
    import blosc as _pyblosc
    _HAS_BLOSC = True
except Exception:
    _HAS_BLOSC = False

def _short(v, k=64):
    s = repr(v)
    return s if len(s) <= k else s[:k] + "…"

def _try_decompress_bytes(b: bytes):
    if not _HAS_BLOSC or not isinstance(b, (bytes, bytearray)):
        return None
    try:
        raw = _pyblosc.decompress(b)
        return len(raw)
    except Exception:
        return None

def _summarize(obj: Any, depth=0, path="$", max_items=8, max_depth=3):
    ind = "  " * depth
    if depth > max_depth:
        print(f"{ind}{path}: <max_depth>")
        return
    t = type(obj).__name__
    if isinstance(obj, np.ndarray):
        print(f"{ind}{path}: ndarray shape={obj.shape} dtype={obj.dtype}")
        return
    if isinstance(obj, (bytes, bytearray)):
        dlen = len(obj)
        dec = _try_decompress_bytes(obj)
        hint = f" | blosc_decompressed={dec}B" if dec is not None else ""
        print(f"{ind}{path}: bytes len={dlen}{hint}")
        return
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{ind}{path}: dict keys={len(keys)} {_short(keys)}")
        for i, k in enumerate(keys[:max_items]):
            _summarize(obj[k], depth+1, f"{path}[{k!r}]", max_items, max_depth)
        if len(keys) > max_items:
            print(f"{ind}  … +{len(keys)-max_items} more")
        return
    if isinstance(obj, (list, tuple)):
        print(f"{ind}{path}: {t} len={len(obj)}")
        for i, v in enumerate(obj[:max_items]):
            _summarize(v, depth+1, f"{path}[{i}]", max_items, max_depth)
        if len(obj) > max_items:
            print(f"{ind}  … +{len(obj)-max_items} more")
        return
    print(f"{ind}{path}: {t} = {_short(obj)}")

def inspect_file(fp: Path):
    print(f"\n=== {fp} ===")
    try:
        with open(fp, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"!! pickle.load failed: {e}")
        return
    _summarize(obj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="根目录，含 HSI/ 与 label/")
    ap.add_argument("--file", help="直接指定一个文件（优先）")
    ap.add_argument("--n", type=int, default=2, help="各目录取多少个样本")
    args = ap.parse_args()

    if args.file:
        inspect_file(Path(args.file))
        return

    if not args.root:
        print("需要 --root 或 --file", file=sys.stderr)
        sys.exit(2)

    root = Path(args.root)
    for sub in ("HSI", "label"):
        p = root / sub
        files = sorted(list(p.glob("*.pkl")) + list(p.glob("*.pickle")) + list(p.glob("*")))
        print(f"\n### 目录 {p} | 共 {len(files)} 个，抽查 {args.n} ###")
        for fp in files[:args.n]:
            inspect_file(fp)

if __name__ == "__main__":
    main()
