#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, numpy as np
from pathlib import Path

# --------------------- 读 .mat 的鲁棒函数 --------------------- #
def load_mat_any(path):
    try:
        import scipy.io as sio
        d = sio.loadmat(path)
        cand = [(k, v) for k, v in d.items() if isinstance(v, np.ndarray)]
        cand.sort(key=lambda kv: -np.prod(kv[1].shape))
        for _, v in cand:
            if v.ndim in (2, 3):
                return v
    except Exception:
        pass
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            dsets = []
            def collect(g):
                for _, v in g.items():
                    if isinstance(v, h5py.Dataset):
                        dsets.append(v)
                    elif isinstance(v, h5py.Group):
                        collect(v)
            collect(f)
            if dsets:
                dsets.sort(key=lambda x: -np.prod(x.shape))
                return np.array(dsets[0][()])
    except Exception:
        pass
    raise RuntimeError(f"Cannot read {path} as .mat")

def find_files(root, ext):
    root = Path(root)
    return [p for p in root.rglob(f"*{ext}") if p.is_file()]

def stem_no_ext(p: Path):
    return p.stem

def pair_by_stem(hsi_paths, lab_paths):
    hsi_map = {stem_no_ext(p): p for p in hsi_paths}
    lab_map = {stem_no_ext(p): p for p in lab_paths}
    common  = sorted(set(hsi_map.keys()) & set(lab_map.keys()))
    only_h  = sorted(set(hsi_map.keys()) - set(lab_map.keys()))
    only_l  = sorted(set(lab_map.keys()) - set(hsi_map.keys()))
    pairs   = [(k, hsi_map[k], lab_map[k]) for k in common]
    return pairs, only_h, only_l

def main():
    ap = argparse.ArgumentParser()
    # 直接把你的路径作为默认值
    ap.add_argument("--hsi_root", type=str, default="/home/bitmhsi/danguancancer/hsi",
                    help="HSI根目录（.mat）")
    ap.add_argument("--lab_root", type=str, default="/home/bitmhsi/danguancancer/label",
                    help="Label根目录（.png）")
    ap.add_argument("--max_pairs", type=int, default=10)
    ap.add_argument("--expect_c", type=int, default=60)
    args = ap.parse_args()

    hsi_paths = find_files(args.hsi_root, ".mat")
    lab_paths = find_files(args.lab_root, ".png")
    print(f"[SCAN] found .mat: {len(hsi_paths)}  .png: {len(lab_paths)}")

    pairs, only_h, only_l = pair_by_stem(hsi_paths, lab_paths)
    print(f"[PAIR] matched pairs: {len(pairs)}")
    if only_h:
        print(f"[WARN] .mat without .png: {len(only_h)} (e.g., {only_h[:5]})")
    if only_l:
        print(f"[WARN] .png without .mat: {len(only_l)} (e.g., {only_l[:5]})")

    from PIL import Image
    for stem, hsi_p, lab_p in pairs[:args.max_pairs]:
        # HSI
        try:
            arr = load_mat_any(str(hsi_p))
        except Exception as e:
            print(f"[HSI][BAD] {hsi_p} -> {e}")
            continue
        shp = arr.shape
        ch_guess, layout = None, "Unknown"
        if arr.ndim == 3:
            c0, c1, c2 = shp
            if 4 <= c0 <= 512 and c0 != c1 and c0 != c2:
                ch_guess, layout = c0, "CHW"
            elif 4 <= c2 <= 512 and c2 != c0 and c2 != c1:
                ch_guess, layout = c2, "HWC"
            else:
                if c0 == args.expect_c: ch_guess, layout = c0, "CHW?"
                elif c2 == args.expect_c: ch_guess, layout = c2, "HWC?"
        print(f"[HSI] {hsi_p.name} shape={shp}  channel_guess={ch_guess} layout={layout}"
              + (f"  (≠{args.expect_c}?)" if (ch_guess and args.expect_c and ch_guess!=args.expect_c) else ""))

        # Label
        try:
            im = Image.open(lab_p)
            mode = im.mode; size = im.size; bands = im.getbands()
            arrL = np.array(im)
            uniq = np.unique(arrL)
            preview = uniq[:10]
            print(f"[LAB] {lab_p.name} mode={mode} size={size} bands={bands} "
                  f"dtype={arrL.dtype} unique(n={uniq.size}): {preview}{' ...' if uniq.size>10 else ''}")
        except Exception as e:
            print(f"[LAB][BAD] {lab_p} -> {e}")

    print("\n[SUMMARY]")
    print(f"Total pairs: {len(pairs)}  |  Unpaired .mat: {len(only_h)}  |  Unpaired .png: {len(only_l)}")
    print("若 HSI 的 channel_guess≈60，且标签 unique≈{0,1}/{0,255}，可直接适配。")

if __name__ == "__main__":
    main()
