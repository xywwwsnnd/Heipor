#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, random, argparse
from pathlib import Path

def collect_stems(hsi_dir: Path, lab_dir: Path):
    hsi = {p.stem for p in hsi_dir.rglob("*.mat")}
    lab = {p.stem for p in lab_dir.rglob("*.png")}
    common = sorted(hsi & lab)
    only_h = sorted(hsi - lab)
    only_l = sorted(lab - hsi)
    print(f"[SCAN] .mat={len(hsi)}  .png={len(lab)}  matched={len(common)}")
    if only_h:
        print(f"[WARN] .mat without .png: {len(only_h)} (e.g., {only_h[:5]})")
    if only_l:
        print(f"[WARN] .png without .mat: {len(only_l)} (e.g., {only_l[:5]})")
    return common

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsi_dir",   default="/home/bitmhsi/danguancancer/hsi")
    ap.add_argument("--label_dir", default="/home/bitmhsi/danguancancer/label")
    ap.add_argument("--out_dir",   default="/home/bitmhsi/danguancancer/split")
    ap.add_argument("--seed",      type=int, default=2025)
    ap.add_argument("--train_ratio", type=float, default=0.8)  # 8:2
    args = ap.parse_args()

    hsi_dir   = Path(args.hsi_dir)
    label_dir = Path(args.label_dir)
    out_dir   = Path(args.out_dir)

    assert hsi_dir.is_dir(),   f"HSI目录不存在: {hsi_dir}"
    assert label_dir.is_dir(), f"Label目录不存在: {label_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = collect_stems(hsi_dir, label_dir)
    if not stems:
        raise SystemExit("[ERROR] 没有找到可用的配对样本")

    random.seed(args.seed)
    random.shuffle(stems)

    n_total = len(stems)
    n_train = int(round(n_total * args.train_ratio))
    n_train = max(1, min(n_total - 1, n_train))  # 避免某侧为空
    train_stems = sorted(stems[:n_train])
    test_stems  = sorted(stems[n_train:])

    (out_dir / "train.txt").write_text("\n".join(train_stems) + "\n", encoding="utf-8")
    (out_dir / "test.txt").write_text("\n".join(test_stems) + "\n",  encoding="utf-8")

    print(f"[OK] train: {len(train_stems)}  test: {len(test_stems)}  → {out_dir}")
    print(f"[OK] 示例: {train_stems[:3]} ... | {test_stems[:3]} ...")

if __name__ == "__main__":
    main()
