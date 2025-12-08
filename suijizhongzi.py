#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patient-level split by SPECIFIED COUNTS for HSI datasets with filenames like:
  Case_<patient_id>_<index>.tif
Example: Case_3_8.tif

- Splits by patient: first choose patients (train_patients/test_patients), then expand all their samples.
- Keeps only samples that have BOTH HSI and label .tif present.
- Optional validation (read TIFFs) to skip corrupted files.
- Deterministic given --seed.
- Writes stem-only lists into <data_dir>/train.txt and <data_dir>/test.txt.

Usage
  python split_by_patient_count.py \
    --data_dir /home/bitmhsi/data_mouth \
    --hsi_dir roi_tif \
    --label_dir roi_label \
    --train_patients 57 \
    --test_patients 15 \
    --seed 2025 \
    --validate
"""

import argparse
import random
import re
import sys
from pathlib import Path

try:
    import tifffile as tiff
except Exception:
    tiff = None

# 'Case_<patient_id>_<idx>' (stem without extension)
FNAME_RE = re.compile(r'^(Case)_(\d+)_([0-9]+)$')

def parse_stem(stem: str):
    m = FNAME_RE.match(stem)
    if not m:
        return None
    # return: (prefix, patient_id:int, idx:int)
    return m.group(1), int(m.group(2)), int(m.group(3))

def ok_tif(p: Path, validate: bool) -> bool:
    """Check file exists; optionally try to read it to ensure it is not corrupted."""
    if not p.is_file():
        return False
    if not validate:
        return True
    if tiff is None:
        print("[WARN] tifffile not installed, skip validation", file=sys.stderr)
        return True
    try:
        tiff.imread(str(p))
        return True
    except Exception as e:
        print(f"[BAD] {p.name} -> {e}", file=sys.stderr)
        return False

def main(args):
    data_dir = Path(args.data_dir)
    hsi_dir  = data_dir / args.hsi_dir
    lab_dir  = data_dir / args.label_dir

    if not hsi_dir.is_dir():
        print(f"[ERROR] HSI dir not found: {hsi_dir}", file=sys.stderr)
        sys.exit(1)
    if not lab_dir.is_dir():
        print(f"[ERROR] Label dir not found: {lab_dir}", file=sys.stderr)
        sys.exit(1)

    # 1) Collect valid stems grouped by **patient**
    by_patient = {}
    total_files = 0
    for p in sorted(hsi_dir.glob("*.tif")):
        total_files += 1
        stem = p.stem
        parsed = parse_stem(stem)
        if parsed is None:
            continue
        _, pid, _ = parsed

        lab_p = lab_dir / f"{stem}.tif"
        if not ok_tif(p, args.validate):
            continue
        if not ok_tif(lab_p, args.validate):
            continue

        by_patient.setdefault(pid, []).append(stem)

    # Remove patients with zero valid pairs
    by_patient = {pid: stems for pid, stems in by_patient.items() if len(stems) > 0}

    if not by_patient:
        print("[ERROR] No valid (HSI,label) pairs found matching 'Case_<id>_<idx>.tif'.", file=sys.stderr)
        sys.exit(2)

    # 2) Deterministic **patient-level** selection
    patients = sorted(by_patient.keys())
    rng = random.Random(args.seed)
    rng.shuffle(patients)

    n_patients = len(patients)
    tr_n, te_n = args.train_patients, args.test_patients

    if tr_n + te_n != n_patients:
        # Fallback to 4:1 if the requested counts do not cover all patients
        tr_n = int(round(n_patients * 0.8))
        te_n = n_patients - tr_n
        print(f"[WARN] Requested train/test patient counts ({args.train_patients}+{args.test_patients}) "
              f"!= total patients ({n_patients}). Fallback to 4:1 -> train={tr_n}, test={te_n}")

    if tr_n <= 0 or te_n <= 0 or tr_n + te_n > n_patients:
        print(f"[ERROR] Invalid counts after adjustment: train={tr_n}, test={te_n}, total={n_patients}", file=sys.stderr)
        sys.exit(3)

    train_patients = set(patients[:tr_n])
    test_patients  = set(patients[tr_n:tr_n+te_n])

    assert len(train_patients & test_patients) == 0, "train/test patient sets overlap!"

    # 3) Expand to sample stems
    train, test = [], []
    for pid, stems in by_patient.items():
        stems = sorted(set(stems), key=lambda s: (parse_stem(s)[1], parse_stem(s)[2]))
        if pid in train_patients:
            train.extend(stems)
        elif pid in test_patients:
            test.extend(stems)
        else:
            # should not happen when tr_n+te_n == n_patients
            pass

    # 4) Final sort for readability (by patient then index)
    def keyfun(s):
        _, c, i = parse_stem(s)
        return (c, i)

    train.sort(key=keyfun)
    test.sort(key=keyfun)

    # 5) Write output files (stem only)
    (data_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (data_dir / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")

    # 6) Summary
    total_imgs  = sum(len(v) for v in by_patient.values())
    print(f"[OK] Patients: {n_patients} | Total valid images: {total_imgs} (from {total_files} files scanned)")
    print(f"[OK] Train patients: {len(train_patients)} | Test patients: {len(test_patients)}")
    print(f"[OK] Train samples: {len(train)} | Test samples: {len(test)}")
    print(f"[INFO] Seed: {args.seed}")
    # Per-patient brief (first 10)
    tr_counts, te_counts = {}, {}
    for s in train:
        _ , pid, _ = parse_stem(s)
        tr_counts[pid] = tr_counts.get(pid, 0) + 1
    for s in test:
        _ , pid, _ = parse_stem(s)
        te_counts[pid] = te_counts.get(pid, 0) + 1

    print("pid   n_all  n_tr  n_te")
    shown = 0
    for pid in sorted(by_patient.keys()):
        if shown >= 10:
            break
        n_all = len(by_patient[pid])
        n_tr  = tr_counts.get(pid, 0)
        n_te  = te_counts.get(pid, 0)
        print(f"{pid:4d}  {n_all:5d}  {n_tr:4d}  {n_te:4d}")
        shown += 1
    if n_patients > shown:
        print("... (per-patient summary truncated)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/bitmhsi/data_mouth")
    parser.add_argument("--hsi_dir",  type=str, default="roi_tif")
    parser.add_argument("--label_dir",type=str, default="roi_label")
    parser.add_argument("--train_patients", type=int, default=57)
    parser.add_argument("--test_patients",  type=int, default=15)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--validate", action="store_true", help="read TIFFs to skip corrupted files")
    args = parser.parse_args()
    main(args)
