# tools/inspect_transpath.py
# -*- coding: utf-8 -*-
import re, math, json, os, sys, torch

# ====== 固定参数：直接在这里改 ======
CKPT_PATH  = "/home/bitmhsi/fushion_resnet+Transpath_bandselect/checkpoints/ctranspath.pth"
IMG_SIZE   = 256
PATCH_SIZE = 16
# ===================================

def load_sd(path: str):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ["state_dict","model","net","backbone"]:
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]; break
    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint格式异常：应为dict或dict['state_dict']")
    out = {}
    for k, v in sd.items():
        kk = re.sub(r"^(module\.|backbone\.|encoder\.|model\.)", "", k)
        out[kk] = v
    return out

def pos_embed_info(sd, img_size: int, patch_size: int):
    H = W = img_size
    h_new, w_new = H // patch_size, W // patch_size
    pe = sd.get("pos_embed", None)
    info = {}
    if isinstance(pe, torch.Tensor) and pe.ndim == 2:
        pe = pe.unsqueeze(0)
    if isinstance(pe, torch.Tensor) and pe.ndim == 3:
        _, N, C = pe.shape
        s = int(math.sqrt(N)) if int(math.sqrt(N))**2 == N else None
        info["pos_embed_shape"] = list(pe.shape)
        info["pos_grid_old"] = [s, s] if s is not None else None
        info["pos_grid_new"] = [h_new, w_new]
    return info

def group_and_print(sd: dict):
    groups = {}
    for k, v in sd.items():
        root = k.split('.')[0]
        groups.setdefault(root, []).append((k, list(v.shape)))
    for g in sorted(groups.keys()):
        print(f"\n[Group] {g}")
        for k, shp in sorted(groups[g], key=lambda x: x[0]):
            print(f"  {k:<60} {shp}")
    total = sum(p.numel() for p in sd.values())
    mb = sum(p.numel()*p.element_size() for p in sd.values())/1024/1024
    print(f"\n[Total] params={total:,}  approx={mb:.2f} MB")

def main():
    print(f"[LOAD] {CKPT_PATH}")
    raw = load_sd(CKPT_PATH)

    print("=== 原始权重（全部键与形状） ===")
    group_and_print(raw)

    pe = pos_embed_info(raw, IMG_SIZE, PATCH_SIZE)
    if pe:
        print("\n[PosEmbed] info:", json.dumps(pe, ensure_ascii=False))

    print("\n=== 和当前实现对齐（只看形状是否匹配） ===")
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
    if PROJ_ROOT not in sys.path:
        sys.path.insert(0, PROJ_ROOT)

    try:
        from vit_model.external.transpath.ctran import ctranspath
    except Exception as e:
        print("[WARN] 无法导入 vit_model.external.transpath.ctran.ctranspath：", repr(e))
        print("       仅做原始权重形状展示，不做对齐检查。")
        return

    # ---- 逐级尝试不同构造签名 ----
    model = None
    tried = []
    try:
        model = ctranspath(img_size=IMG_SIZE, patch_size=PATCH_SIZE)
        tried.append("ctranspath(img_size, patch_size)")
    except TypeError:
        try:
            model = ctranspath(patch_size=PATCH_SIZE)
            tried.append("ctranspath(patch_size)")
        except TypeError:
            model = ctranspath()
            tried.append("ctranspath()")

    print(f"[Instantiate] 尝试顺序: {tried[-1]} 成功")
    msd = model.state_dict()
    embed_dim = getattr(model, "embed_dim", None)
    num_features = getattr(model, "num_features", None)
    if embed_dim is not None:
        print(f"[Model] embed_dim   = {embed_dim}")
    if num_features is not None:
        print(f"[Model] num_features= {num_features}")

    ok = 0
    miss = []
    unexp = []
    shape_mismatch = []

    for k, v in raw.items():
        if k in msd:
            if v.shape == msd[k].shape:
                ok += 1
            else:
                shape_mismatch.append((k, list(v.shape), list(msd[k].shape)))
        else:
            unexp.append(k)

    for k in msd.keys():
        if k not in raw:
            miss.append(k)

    print(f"[Match] 形状完全匹配: {ok} / {len(msd)}")
    print(f"[Missing in CKPT] 模型需要但权重里没有的键: {len(miss)}")
    if miss:
        print("  示例:", miss[:10])
    print(f"[Unexpected] 权重里有但模型没有的键: {len(unexp)}")
    if unexp:
        print("  示例:", unexp[:10])
    print(f"[Shape Mismatch] 同名但形状不同的键: {len(shape_mismatch)}")
    if shape_mismatch:
        for k, s_ckpt, s_model in shape_mismatch[:10]:
            print(f"  {k}: ckpt{tuple(s_ckpt)} vs model{tuple(s_model)}")
        if len(shape_mismatch) > 10:
            print("  ...（其余略）")

if __name__ == "__main__":
    main()
