# vit_model/vit_transformer/ckpt_probe.py
import os, re, sys, argparse, importlib, torch

def load_sd(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ["state_dict","model","net","backbone"]:
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]; break
    out = {}
    for k,v in sd.items():
        k2 = re.sub(r'^(module\.|backbone\.|encoder\.|model\.)', '', k)
        out[k2] = v
    return out

def score(model, sd):
    msd = model.state_dict()
    return sum(1 for k,v in sd.items() if k in msd and v.shape == msd[k].shape), len(msd)

def build_candidates():
    cands = []
    # vits
    try:
        vits = importlib.import_module("vit_model.external.transpath.vits")
        for name in ["vit_small","vit_base"]:
            if hasattr(vits, name):
                fn = getattr(vits, name)
                try: m = fn(patch_size=16)
                except: m = fn()
                cands.append((name, m))
    except Exception as e:
        print("[warn] load vits failed:", e)
    # ctrans
    try:
        ctran = importlib.import_module("vit_model.external.transpath.ctran")
        for name in ["ctrans_small","ctrans_base","build_ctranspath","build_model"]:
            if hasattr(ctran, name):
                fn = getattr(ctran, name)
                try: m = fn(patch_size=16)
                except: m = fn()
                cands.append((name, m))
    except Exception as e:
        pass
    return cands

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to pretrain .pth")
    args = ap.parse_args()

    sd = load_sd(args.ckpt)
    cands = build_candidates()
    if not cands:
        print("没有可用候选模型；确认 vits.py/ctran.py 在 vit_model/external/transpath/ 且可导入。")
        sys.exit(1)

    print(f"候选模型：{[n for n,_ in cands]}")
    best = None
    for name, m in cands:
        ok, total = score(m, sd)
        print(f"{name:18s}  形状匹配: {ok}/{total}")
        if best is None or ok > best[1]:
            best = (name, ok, total)

    if best:
        name, ok, total = best
        print("\n>>> 判定：这份权重最像 ->", name, f"（匹配 {ok}/{total}）")
        if "vit_small" in name: print("建议 config：transpath_variant='vit_small', transpath_hidden=384")
        if "vit_base"  in name: print("建议 config：transpath_variant='vit_base',  transpath_hidden=768")
        if "ctrans"    in name: print("建议 config：transpath_variant='ctrans_small' 或相应名字；hidden: small=384/base=768")
    else:
        print("无法判定。")

if __name__ == "__main__":
    main()
