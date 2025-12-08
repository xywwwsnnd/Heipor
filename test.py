# sanity_check_v2.py
import os, json, inspect, warnings
import torch
import torch.nn as nn

from vit_model.config  import Config
from vit_model.model   import VisionTransformer

# --------- 可按你工程实际命名做微调的规则 --------- #
STAGEWISE_NAMES = [
    "StagewiseHSIFusion", "HSIStageFusion", "StageFusion", "HSIFuser",
    "HSIStagewiseFuse", "StageWiseFuse", "StageWiseHSIFusion"
]
SHALLOW_NAMES = ["SELayer", "ShallowFusion", "SEFusion", "SEGate"]
HYBRID_HINTS  = ["FuseResNet", "HybridResNet", "FuseResNetV2"]
TRANSPATH_WRAPPER_HINT = ["TransPathRGBEncoder"]
WINDOW_ATTN_NAMES = ["WindowAttention", "LocalWindowAttention", "SwinTransformerBlock"]
OUR_ATTN_HINT_ROOTS = [
    # 你自己 Transformer 的路径前缀（避免把 TransPath 的统计进来）
    "transformer.encoder", "transformer.blocks", "vit_transformer.encoder",
    "vit_model.transformer.encoder", "transformer.block", "vit_model.transformer.block"
]
# -------------------------------------------------- #

def _classname(m): return m.__class__.__name__

def _module_path(root, target):
    # 拼出模块层级路径，便于判定属于谁
    for name, mod in root.named_modules():
        if mod is target:
            return name
    return ""

def main():
    dev = torch.device(Config.device)
    model = VisionTransformer(Config.model_config, img_size=Config.image_size, num_classes=1).to(dev)
    model.eval()

    # ------- 读取关键 config flag ------- #
    cfg_flags = {
        "use_hybrid_resnet":     bool(getattr(Config.model_config, "use_hybrid_resnet", False)),
        "use_transpath_rgb":     bool(getattr(Config.model_config, "use_transpath_rgb", False)),
        "use_shallow_fusion":    bool(getattr(Config.model_config, "use_shallow_fusion", False)),
        "use_stagewise_hsi_fusion": bool(getattr(Config.model_config, "use_stagewise_hsi_fusion", False)),
        "use_window_msa_sa":     bool(getattr(Config.model_config, "use_window_msa_sa", False)),
        "use_window_msa_mba":    bool(getattr(Config.model_config, "use_window_msa_mba", False)),
        "use_pseudo_rgb":        bool(getattr(Config.model_config, "use_pseudo_rgb", False)),
        "use_medsam2_encoder":   bool(getattr(Config.model_config, "use_medsam2_encoder", False)),
        "transpath_variant":     getattr(Config.model_config, "transpath_variant", None),
    }

    print("="*36)
    print("Sanity Check v2: Model Wiring vs Config")
    print("="*36)
    print("[Config flags]")
    print(json.dumps(cfg_flags, indent=2, ensure_ascii=False))

    # ------- 静态扫描：有没有这些模块 ------- #
    have = dict(
        hybrid=False, transpath=False, shallow=False, stagewise=False,
        window_transpath=False, window_ours=False,
        pseudo_rgb=False, sam2=False
    )

    # 统计收集
    hits = {
        "Hybrid-ResNet": [],
        "CTransPath":    [],
        "Shallow":       [],
        "Stagewise":     [],
        "Window@TransPath": [],
        "Window@Ours":      [],
        "PseudoRGB":     [],
        "SAM2":          [],
    }

    # 方便找 embeddings / encoder
    emb = getattr(getattr(model, "transformer", model), "embeddings", None)
    enc = getattr(getattr(model, "transformer", model), "encoder", None)

    # ---- 静态遍历所有子模块 ---- #
    for name, mod in model.named_modules():
        cname = _classname(mod)

        # Hybrid-ResNet
        if any(h in cname for h in HYBRID_HINTS):
            have["hybrid"] = True
            hits["Hybrid-ResNet"].append((name, cname))

        # TransPath Wrapper
        if any(h in cname for h in TRANSPATH_WRAPPER_HINT):
            have["transpath"] = True
            hits["CTransPath"].append((name, cname))

        # 浅层融合（SE 类）
        if any(sn == cname for sn in SHALLOW_NAMES):
            have["shallow"] = True
            hits["Shallow"].append((name, cname))

        # 分阶段融合（按常见命名）
        if any(sn == cname for sn in STAGEWISE_NAMES):
            have["stagewise"] = True
            hits["Stagewise"].append((name, cname))

        # 窗口注意力（先不区分归属；稍后按路径再分）
        if any(w in cname for w in WINDOW_ATTN_NAMES):
            # 先记录，后面根据路径分到 ours/transpath
            # 这里保存模块对象，稍后二次判定
            hits.setdefault("_window_candidates", []).append((name, mod))

        # Pseudo RGB
        if "BandSelector" in cname or "BandWeightGenerator" in cname or "DynamicBandSelector" in cname:
            have["pseudo_rgb"] = True
            hits["PseudoRGB"].append((name, cname))

        # SAM2/MEDSAM（仅按命名提示）
        if "SAM" in cname or "Hiera" in cname:
            have["sam2"] = True
            hits["SAM2"].append((name, cname))

    # ---- 把窗口注意力按归属分桶 ---- #
    for name, mod in hits.get("_window_candidates", []):
        # 属于 TransPath？（判断名字路径在 embeddings.rgb_encoder_tp 子树下）
        in_transpath = name.startswith("transformer.embeddings.rgb_encoder_tp")
        # 属于我们自己的 transformer？（路径里包含 OUR_ATTN_HINT_ROOTS 的前缀）
        in_ours = any(name.startswith(prefix) for prefix in OUR_ATTN_HINT_ROOTS)
        if in_transpath:
            have["window_transpath"] = True
            hits["Window@TransPath"].append((name, _classname(mod)))
        elif in_ours:
            have["window_ours"] = True
            hits["Window@Ours"].append((name, _classname(mod)))

    # ---- 动态：前向调用“是否真的被走到” ---- #
    called = dict(
        hybrid=False, transpath=False, shallow=False, stagewise=False,
        window_transpath=False, window_ours=False
    )
    hooks = []

    def _flagger(flag_key):
        def _hook(*_):
            called[flag_key] = True
        return _hook

    # Hybrid-ResNet 调用
    if emb is not None and hasattr(emb, "hybrid_model"):
        hooks.append(emb.hybrid_model.register_forward_hook(_flagger("hybrid")))
    # TransPath 调用
    if emb is not None and hasattr(emb, "rgb_encoder_tp"):
        hooks.append(emb.rgb_encoder_tp.register_forward_hook(_flagger("transpath")))
    # 浅层融合（SE 层）
    if emb is not None:
        for name, mod in emb.named_modules():
            if _classname(mod) in SHALLOW_NAMES:
                hooks.append(mod.register_forward_hook(_flagger("shallow")))
    # Stagewise（仅对命名匹配的模块挂钩）
    if enc is not None:
        for name, mod in model.named_modules():
            if _classname(mod) in STAGEWISE_NAMES:
                hooks.append(mod.register_forward_hook(_flagger("stagewise")))
    # 窗口注意力：TransPath
    if emb is not None and hasattr(emb, "rgb_encoder_tp"):
        for name, mod in emb.rgb_encoder_tp.named_modules():
            if _classname(mod) in WINDOW_ATTN_NAMES:
                hooks.append(mod.register_forward_hook(_flagger("window_transpath")))
    # 窗口注意力：我们的 transformer
    if enc is not None:
        for name, mod in enc.named_modules():
            if _classname(mod) in WINDOW_ATTN_NAMES:
                # 限定路径在我们的 encoder 子树里
                hooks.append(mod.register_forward_hook(_flagger("window_ours")))

    # ---- 做一次前向，触发 hooks ---- #
    B, H, W = 2, Config.image_size, Config.image_size
    rgb = torch.randn(B, Config.num_channels_rgb,  H, W, device=dev)
    hsi = torch.randn(B, Config.num_channels_hsi, H, W, device=dev)
    with torch.no_grad():
        out = model(rgb, hsi)
        if isinstance(out, (list, tuple)):
            out = out[0]
    for h in hooks:
        h.remove()

    # ------- 打印结果 ------- #
    print("\n[Static scan hits] (first few)")
    def _show(title, key):
        arr = hits.get(key, [])[:10]
        print(f"{title:<20}: {arr}")

    _show("Hybrid-ResNet", "Hybrid-ResNet")
    _show("CTransPath",    "CTransPath")
    _show("Shallow",       "Shallow")
    _show("Stagewise",     "Stagewise")
    _show("Window@TransPath","Window@TransPath")
    _show("Window@Ours",     "Window@Ours")
    _show("Band/PseudoRGB",  "PseudoRGB")
    _show("SAM2/MEDSAM",     "SAM2")

    # 汇总表
    def yn(b): return "YES" if b else "NO "
    print("\n[Report]")
    print(f"{'Item':<30} | {'Cfg':^8} | {'Found':^8} | {'Called':^8}")
    print("-"*60)
    print(f"{'HSI uses Hybrid-ResNet':<30} | {yn(cfg_flags['use_hybrid_resnet']):^8} | {yn(have['hybrid']):^8} | {yn(called['hybrid']):^8}")
    print(f"{'RGB uses CTransPath':<30}   | {yn(cfg_flags['use_transpath_rgb']):^8} | {yn(have['transpath']):^8} | {yn(called['transpath']):^8}")
    print(f"{'Shallow fusion (SE)':<30}   | {yn(cfg_flags['use_shallow_fusion']):^8} | {yn(have['shallow']):^8} | {yn(called['shallow']):^8}")
    print(f"{'Stagewise HSI fusion':<30}  | {yn(cfg_flags['use_stagewise_hsi_fusion']):^8} | {yn(have['stagewise']):^8} | {yn(called['stagewise']):^8}")
    print(f"{'Window Attn @TransPath':<30}| {yn(False):^8} | {yn(have['window_transpath']):^8} | {yn(called['window_transpath']):^8}")
    print(f"{'Window Attn @Ours':<30}     | {yn(cfg_flags['use_window_msa_sa'] or cfg_flags['use_window_msa_mba']):^8} | {yn(have['window_ours']):^8} | {yn(called['window_ours']):^8}")
    print(f"{'Band selector / pseudoRGB':<30}| {yn(cfg_flags['use_pseudo_rgb']):^8} | {yn(have['pseudo_rgb']):^8} | {yn(True):^8}")
    print(f"{'SAM2/MEDSAM encoder':<30}    | {yn(cfg_flags['use_medsam2_encoder']):^8} | {yn(have['sam2']):^8} | {yn(False):^8}")

    # 额外提示
    if not cfg_flags["use_stagewise_hsi_fusion"] and (have["stagewise"] or called["stagewise"]):
        warnings.warn("检测到 Stagewise 模块存在或被调用，但配置为关闭；请确认命名是否误匹配。", RuntimeWarning)

    # 末尾形状/显存
    try:
        vram = torch.cuda.max_memory_allocated(dev) / 1e9
        print(f"\n[Runtime] device={dev} | peak_alloc_vram≈{vram:.2f} GB | out_shape={tuple(out.shape) if torch.is_tensor(out) else None}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
