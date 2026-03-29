"""
ilr_scoring.py
--------------
Inside-Layer Ranking (ILR) scoring for structured CNN pruning.

ILR fuses three forward-only statistics into a single per-channel
importance score in [0, 1] within each layer. Channels with SMALLER
ILR scores are considered LESS important and are candidates for pruning.

Components:
    act_rms  (weight 0.6) — RMS of post-ReLU activation maps (streaming)
    bn_gamma (weight 0.4) — Absolute value of BatchNorm scale parameter γ
    hrank    (weight 0.2) — Frobenius norm of feature maps (HRank-style energy)

All three signals are per-layer min-max normalized to [0, 1] before fusion,
so the final ILR score is independent of absolute magnitude differences
between layers.

ILR is used in the canonical Global pruning runs (ILR_REPLACE_GLOBAL=True),
completely replacing Taylor-saliency as the ranking signal. Advantages:
    - Forward-only: no gradient computation needed
    - Stable across torch.compile, mixed-precision, and long runs
    - Cheap to evaluate: runs in a few forward passes per round

Author: Sai Naga Chaithanya Aavula (pipeline), integrated into
        Global scoring framework
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set


# Default ILR weights (matching canonical experiment config)
DEFAULT_WEIGHTS = {"act_rms": 0.6, "bn_gamma": 0.4, "hrank": 0.2}
DEFAULT_COMPONENTS = {"act_rms": True, "bn_gamma": True, "hrank": True}


def compute_ilr_scores(
    model: nn.Module,
    layer_specs: List[Dict],
    eval_loader,
    device: torch.device,
    weights: Dict[str, float] = None,
    components: Dict[str, bool] = None,
    max_batches: int = 6,
    target_layers: Optional[Set[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute ILR importance scores for all channels in the specified layers.

    Args:
        model: The (possibly partially pruned) CNN model.
        layer_specs: List of dicts, each with keys:
                     'conv' (Conv2d path), 'bn' (BN path or None),
                     'pool' (pool path or None).
        eval_loader: DataLoader for scoring data.
        device: torch.device.
        weights: Component weights. Defaults to {act_rms:0.6, bn_gamma:0.4, hrank:0.2}.
        components: Which components to include. Defaults to all True.
        max_batches: Max number of forward-pass batches per component.
        target_layers: If set, only compute scores for these conv paths.

    Returns:
        Dict mapping conv_path -> tensor of shape (C,), values in [0, 1].
        Smaller = less important = candidate for pruning.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    if components is None:
        components = DEFAULT_COMPONENTS.copy()

    # Filter to target layers if specified
    specs = layer_specs
    if target_layers is not None:
        specs = [s for s in layer_specs if s["conv"] in target_layers]

    raw: Dict[str, Dict[str, torch.Tensor]] = {}

    # --- Component 1: Activation RMS ---
    if components.get("act_rms", True) and weights.get("act_rms", 0) > 0:
        raw["act_rms"] = _collect_act_rms(model, specs, eval_loader, device, max_batches)

    # --- Component 2: BN-gamma ---
    if components.get("bn_gamma", True) and weights.get("bn_gamma", 0) > 0:
        raw["bn_gamma"] = _collect_bn_gamma(model, specs)

    # --- Component 3: HRank (Frobenius energy) ---
    if components.get("hrank", True) and weights.get("hrank", 0) > 0:
        raw["hrank"] = _collect_hrank(model, specs, eval_loader, device, max_batches)

    # --- Normalize each component per-layer to [0, 1] ---
    normalized: Dict[str, Dict[str, torch.Tensor]] = {}
    for comp_name, comp_dict in raw.items():
        normalized[comp_name] = _normalize_per_layer(comp_dict)

    # --- Fuse with weights ---
    fused: Dict[str, torch.Tensor] = {}
    for spec in specs:
        k = spec["conv"]
        acc = None
        wsum = 0.0
        for comp_name, comp_dict in normalized.items():
            if k not in comp_dict:
                continue
            w = float(weights.get(comp_name, 0.0))
            if w == 0.0:
                continue
            v = comp_dict[k].to(torch.float32)
            acc = v * w if acc is None else (acc + v * w)
            wsum += w
        if acc is None:
            # Fallback: flat zero (treat all channels as equally unimportant)
            C = _get_conv_out_channels(model, k)
            fused[k] = torch.zeros(C, dtype=torch.float32)
        else:
            fused[k] = acc / max(1e-12, wsum)

    return fused


# ------------------------------------------------------------------ #
#  Component collectors                                                #
# ------------------------------------------------------------------ #

def _collect_act_rms(model, specs, loader, device, max_batches) -> Dict[str, torch.Tensor]:
    """
    Compute per-channel activation RMS using streaming accumulation.
    RMS = sqrt(mean(activation^2)) over spatial dims and batches.
    Higher RMS → more active channel → more important.
    """
    accum: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    hooks = []
    activations = {}

    def make_hook(path):
        def hook(m, inp, out):
            # out: (B, C, H, W) — post-conv, pre-BN/ReLU
            # Use absolute values to handle pre-ReLU activations
            rms = out.detach().float().pow(2).mean(dim=[0, 2, 3])  # (C,)
            if path not in accum:
                accum[path] = torch.zeros(rms.shape)
                counts[path] = 0
            accum[path] += rms.cpu()
            counts[path] += 1
        return hook

    for spec in specs:
        conv = _get_module(model, spec["conv"])
        hooks.append(conv.register_forward_hook(make_hook(spec["conv"])))

    model.eval()
    with torch.no_grad():
        for i, (x, *_) in enumerate(loader):
            if i >= max_batches:
                break
            model(x.to(device))

    for h in hooks:
        h.remove()

    result = {}
    for path, acc in accum.items():
        n = counts[path]
        result[path] = (acc / max(1, n)).sqrt()  # per-channel RMS
    return result


def _collect_bn_gamma(model, specs) -> Dict[str, torch.Tensor]:
    """
    Read |BN.gamma| (scale parameter) for each channel.
    BatchNorm gamma near zero → the layer is suppressing this channel.
    """
    result = {}
    for spec in specs:
        bn_path = spec.get("bn")
        if not bn_path:
            continue
        try:
            bn = _get_module(model, bn_path)
            if isinstance(bn, nn.BatchNorm2d):
                result[spec["conv"]] = bn.weight.data.abs().cpu()
        except AttributeError:
            continue
    return result


def _collect_hrank(model, specs, loader, device, max_batches) -> Dict[str, torch.Tensor]:
    """
    Compute Frobenius norm of feature maps per channel (HRank-style energy).
    Higher Frobenius norm → richer feature map → more important.
    """
    accum: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    hooks = []

    def make_hook(path):
        def hook(m, inp, out):
            # Frobenius norm per channel: sqrt(sum(x^2)) over H, W
            frob = out.detach().float().pow(2).sum(dim=[2, 3]).sqrt().mean(dim=0)  # (C,)
            if path not in accum:
                accum[path] = torch.zeros(frob.shape)
                counts[path] = 0
            accum[path] += frob.cpu()
            counts[path] += 1
        return hook

    for spec in specs:
        conv = _get_module(model, spec["conv"])
        hooks.append(conv.register_forward_hook(make_hook(spec["conv"])))

    model.eval()
    with torch.no_grad():
        for i, (x, *_) in enumerate(loader):
            if i >= max_batches:
                break
            model(x.to(device))

    for h in hooks:
        h.remove()

    return {path: acc / max(1, counts[path]) for path, acc in accum.items()}


# ------------------------------------------------------------------ #
#  Normalization                                                        #
# ------------------------------------------------------------------ #

def _normalize_per_layer(
    comp_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Per-layer min-max normalization to [0, 1].
    Channels at 0 are least important; channels at 1 are most important.
    """
    result = {}
    for path, scores in comp_dict.items():
        s = scores.float()
        lo, hi = s.min(), s.max()
        if (hi - lo).abs() < 1e-10:
            result[path] = torch.ones_like(s) * 0.5
        else:
            result[path] = (s - lo) / (hi - lo)
    return result


# ------------------------------------------------------------------ #
#  Utilities                                                            #
# ------------------------------------------------------------------ #

def _get_module(model: nn.Module, path: str) -> nn.Module:
    m = model
    for part in path.split("."):
        m = getattr(m, part)
    return m


def _get_conv_out_channels(model: nn.Module, conv_path: str) -> int:
    try:
        return _get_module(model, conv_path).out_channels
    except AttributeError:
        return 1
