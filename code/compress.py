"""Post-hoc compression operators and the proposed Explanation-Aware
Quantization (EAQ) bit-allocation.

All operators act on a *copy* of a trained model's weights so the baseline is
never mutated. Quantization is uniform symmetric per-tensor; pruning is global
magnitude pruning on convolutional / linear weight tensors.
"""
import copy
import numpy as np
import torch
import torch.nn as nn

QUANT_LAYERS = (nn.Conv2d, nn.Linear)


def _clone(model):
    """deepcopy a model after clearing any cached non-leaf feature tensor
    (SmallResNet caches ._feat for Grad-CAM, which is not deepcopy-able)."""
    f = getattr(model, "_feat", None)
    if hasattr(model, "_feat"):
        model._feat = None
    m = copy.deepcopy(model)
    if hasattr(model, "_feat"):
        model._feat = f
    return m


def _weight_modules(model):
    return [m for m in model.modules() if isinstance(m, QUANT_LAYERS)]


def uniform_quantize_tensor(w, bits):
    """Symmetric uniform quantization of tensor w to `bits` bits.
    Returns the de-quantized (reconstructed) tensor."""
    if bits >= 32:
        return w.clone()
    qmax = 2 ** (bits - 1) - 1          # symmetric signed range
    if qmax < 1:
        qmax = 1
    r = w.abs().max()
    if r == 0:
        return w.clone()
    scale = r / qmax
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return q * scale


def quantize_model(model, bits):
    m = _clone(model)
    with torch.no_grad():
        for mod in _weight_modules(m):
            mod.weight.copy_(uniform_quantize_tensor(mod.weight.data, bits))
    return m


def prune_model(model, sparsity):
    """Global magnitude pruning: zero the `sparsity` fraction of smallest-|w|
    weights across all conv/linear layers."""
    m = _clone(model)
    if sparsity <= 0:
        return m
    allw = torch.cat([mod.weight.data.abs().flatten()
                      for mod in _weight_modules(m)])
    k = int(sparsity * allw.numel())
    if k <= 0:
        return m
    thresh = torch.kthvalue(allw, k).values
    with torch.no_grad():
        for mod in _weight_modules(m):
            mask = (mod.weight.data.abs() > thresh).float()
            mod.weight.mul_(mask)
    return m


# ---------------------------------------------------------------------------
# Explanation-Aware Quantization (EAQ)
# ---------------------------------------------------------------------------
def relevance_scores(model, loader, device, max_batches=8):
    """Layer relevance R_l = E[ |w_l| * |dL/dw_l| ] summed per weight tensor,
    i.e. a first-order (Taylor) saliency of the loss w.r.t. each weight,
    aggregated to a per-layer scalar. Used to drive EAQ bit allocation."""
    model = model.to(device)
    model.eval()
    mods = _weight_modules(model)
    accum = [torch.zeros_like(mm.weight.data) for mm in mods]
    crit = nn.CrossEntropyLoss()
    nb = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        for i, mm in enumerate(mods):
            if mm.weight.grad is not None:
                accum[i] += (mm.weight.data.abs() * mm.weight.grad.abs())
        nb += 1
        if nb >= max_batches:
            break
    scores = np.array([a.sum().item() / max(nb, 1) for a in accum])
    return scores


def eaq_bit_allocation(scores, mean_bits, bmin=2, bmax=8):
    """Water-filling bit allocation. Given per-layer relevance R_l and a target
    average bit budget `mean_bits`, allocate b_l ~ b0 + 0.5*log2(R_l / Rbar)
    then round and project to satisfy sum(b_l * n_l)/sum(n_l) ~= mean_bits.
    Here we treat layers as equally weighted for the average (the paper's
    Theorem 4 derives the continuous solution; this is its rounded projection).
    """
    R = np.maximum(scores, 1e-12)
    logR = np.log2(R)
    b = mean_bits + 0.5 * (logR - logR.mean())
    b = np.clip(b, bmin, bmax)
    # iterative rounding to hit the average budget
    for _ in range(200):
        br = np.round(b)
        diff = br.mean() - mean_bits
        if abs(diff) < 1e-6:
            break
        # nudge the continuous solution against the residual
        b = np.clip(b - diff, bmin, bmax)
    return np.clip(np.round(b), bmin, bmax).astype(int)


def quantize_model_mixed(model, bit_list):
    """Quantize each conv/linear layer with its own bit-width from bit_list."""
    m = _clone(model)
    mods = _weight_modules(m)
    assert len(mods) == len(bit_list)
    with torch.no_grad():
        for mod, b in zip(mods, bit_list):
            mod.weight.copy_(uniform_quantize_tensor(mod.weight.data, int(b)))
    return m
