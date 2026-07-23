"""Post-hoc attribution methods and fidelity metrics.

All maps are returned as non-negative HxW arrays normalised to [0,1] so that
methods with different native scales can be compared on common footing.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _to_map(t, H, W):
    """Reduce a (C,H,W) or (H,W) tensor to a normalised HxW numpy heatmap."""
    a = t.detach().abs()
    if a.dim() == 3:
        a = a.sum(0)
    a = a.cpu().numpy().astype(np.float32)
    if a.shape != (H, W):
        a = np.array(_resize(a, H, W), dtype=np.float32)
    a = a - a.min()
    m = a.max()
    if m > 0:
        a = a / m
    return a


def _resize(a, H, W):
    t = torch.tensor(a)[None, None]
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def saliency(model, x, target, device):
    """Vanilla gradient saliency |dS_c/dx| aggregated over channels."""
    model.eval()
    x = x.clone().to(device).requires_grad_(True)
    out = model(x[None])
    s = out[0, target]
    model.zero_grad()
    s.backward()
    H, W = x.shape[-2:]
    return _to_map(x.grad[0], H, W)


def integrated_gradients(model, x, target, device, steps=32, baseline=None):
    """Integrated Gradients along a straight-line path from baseline to x."""
    model.eval()
    x = x.to(device)
    if baseline is None:
        baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for a in np.linspace(0, 1, steps):
        xi = (baseline + a * (x - baseline)).clone().requires_grad_(True)
        out = model(xi[None])
        s = out[0, target]
        model.zero_grad()
        s.backward()
        total += xi.grad[0]
    ig = (x - baseline) * total / steps
    H, W = x.shape[-2:]
    return _to_map(ig, H, W)


def grad_cam(model, x, target, device):
    """Grad-CAM on the last convolutional block (model._feat)."""
    model.eval()
    x = x.clone().to(device).requires_grad_(True)
    out, feat = model(x[None], return_feat=True)
    s = out[0, target]
    model.zero_grad()
    s.backward()
    grads = feat.grad[0]                       # (C,h,w)
    weights = grads.mean(dim=(1, 2))           # GAP of gradients
    cam = torch.relu((weights[:, None, None] * feat[0]).sum(0))
    H, W = x.shape[-2:]
    return _to_map(cam, H, W)


def occlusion(model, x, target, device, patch=8, stride=4):
    """Occlusion sensitivity: drop in target score when a patch is masked."""
    model.eval()
    x = x.to(device)
    C, H, W = x.shape
    with torch.no_grad():
        base = F.softmax(model(x[None])[0], 0)[target].item()
    heat = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    for i in range(0, H - patch + 1, stride):
        for j in range(0, W - patch + 1, stride):
            xm = x.clone()
            xm[:, i:i + patch, j:j + patch] = 0.0
            with torch.no_grad():
                p = F.softmax(model(xm[None])[0], 0)[target].item()
            heat[i:i + patch, j:j + patch] += (base - p)
            cnt[i:i + patch, j:j + patch] += 1
    cnt[cnt == 0] = 1
    heat = np.maximum(heat / cnt, 0)
    a = heat - heat.min()
    if a.max() > 0:
        a = a / a.max()
    return a


METHODS = {
    "Saliency": saliency,
    "IG": integrated_gradients,
    "GradCAM": grad_cam,
    "Occlusion": occlusion,
}


# ---------------------------------------------------------------------------
# fidelity / faithfulness metrics
# ---------------------------------------------------------------------------
def spearman(a, b):
    from scipy.stats import spearmanr
    r, _ = spearmanr(a.flatten(), b.flatten())
    return 0.0 if np.isnan(r) else r


def pearson(a, b):
    af, bf = a.flatten(), b.flatten()
    af = af - af.mean(); bf = bf - bf.mean()
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return 0.0 if d == 0 else float(af @ bf / d)


def topk_iou(a, b, k=0.2):
    n = a.size
    kk = max(1, int(k * n))
    ia = np.zeros(n, bool); ib = np.zeros(n, bool)
    ia[np.argsort(a.flatten())[-kk:]] = True
    ib[np.argsort(b.flatten())[-kk:]] = True
    inter = np.logical_and(ia, ib).sum()
    union = np.logical_or(ia, ib).sum()
    return inter / union if union else 0.0


def ssim(a, b):
    """Global SSIM (single-window) between two normalised maps."""
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def mask_iou_gt(a, mask, k=0.2):
    """IoU of top-k attribution against ground-truth object mask."""
    n = a.size
    kk = max(1, int(k * n))
    ia = np.zeros(n, bool)
    ia[np.argsort(a.flatten())[-kk:]] = True
    ib = mask.flatten() > 0.5
    inter = np.logical_and(ia, ib).sum()
    union = np.logical_or(ia, ib).sum()
    return inter / union if union else 0.0


def pointing_game(a, mask):
    """1 if the argmax attribution pixel lies inside the GT mask."""
    idx = np.argmax(a)
    return float(mask.flatten()[idx] > 0.5)


def deletion_auc(model, x, target, heat, device, steps=16):
    """Faithfulness: progressively delete most-salient pixels; lower AUC of the
    target-probability curve => more faithful."""
    model.eval()
    x = x.to(device)
    C, H, W = x.shape
    order = np.ascontiguousarray(np.argsort(heat.flatten())[::-1])
    probs = []
    xm = x.clone()
    n = H * W
    chunk = max(1, n // steps)
    with torch.no_grad():
        p0 = F.softmax(model(xm[None])[0], 0)[target].item()
    probs.append(p0)
    flat = xm.reshape(C, -1)
    for s in range(steps):
        idx = order[s * chunk:(s + 1) * chunk]
        flat[:, idx] = 0.0
        with torch.no_grad():
            p = F.softmax(model(flat.reshape(C, H, W)[None])[0], 0)[target].item()
        probs.append(p)
    return float(np.trapezoid(probs, dx=1.0 / (len(probs) - 1)))


def insertion_auc(model, x, target, heat, device, steps=16):
    """Faithfulness: progressively insert most-salient pixels starting from a blank image;
    higher AUC of the target-probability curve => more faithful."""
    model.eval()
    x = x.to(device)
    C, H, W = x.shape
    order = np.ascontiguousarray(np.argsort(heat.flatten())[::-1])
    probs = []
    xm = torch.zeros_like(x)
    n = H * W
    chunk = max(1, n // steps)
    with torch.no_grad():
        p0 = F.softmax(model(xm[None])[0], 0)[target].item()
    probs.append(p0)
    flat = xm.reshape(C, -1)
    flat_orig = x.reshape(C, -1)
    for s in range(steps):
        idx = order[s * chunk:(s + 1) * chunk]
        flat[:, idx] = flat_orig[:, idx]
        with torch.no_grad():
            p = F.softmax(model(flat.reshape(C, H, W)[None])[0], 0)[target].item()
        probs.append(p)
    return float(np.trapezoid(probs, dx=1.0 / (len(probs) - 1)))

