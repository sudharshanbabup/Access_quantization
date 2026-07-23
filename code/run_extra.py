"""Extended experiments requested in peer review, run on the released
SynthShapes pipeline + baseline.pt (no external data required):

  A. Statistical significance: per-image PAIRED comparison of EAQ vs uniform
     quantization at matched budgets -> Wilcoxon signed-rank p-values.
  B. Multi-seed confidence intervals for the headline quantization fidelity.
  C. Runtime (per-image attribution latency, compression-op latency) and
     model compression ratio at each bit-width.
  D. Sample-level consistency: skewness / excess-kurtosis of the per-image
     deletion-AUC distribution.

Outputs CSVs to extra_out/.
"""
import os, time, json, numpy as np, torch, pandas as pd
from scipy.stats import wilcoxon, skew, kurtosis, t as student_t
from torch.utils.data import TensorDataset, DataLoader
from dataset import make_dataset, normalise, NUM_CLASSES
from model import SmallResNet
import compress as C
import attributions as A

torch.set_num_threads(2)
DEV = "cpu"
OUT = "extra_out"; os.makedirs(OUT, exist_ok=True)
METHODS = ["Saliency", "IG", "GradCAM", "Occlusion"]
IG_STEPS = 16


def load_baseline():
    ck = torch.load("baseline.pt", weights_only=False)
    m = SmallResNet(NUM_CLASSES); m.load_state_dict(ck["model"]); m.eval()
    return m, ck["mean"], ck["std"], ck["acc"]


def all_maps(model, x, target):
    return {
        "Saliency": A.saliency(model, x, target, DEV),
        "IG": A.integrated_gradients(model, x, target, DEV, steps=IG_STEPS),
        "GradCAM": A.grad_cam(model, x, target, DEV),
        "Occlusion": A.occlusion(model, x, target, DEV, patch=8, stride=8),
    }


def accuracy(model, loader):
    model.eval(); c = t = 0
    with torch.no_grad():
        for xb, yb in loader:
            c += (model(xb).argmax(1) == yb).sum().item(); t += yb.numel()
    return c / t


def eval_set(mean, std, n, seed):
    Xte, Yte, Mte = make_dataset(n, seed=seed)
    Xn, _, _ = normalise(Xte, mean, std)
    xs = torch.tensor(Xn[:n]).permute(0, 3, 1, 2)
    return xs, Yte[:n], Mte[:n], Xn


# ---------------------------------------------------------------------------
def run_significance(base, mean, std, budgets=(5, 4, 3), n=120):
    """Per-image paired EAQ vs Uniform fidelity + Wilcoxon signed-rank."""
    t0 = time.time()
    xs, ys, ms, Xn = eval_set(mean, std, n, seed=99)
    # relevance for EAQ from calibration split
    Xc, Yc, _ = make_dataset(512, seed=7)
    Xcn, _, _ = normalise(Xc, mean, std)
    cal = DataLoader(TensorDataset(torch.tensor(Xcn).permute(0, 3, 1, 2),
                                   torch.tensor(Yc)), batch_size=64)
    scores = C.relevance_scores(base, cal, DEV)
    base_maps = [all_maps(base, xs[i], int(ys[i])) for i in range(n)]
    print(f"[A] baseline maps done t={time.time()-t0:.0f}s")
    rows = []
    for tb in budgets:
        alloc = C.eaq_bit_allocation(scores, tb)
        eaq = C.quantize_model_mixed(base, alloc); eaq.eval()
        uni = C.quantize_model(base, tb); uni.eval()
        per = {m: {"EAQ": [], "Uniform": []} for m in METHODS}
        for i in range(n):
            em = all_maps(eaq, xs[i], int(ys[i]))
            um = all_maps(uni, xs[i], int(ys[i]))
            for meth in METHODS:
                per[meth]["EAQ"].append(A.spearman(em[meth], base_maps[i][meth]))
                per[meth]["Uniform"].append(A.spearman(um[meth], base_maps[i][meth]))
        for meth in METHODS:
            e = np.array(per[meth]["EAQ"]); u = np.array(per[meth]["Uniform"])
            diff = e - u
            try:
                stat, p = wilcoxon(e, u, alternative="greater")
            except ValueError:
                stat, p = float("nan"), float("nan")
            rows.append(dict(budget=tb, method=meth,
                             eaq_mean=e.mean(), uni_mean=u.mean(),
                             delta=diff.mean(), delta_sd=diff.std(ddof=1),
                             wilcoxon_stat=stat, p_value=p, n=n))
        print(f"[A] budget {tb} alloc={list(alloc)} t={time.time()-t0:.0f}s")
    pd.DataFrame(rows).to_csv(f"{OUT}/significance.csv", index=False)
    return rows


def run_seed_ci(base, mean, std, bits=(8, 6, 5, 4, 3), seeds=(21, 22, 23, 24, 25), n=60):
    """Multi-seed mean +/- 95% CI of quant fidelity (Grad-CAM, IG, Saliency)."""
    t0 = time.time()
    meths = ["Saliency", "IG", "GradCAM"]
    per_seed = {b: {m: [] for m in meths} for b in bits}
    for sd in seeds:
        xs, ys, ms, _ = eval_set(mean, std, n, seed=sd)
        bmaps = [{m: (A.saliency(base, xs[i], int(ys[i]), DEV) if m == "Saliency"
                      else A.integrated_gradients(base, xs[i], int(ys[i]), DEV, steps=IG_STEPS)
                      if m == "IG" else A.grad_cam(base, xs[i], int(ys[i]), DEV))
                  for m in meths} for i in range(n)]
        for b in bits:
            qm = C.quantize_model(base, b); qm.eval()
            acc = {m: [] for m in meths}
            for i in range(n):
                mp = {"Saliency": A.saliency(qm, xs[i], int(ys[i]), DEV),
                      "IG": A.integrated_gradients(qm, xs[i], int(ys[i]), DEV, steps=IG_STEPS),
                      "GradCAM": A.grad_cam(qm, xs[i], int(ys[i]), DEV)}
                for m in meths:
                    acc[m].append(A.spearman(mp[m], bmaps[i][m]))
            for m in meths:
                per_seed[b][m].append(np.mean(acc[m]))
        print(f"[B] seed {sd} done t={time.time()-t0:.0f}s")
    rows = []
    for b in bits:
        for m in meths:
            v = np.array(per_seed[b][m]); k = len(v)
            ci = student_t.ppf(0.975, k - 1) * v.std(ddof=1) / np.sqrt(k)
            rows.append(dict(bits=b, method=m, mean=v.mean(), ci95=ci,
                             lo=v.mean() - ci, hi=v.mean() + ci, seeds=k))
    pd.DataFrame(rows).to_csv(f"{OUT}/seed_ci.csv", index=False)
    return rows


def run_runtime(base, mean, std, n=30):
    """Per-image attribution latency, compression-op latency, compression ratio."""
    xs, ys, ms, _ = eval_set(mean, std, n, seed=99)
    # attribution latency
    rt = {}
    for meth, fn in [("Saliency", lambda i: A.saliency(base, xs[i], int(ys[i]), DEV)),
                     ("IG", lambda i: A.integrated_gradients(base, xs[i], int(ys[i]), DEV, steps=IG_STEPS)),
                     ("GradCAM", lambda i: A.grad_cam(base, xs[i], int(ys[i]), DEV)),
                     ("Occlusion", lambda i: A.occlusion(base, xs[i], int(ys[i]), DEV, patch=8, stride=8))]:
        _ = fn(0)  # warmup
        t0 = time.time()
        for i in range(n):
            _ = fn(i)
        rt[meth] = 1000.0 * (time.time() - t0) / n  # ms/image
    # compression-op latency
    t0 = time.time(); _ = C.quantize_model(base, 4); q_ms = 1000 * (time.time() - t0)
    t0 = time.time(); _ = C.prune_model(base, 0.5); p_ms = 1000 * (time.time() - t0)
    nparams = sum(p.numel() for p in base.parameters())
    rows = [dict(kind="attribution_ms_per_image", **{k: round(v, 2) for k, v in rt.items()}),
            dict(kind="compression_op_ms", quantize=round(q_ms, 2), prune=round(p_ms, 2))]
    pd.DataFrame(rows).to_csv(f"{OUT}/runtime.csv", index=False)
    # compression ratio: weight memory FP32 vs b-bit (weights of conv/linear)
    wparams = sum(m.weight.numel() for m in base.modules()
                  if isinstance(m, C.QUANT_LAYERS))
    crows = []
    for b in [8, 6, 5, 4, 3, 2]:
        crows.append(dict(bits=b, weight_params=wparams,
                          fp32_KB=round(wparams * 32 / 8 / 1024, 1),
                          quant_KB=round(wparams * b / 8 / 1024, 1),
                          ratio=round(32.0 / b, 2)))
    pd.DataFrame(crows).to_csv(f"{OUT}/compression_ratio.csv", index=False)
    return rows, crows, nparams, wparams


def run_skew_kurt(base, mean, std, bits=(8, 5, 4, 3), n=60):
    """Distribution of per-image deletion AUC -> skew / excess kurtosis."""
    xs, ys, ms, _ = eval_set(mean, std, n, seed=99)
    meths = ["Saliency", "GradCAM"]
    rows = []
    variants = [("baseline", 32, base)] + [("quant", b, C.quantize_model(base, b)) for b in bits]
    for kind, lvl, vm in variants:
        vm.eval()
        for meth in meths:
            aucs = []
            for i in range(n):
                if meth == "Saliency":
                    h = A.saliency(vm, xs[i], int(ys[i]), DEV)
                else:
                    h = A.grad_cam(vm, xs[i], int(ys[i]), DEV)
                aucs.append(A.deletion_auc(vm, xs[i], int(ys[i]), h, DEV, steps=12))
            aucs = np.array(aucs)
            rows.append(dict(level=lvl, method=meth, mean_auc=aucs.mean(),
                             auc_skew=float(skew(aucs)),
                             auc_kurt=float(kurtosis(aucs)), n=n))
    pd.DataFrame(rows).to_csv(f"{OUT}/skew_kurt.csv", index=False)
    return rows


if __name__ == "__main__":
    T = time.time()
    base, mean, std, base_acc = load_baseline()
    print("baseline acc", base_acc)
    sig = run_significance(base, mean, std)
    ci = run_seed_ci(base, mean, std)
    rt, cr, npar, wpar = run_runtime(base, mean, std)
    sk = run_skew_kurt(base, mean, std)
    print("TOTAL", round(time.time() - T, 1), "s")
