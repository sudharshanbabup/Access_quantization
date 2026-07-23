import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
from model import SmallResNet
from dataset import make_dataset, normalise, NUM_CLASSES, CLASSES
import compress as C
import attributions as A

plt.rcParams.update({
    "font.size": 11, "font.family": "serif", "axes.grid": True,
    "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False})
R = "../results/"; F = "../figs/"
METH = ["Saliency", "IG", "GradCAM", "Occlusion"]
CMAP = {"Saliency": "#1f77b4", "IG": "#d62728",
        "GradCAM": "#2ca02c", "Occlusion": "#9467bd"}
MARK = {"Saliency": "o", "IG": "s", "GradCAM": "^", "Occlusion": "D"}


def load():
    return (pd.read_csv(R + "compression_metrics.csv"),
            pd.read_csv(R + "noise_stability.csv"),
            pd.read_csv(R + "eaq.csv"))


# ---------- Fig: dataset samples ----------
def fig_dataset():
    X, Y, M = make_dataset(12, seed=7)
    fig, ax = plt.subplots(2, 6, figsize=(11, 4))
    for i in range(12):
        r, c = divmod(i, 6)
        ax[r, c].imshow(X[i]); ax[r, c].set_title(CLASSES[Y[i]], fontsize=9)
        ax[r, c].contour(M[i], levels=[0.5], colors="w", linewidths=1.0)
        ax[r, c].axis("off")
    fig.suptitle("SynthShapes-32 samples (white contour = ground-truth object mask)",
                 fontsize=11)
    fig.savefig(F + "dataset.pdf"); plt.close(fig)


# ---------- Fig: accuracy vs compression ----------
def fig_accuracy(df):
    q = df[df.kind == "quant"].groupby("level").acc.first().sort_index()
    p = df[df.kind == "prune"].groupby("level").acc.first().sort_index()
    b = df[df.kind == "baseline"].acc.iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    ax[0].plot(q.index, q.values * 100, "o-", color="#d62728", lw=2)
    ax[0].axhline(b * 100, ls="--", color="gray", label="FP32 baseline")
    ax[0].set_xlabel("bit-width $b$"); ax[0].set_ylabel("test accuracy (%)")
    ax[0].set_title("(a) Uniform quantization"); ax[0].invert_xaxis(); ax[0].legend()
    ax[1].plot(p.index * 100, p.values * 100, "s-", color="#1f77b4", lw=2)
    ax[1].axhline(b * 100, ls="--", color="gray", label="FP32 baseline")
    ax[1].set_xlabel("sparsity (%)"); ax[1].set_ylabel("test accuracy (%)")
    ax[1].set_title("(b) Magnitude pruning"); ax[1].legend()
    fig.savefig(F + "accuracy.pdf"); plt.close(fig)


# ---------- Fig: fidelity (spearman) vs bit-width ----------
def _line_vs_level(df, kind, metric, xlabel, title, fname, xscale=1, invert=False):
    sub = df[df.kind == kind]
    fig, ax = plt.subplots(figsize=(6, 4))
    for meth in METH:
        s = sub[sub.method == meth].sort_values("level")
        ax.plot(s.level * xscale, s[metric], marker=MARK[meth], color=CMAP[meth],
                lw=2, label=meth)
    ax.set_xlabel(xlabel); ax.set_ylabel(metric)
    ax.set_title(title); ax.legend()
    if invert:
        ax.invert_xaxis()
    fig.savefig(F + fname); plt.close(fig)


def fig_fidelity_quant(df):
    _line_vs_level(df, "quant", "spearman", "bit-width $b$",
                   "Explanation fidelity (Spearman $\\rho$) vs. quantization",
                   "fidelity_quant.pdf", invert=True)


def fig_fidelity_prune(df):
    _line_vs_level(df, "prune", "spearman", "sparsity (%)",
                   "Explanation fidelity (Spearman $\\rho$) vs. pruning",
                   "fidelity_prune.pdf", xscale=100)


# ---------- Fig: IoU + SSIM vs bit-width ----------
def fig_iou_ssim(df):
    sub = df[df.kind == "quant"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for meth in METH:
        s = sub[sub.method == meth].sort_values("level")
        ax[0].plot(s.level, s.iou_self, marker=MARK[meth], color=CMAP[meth], lw=2, label=meth)
        ax[1].plot(s.level, s.ssim, marker=MARK[meth], color=CMAP[meth], lw=2, label=meth)
    for a, t, yl in zip(ax, ["(a) Top-20% mask IoU", "(b) Heatmap SSIM"],
                        ["IoU", "SSIM"]):
        a.set_xlabel("bit-width $b$"); a.set_ylabel(yl); a.set_title(t)
        a.invert_xaxis(); a.legend(fontsize=9)
    fig.savefig(F + "iou_ssim.pdf"); plt.close(fig)


# ---------- Fig: localisation vs GT ----------
def fig_localisation(df):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for kind, a, xl, xs, inv in [("quant", ax[0], "bit-width $b$", 1, True),
                                 ("prune", ax[1], "sparsity (%)", 100, False)]:
        sub = df[df.kind == kind]
        base = df[df.kind == "baseline"]
        for meth in METH:
            s = sub[sub.method == meth].sort_values("level")
            ax_ = a
            ax_.plot(s.level * xs, s.gt_iou, marker=MARK[meth], color=CMAP[meth],
                     lw=2, label=meth)
        a.set_xlabel(xl); a.set_ylabel("GT-mask IoU (top-20%)")
        a.set_title(f"({'a' if kind=='quant' else 'b'}) {kind}")
        if inv: a.invert_xaxis()
        a.legend(fontsize=8)
    fig.savefig(F + "localisation.pdf"); plt.close(fig)


# ---------- Fig: faithfulness (deletion AUC) ----------
def fig_deletion(df):
    sub = df[df.kind == "quant"]; base = df[df.kind == "baseline"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for meth in METH:
        s = sub[sub.method == meth].sort_values("level")
        ax.plot(s.level, s.deletion, marker=MARK[meth], color=CMAP[meth], lw=2, label=meth)
        b = base[base.method == meth].deletion.iloc[0]
        ax.scatter([9], [b], color=CMAP[meth], marker="*", s=90, zorder=5)
    ax.set_xlabel("bit-width $b$ ")
    ax.set_ylabel("deletion AUC  (lower = more faithful)")
    ax.set_title("Faithfulness under quantization"); ax.invert_xaxis(); ax.legend()
    fig.savefig(F + "deletion.pdf"); plt.close(fig)


# ---------- Fig: noise stability ----------
def fig_noise(ndf):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    d = ndf[ndf.sigma == 0.10]
    for meth in METH:
        s = d[d.method == meth].sort_values("level")
        ax[0].plot(s.level, s.stability, marker=MARK[meth], color=CMAP[meth], lw=2, label=meth)
    ax[0].set_xlabel("bit-width $b$"); ax[0].set_ylabel("stability $\\rho$ ($\\sigma{=}0.1$)")
    ax[0].set_title("(a) Attribution stability vs. bit-width"); ax[0].invert_xaxis(); ax[0].legend(fontsize=8)
    dm = ndf[ndf.method == "GradCAM"]
    for lv in sorted(dm.level.unique()):
        s = dm[dm.level == lv].sort_values("sigma")
        lab = "FP32" if lv == 32 else f"{lv}-bit"
        ax[1].plot(s.sigma, s.stability, "o-", lw=2, label=lab)
    ax[1].set_xlabel("input noise $\\sigma$"); ax[1].set_ylabel("Grad-CAM stability $\\rho$")
    ax[1].set_title("(b) Grad-CAM stability vs. noise"); ax[1].legend(fontsize=8, ncol=2)
    fig.savefig(F + "noise.pdf"); plt.close(fig)


# ---------- Fig: EAQ vs uniform ----------
def fig_eaq(edf):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    accs = edf.groupby(["target_bits", "method_type"]).acc.first().unstack()
    accs = accs.sort_index()
    x = np.arange(len(accs.index)); w = 0.35
    ax[0].bar(x - w/2, accs["Uniform"] * 100, w, label="Uniform", color="#9aa0a6")
    ax[0].bar(x + w/2, accs["EAQ"] * 100, w, label="EAQ (ours)", color="#2ca02c")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{b}-bit avg" for b in accs.index])
    ax[0].set_ylabel("test accuracy (%)"); ax[0].set_title("(a) Accuracy")
    ax[0].set_ylim(min(accs.min())*100 - 3, 100); ax[0].legend()
    fdel = edf[edf.xai == "GradCAM"].groupby(["target_bits", "method_type"]).fidelity.first().unstack().sort_index()
    ax[1].bar(x - w/2, fdel["Uniform"], w, label="Uniform", color="#9aa0a6")
    ax[1].bar(x + w/2, fdel["EAQ"], w, label="EAQ (ours)", color="#2ca02c")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{b}-bit avg" for b in fdel.index])
    ax[1].set_ylabel("Grad-CAM fidelity $\\rho$"); ax[1].set_title("(b) Explanation fidelity")
    ax[1].legend()
    fig.savefig(F + "eaq.pdf"); plt.close(fig)


# ---------- Fig: theory bound validation ----------
def fig_theory(df):
    sub = df[(df.kind == "quant") & (df.method == "GradCAM")].sort_values("level")
    b = sub.level.values.astype(float); f = sub.spearman.values
    # fit F = 1 - kappa * 4^{-b}
    from scipy.optimize import curve_fit
    model = lambda bb, k: 1 - k * np.power(4.0, -bb)
    kappa, _ = curve_fit(model, b, f, p0=[10.0], maxfev=10000)
    bb = np.linspace(b.min(), b.max(), 100)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(b, f, color="#2ca02c", s=60, zorder=5, label="empirical (Grad-CAM)")
    ax.plot(bb, model(bb, *kappa), "k--", lw=2,
            label=f"Thm.~3 bound $1-\\kappa 4^{{-b}}$, $\\kappa={kappa[0]:.2f}$")
    ax.set_xlabel("bit-width $b$"); ax.set_ylabel("fidelity $\\rho$")
    ax.set_title("Empirical validation of the fidelity--bit-width law")
    ax.invert_xaxis(); ax.legend()
    fig.savefig(F + "theory.pdf"); plt.close(fig)
    return float(kappa[0])


# ---------- Fig: qualitative maps ----------
def fig_qualitative():
    ck = torch.load("baseline.pt", weights_only=False)
    base = SmallResNet(NUM_CLASSES); base.load_state_dict(ck["model"]); base.eval()
    dat = np.load(R + "qualitative.npz")
    Xn, Y, M = dat["X"], dat["Y"], dat["M"]
    q4 = C.quantize_model(base, 4); q2 = C.quantize_model(base, 2)
    models = [("FP32", base), ("4-bit", q4), ("2-bit", q2)]
    idxs = [0, 1, 2]
    fig = plt.figure(figsize=(11, 8.5))
    gs = gridspec.GridSpec(len(idxs), 1 + 3 * 2, wspace=0.12, hspace=0.25)
    for r, ix in enumerate(idxs):
        x = torch.tensor(Xn[ix]).permute(2, 0, 1); t = int(Y[ix])
        raw = (Xn[ix] - Xn[ix].min()) / (np.ptp(Xn[ix]) + 1e-6)
        a0 = fig.add_subplot(gs[r, 0]); a0.imshow(raw)
        a0.contour(M[ix], levels=[0.5], colors="w", linewidths=1)
        a0.set_ylabel(CLASSES[Y[ix]], fontsize=10); a0.set_xticks([]); a0.set_yticks([])
        if r == 0: a0.set_title("input", fontsize=10)
        col = 1
        for meth in ["GradCAM", "IG"]:
            for mname, mdl in models:
                a = fig.add_subplot(gs[r, col])
                mp = A.METHODS[meth](mdl, x, t, "cpu")
                a.imshow(raw, alpha=0.5); a.imshow(mp, cmap="jet", alpha=0.55)
                a.axis("off")
                if r == 0: a.set_title(f"{meth}\n{mname}", fontsize=8.5)
                col += 1
    fig.suptitle("Qualitative attributions: baseline vs. 4-bit vs. 2-bit quantization",
                 fontsize=12)
    fig.savefig(F + "qualitative.pdf"); plt.close(fig)


def main():
    df, ndf, edf = load()
    fig_dataset()
    fig_accuracy(df)
    fig_fidelity_quant(df)
    fig_fidelity_prune(df)
    fig_iou_ssim(df)
    fig_localisation(df)
    fig_deletion(df)
    fig_noise(ndf)
    fig_eaq(edf)
    kappa = fig_theory(df)
    fig_qualitative()
    print("kappa fit =", kappa)
    print("figures written")


if __name__ == "__main__":
    main()
