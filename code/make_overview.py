import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "serif", "font.size": 10})

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                 fc=fc, ec="#333333", lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.3)


def arrow(x1, y1, x2, y2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=14, lw=1.4, color="#444444"))


box(1, 20, 15, 10, "Train baseline\nCNN $f_{\\theta}$", "#dbe9f6")
box(20, 20, 17, 10, "Compression $\\mathcal{C}$\n(quantize $Q_b$ /\nprune $P_s$)", "#f6e2c9")
box(41, 20, 18, 10, "Attribution\n$\\varphi(\\cdot;\\tilde\\theta)$\n(Sal/IG/GC/Occ)", "#dcefd7")
box(63, 20, 17, 10, "Fidelity metrics\n$\\rho$, IoU, SSIM,\ndeletion AUC", "#efd7e8")
box(84, 20, 15, 10, "Robustness\nordering &\nlaw $1{-}\\kappa4^{-b}$", "#e8e2f2")

box(30, 3, 26, 9, "Explanation-Aware Quantization (EAQ)\nbit allocation $b_l^\\star=\\bar B+\\frac{1}{2}\\log_2(\\omega_l/\\bar\\omega_G)$",
    "#f6d7d2")

arrow(16, 25, 20, 25)
arrow(37, 25, 41, 25)
arrow(59, 25, 63, 25)
arrow(80, 25, 84, 25)
# feedback loop EAQ -> compression
arrow(43, 12, 28.5, 20)
ax.text(35.5, 15.6, "relevance-guided\nre-quantization", fontsize=7.8, color="#a11",
        ha="center")
arrow(70, 20, 56, 12)
ax.text(64.5, 15.2, "sensitivity\n$\\omega_l$", fontsize=7.8, color="#a11", ha="center")

fig.savefig("../figs/overview.pdf", bbox_inches="tight"); print("overview done")
