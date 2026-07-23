import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "font.family": "serif", "axes.grid": True,
    "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False
})

R = "./results/"
F = "./paper/figs/"
os.makedirs(F, exist_ok=True)

METH = ["Saliency", "IG", "GradCAM", "Occlusion"]
CMAP = {"Saliency": "#1f77b4", "IG": "#d62728", "GradCAM": "#2ca02c", "Occlusion": "#9467bd"}
MARK = {"Saliency": "o", "IG": "s", "GradCAM": "^", "Occlusion": "D"}

def plot_accuracy(df_cifar, df_imagenette):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    
    # CIFAR-10
    q_c = df_cifar[df_cifar.kind == "quant"].groupby("level").acc.first().sort_index()
    p_c = df_cifar[df_cifar.kind == "prune"].groupby("level").acc.first().sort_index()
    b_c = df_cifar[df_cifar.kind == "baseline"].acc.iloc[0]
    
    ax[0, 0].plot(q_c.index, q_c.values * 100, "o-", color="#d62728", lw=2)
    ax[0, 0].axhline(b_c * 100, ls="--", color="gray", label="Baseline")
    ax[0, 0].set_xlabel("bit-width $b$")
    ax[0, 0].set_ylabel("CIFAR-10 test accuracy (%)")
    ax[0, 0].set_title("(a) CIFAR-10 Quantization")
    ax[0, 0].invert_xaxis()
    ax[0, 0].legend()
    
    ax[0, 1].plot(p_c.index * 100, p_c.values * 100, "s-", color="#1f77b4", lw=2)
    ax[0, 1].axhline(b_c * 100, ls="--", color="gray", label="Baseline")
    ax[0, 1].set_xlabel("sparsity (%)")
    ax[0, 1].set_ylabel("CIFAR-10 test accuracy (%)")
    ax[0, 1].set_title("(b) CIFAR-10 Pruning")
    ax[0, 1].legend()
    
    # ImageNette
    q_i = df_imagenette[df_imagenette.kind == "quant"].groupby("level").acc.first().sort_index()
    p_i = df_imagenette[df_imagenette.kind == "prune"].groupby("level").acc.first().sort_index()
    b_i = df_imagenette[df_imagenette.kind == "baseline"].acc.iloc[0]
    
    ax[1, 0].plot(q_i.index, q_i.values * 100, "o-", color="#d62728", lw=2)
    ax[1, 0].axhline(b_i * 100, ls="--", color="gray", label="Baseline")
    ax[1, 0].set_xlabel("bit-width $b$")
    ax[1, 0].set_ylabel("ImageNette test accuracy (%)")
    ax[1, 0].set_title("(c) ImageNette Quantization")
    ax[1, 0].invert_xaxis()
    ax[1, 0].legend()
    
    ax[1, 1].plot(p_i.index * 100, p_i.values * 100, "s-", color="#1f77b4", lw=2)
    ax[1, 1].axhline(b_i * 100, ls="--", color="gray", label="Baseline")
    ax[1, 1].set_xlabel("sparsity (%)")
    ax[1, 1].set_ylabel("ImageNette test accuracy (%)")
    ax[1, 1].set_title("(d) ImageNette Pruning")
    ax[1, 1].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(F, "accuracy_natural.pdf"))
    plt.close(fig)

def plot_fidelity_quant(df, dataset_name, filename):
    sub = df[df.kind == "quant"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for meth in METH:
        s = sub[sub.method == meth].sort_values("level")
        ax.plot(s.level, s.spearman, marker=MARK[meth], color=CMAP[meth], lw=2, label=meth)
    ax.set_xlabel("bit-width $b$")
    ax.set_ylabel("Explanation fidelity (Spearman $\\rho$)")
    ax.set_title(f"Fidelity vs. Quantization ({dataset_name})")
    ax.invert_xaxis()
    ax.legend()
    fig.savefig(os.path.join(F, filename))
    plt.close(fig)

def plot_fidelity_prune(df, dataset_name, filename):
    sub = df[df.kind == "prune"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for meth in METH:
        s = sub[sub.method == meth].sort_values("level")
        ax.plot(s.level * 100, s.spearman, marker=MARK[meth], color=CMAP[meth], lw=2, label=meth)
    ax.set_xlabel("sparsity (%)")
    ax.set_ylabel("Explanation fidelity (Spearman $\\rho$)")
    ax.set_title(f"Fidelity vs. Pruning ({dataset_name})")
    ax.legend()
    fig.savefig(os.path.join(F, filename))
    plt.close(fig)

def plot_eaq(df_eaq, dataset_name, filename):
    sub_eaq = df_eaq[df_eaq.method_type == "EAQ"]
    sub_uni = df_eaq[df_eaq.method_type == "Uniform"]
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    budgets = sorted(df_eaq.target_bits.unique())
    
    acc_eaq = [sub_eaq[sub_eaq.target_bits == b].acc.iloc[0] * 100 for b in budgets]
    acc_uni = [sub_uni[sub_uni.target_bits == b].acc.iloc[0] * 100 for b in budgets]
    
    ax[0].plot(budgets, acc_eaq, "o-", color="#2ca02c", lw=2, label="EAQ (ours)")
    ax[0].plot(budgets, acc_uni, "x--", color="#d62728", lw=2, label="Uniform")
    ax[0].set_xlabel("average bit budget $\\bar{B}$")
    ax[0].set_ylabel("test accuracy (%)")
    ax[0].set_title(f"(a) Accuracy - {dataset_name}")
    ax[0].invert_xaxis()
    ax[0].legend()
    
    gc_eaq = [sub_eaq[(sub_eaq.target_bits == b) & (sub_eaq.xai == "GradCAM")].fidelity.iloc[0] for b in budgets]
    gc_uni = [sub_uni[(sub_uni.target_bits == b) & (sub_uni.xai == "GradCAM")].fidelity.iloc[0] for b in budgets]
    
    ax[1].plot(budgets, gc_eaq, "o-", color="#2ca02c", lw=2, label="EAQ (ours)")
    ax[1].plot(budgets, gc_uni, "x--", color="#d62728", lw=2, label="Uniform")
    ax[1].set_xlabel("average bit budget $\\bar{B}$")
    ax[1].set_ylabel("Grad-CAM Spearman $\\rho$")
    ax[1].set_title(f"(b) Grad-CAM Fidelity - {dataset_name}")
    ax[1].invert_xaxis()
    ax[1].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(F, filename))
    plt.close(fig)

def main():
    print("Loading natural metrics CSV files from:", R)
    try:
        df_cifar = pd.read_csv(os.path.join(R, "CIFAR-10_metrics.csv"))
        df_imagenette = pd.read_csv(os.path.join(R, "ImageNette_metrics.csv"))
        df_eaq_c = pd.read_csv(os.path.join(R, "CIFAR-10_eaq.csv"))
        df_eaq_i = pd.read_csv(os.path.join(R, "ImageNette_eaq.csv"))
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}. Make sure run_natural.py has run successfully.")
        return
        
    print("Generating accuracy plots...")
    plot_accuracy(df_cifar, df_imagenette)
    
    print("Generating fidelity plots for CIFAR-10...")
    plot_fidelity_quant(df_cifar, "CIFAR-10", "fidelity_quant_cifar.pdf")
    plot_fidelity_prune(df_cifar, "CIFAR-10", "fidelity_prune_cifar.pdf")
    
    print("Generating fidelity plots for ImageNette...")
    plot_fidelity_quant(df_imagenette, "ImageNette", "fidelity_quant_imagenette.pdf")
    plot_fidelity_prune(df_imagenette, "ImageNette", "fidelity_prune_imagenette.pdf")
    
    print("Generating EAQ plots...")
    plot_eaq(df_eaq_c, "CIFAR-10", "eaq_cifar.pdf")
    plot_eaq(df_eaq_i, "ImageNette", "eaq_imagenette.pdf")
    
    print("ALL NATURAL IMAGES FIGURES GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
