"""Resumable compression sweep. Appends one row-block per (variant, method) to
compression_metrics.csv. Re-running skips already-completed variants, so the
sweep can be advanced across several short invocations.
"""
import os, time, numpy as np, torch, pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from dataset import make_dataset, normalise, NUM_CLASSES
from model import SmallResNet
import compress as C
import attributions as A

torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"; OUT = "../results/compression_metrics.csv"
BITS = [8, 6, 5, 4, 3, 2]; SPARS = [0.3, 0.5, 0.7, 0.8, 0.9]
N_EXPL = 90          # images for fidelity/localisation
N_DEL = 40           # images for deletion AUC (expensive)
IG_STEPS = 20; DEL_STEPS = 12
TIME_BUDGET = 240    # seconds per invocation before graceful stop


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


def main():
    t0 = time.time()
    base, mean, std, base_acc = load_baseline()
    Xte, Yte, Mte = make_dataset(1000, seed=99)
    Xn, _, _ = normalise(Xte, mean, std)
    loader = DataLoader(TensorDataset(torch.tensor(Xn).permute(0, 3, 1, 2),
                        torch.tensor(Yte)), batch_size=128)
    xs = torch.tensor(Xn[:N_EXPL]).permute(0, 3, 1, 2)
    ys = Yte[:N_EXPL]; ms = Mte[:N_EXPL]
    methods = list(A.METHODS.keys())

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT).scheme.unique())
    header = not os.path.exists(OUT)

    # cache baseline maps to npz (persist across runs)
    bm_path = "../results/base_maps.npz"
    if os.path.exists(bm_path):
        bm = np.load(bm_path)
        base_maps = [{m: bm[f"{i}_{m}"] for m in methods} for i in range(N_EXPL)]
    else:
        base_maps = [all_maps(base, xs[i], int(ys[i])) for i in range(N_EXPL)]
        flat = {f"{i}_{m}": base_maps[i][m] for i in range(N_EXPL) for m in methods}
        np.savez(bm_path, **flat)
        print(f"baseline maps cached t={time.time()-t0:.1f}")

    # baseline row
    if "baseline" not in done:
        rows = []
        for meth in methods:
            iou = np.mean([A.mask_iou_gt(base_maps[i][meth], ms[i]) for i in range(N_EXPL)])
            pg = np.mean([A.pointing_game(base_maps[i][meth], ms[i]) for i in range(N_EXPL)])
            dele = np.mean([A.deletion_auc(base, xs[i], int(ys[i]), base_maps[i][meth],
                            DEV, steps=DEL_STEPS) for i in range(N_DEL)])
            rows.append(dict(scheme="baseline", kind="baseline", level=32, method=meth,
                             acc=base_acc, spearman=1.0, pearson=1.0, iou_self=1.0,
                             ssim=1.0, gt_iou=iou, pointing=pg, deletion=dele))
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=header, index=False)
        header = False; done.add("baseline")
        print(f"baseline row done t={time.time()-t0:.1f}")

    variants = [("quant", b) for b in BITS] + [("prune", s) for s in SPARS]
    for kind, level in variants:
        scheme = f"{kind}-{level}"
        if scheme in done:
            continue
        if time.time() - t0 > TIME_BUDGET:
            print(f"TIME BUDGET hit, stopping before {scheme}. Re-run to continue.")
            return
        vm = C.quantize_model(base, level) if kind == "quant" else C.prune_model(base, level)
        vm.eval()
        acc = accuracy(vm, loader)
        agg = {m: dict(sp=[], pe=[], iou=[], ss=[], gt=[], pg=[], de=[]) for m in methods}
        for i in range(N_EXPL):
            vmaps = all_maps(vm, xs[i], int(ys[i]))
            for meth in methods:
                a, b = vmaps[meth], base_maps[i][meth]
                agg[meth]["sp"].append(A.spearman(a, b))
                agg[meth]["pe"].append(A.pearson(a, b))
                agg[meth]["iou"].append(A.topk_iou(a, b))
                agg[meth]["ss"].append(A.ssim(a, b))
                agg[meth]["gt"].append(A.mask_iou_gt(a, ms[i]))
                agg[meth]["pg"].append(A.pointing_game(a, ms[i]))
                if i < N_DEL:
                    agg[meth]["de"].append(
                        A.deletion_auc(vm, xs[i], int(ys[i]), a, DEV, steps=DEL_STEPS))
        rows = []
        for meth in methods:
            d = agg[meth]
            rows.append(dict(scheme=scheme, kind=kind, level=level, method=meth, acc=acc,
                             spearman=np.mean(d["sp"]), pearson=np.mean(d["pe"]),
                             iou_self=np.mean(d["iou"]), ssim=np.mean(d["ss"]),
                             gt_iou=np.mean(d["gt"]), pointing=np.mean(d["pg"]),
                             deletion=np.mean(d["de"])))
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=header, index=False)
        header = False; done.add(scheme)
        print(f"{scheme} acc={acc:.4f} GC-sp={np.mean(agg['GradCAM']['sp']):.3f} "
              f"t={time.time()-t0:.1f}")
    print("compression sweep COMPLETE")


if __name__ == "__main__":
    main()
