import time, json, numpy as np, torch, pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from dataset import make_dataset, normalise, NUM_CLASSES, CLASSES
from model import SmallResNet
import compress as C
import attributions as A

torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"
OUT = "../results/"

BITS = [8, 6, 5, 4, 3, 2]
SPARS = [0.3, 0.5, 0.7, 0.8, 0.9]
N_EXPL = 120          # images used for attribution study
IG_STEPS = 24


def load_baseline():
    ck = torch.load("baseline.pt", weights_only=False)
    m = SmallResNet(NUM_CLASSES); m.load_state_dict(ck["model"]); m.eval()
    return m, ck["mean"], ck["std"], ck["acc"]


def test_loader(mean, std, n=1000, bs=128):
    Xte, Yte, Mte = make_dataset(n, seed=99)
    Xn, _, _ = normalise(Xte, mean, std)
    ds = TensorDataset(torch.tensor(Xn).permute(0, 3, 1, 2), torch.tensor(Yte))
    return DataLoader(ds, batch_size=bs), (Xn, Yte, Mte)


def accuracy(model, loader):
    model.eval(); c = t = 0
    with torch.no_grad():
        for xb, yb in loader:
            c += (model(xb).argmax(1) == yb).sum().item(); t += yb.numel()
    return c / t


def all_maps(model, x, target):
    return {
        "Saliency": A.saliency(model, x, target, DEV),
        "IG": A.integrated_gradients(model, x, target, DEV, steps=IG_STEPS),
        "GradCAM": A.grad_cam(model, x, target, DEV),
        "Occlusion": A.occlusion(model, x, target, DEV, patch=8, stride=8),
    }


def main():
    t0 = time.time()
    base, mean, std, base_acc = load_baseline()
    loader, (Xn, Yte, Mte) = test_loader(mean, std)
    print("baseline acc", base_acc)

    # fixed explanation image subset
    xs = torch.tensor(Xn[:N_EXPL]).permute(0, 3, 1, 2)
    ys = Yte[:N_EXPL]; ms = Mte[:N_EXPL]

    # ---- precompute baseline maps ----
    base_maps = []
    for i in range(N_EXPL):
        base_maps.append(all_maps(base, xs[i], int(ys[i])))
    print(f"baseline maps done  t={time.time()-t0:.1f}s")

    methods = list(A.METHODS.keys())

    # baseline localisation / faithfulness rows
    rows = []
    for meth in methods:
        iou = np.mean([A.mask_iou_gt(base_maps[i][meth], ms[i]) for i in range(N_EXPL)])
        pg = np.mean([A.pointing_game(base_maps[i][meth], ms[i]) for i in range(N_EXPL)])
        dele = np.mean([A.deletion_auc(base, xs[i], int(ys[i]), base_maps[i][meth], DEV)
                        for i in range(N_EXPL)])
        rows.append(dict(scheme="baseline", kind="baseline", level=32, method=meth,
                         acc=base_acc, spearman=1.0, pearson=1.0, iou_self=1.0,
                         ssim=1.0, gt_iou=iou, pointing=pg, deletion=dele))

    # ---- compression variants ----
    variants = [("quant", b, C.quantize_model(base, b)) for b in BITS] + \
               [("prune", s, C.prune_model(base, s)) for s in SPARS]

    for kind, level, vm in variants:
        vm.eval()
        acc = accuracy(vm, loader)
        agg = {meth: dict(sp=[], pe=[], iou=[], ss=[], gtiou=[], pg=[], de=[])
               for meth in methods}
        for i in range(N_EXPL):
            vmaps = all_maps(vm, xs[i], int(ys[i]))
            for meth in methods:
                a, b = vmaps[meth], base_maps[i][meth]
                agg[meth]["sp"].append(A.spearman(a, b))
                agg[meth]["pe"].append(A.pearson(a, b))
                agg[meth]["iou"].append(A.topk_iou(a, b))
                agg[meth]["ss"].append(A.ssim(a, b))
                agg[meth]["gtiou"].append(A.mask_iou_gt(a, ms[i]))
                agg[meth]["pg"].append(A.pointing_game(a, ms[i]))
                agg[meth]["de"].append(A.deletion_auc(vm, xs[i], int(ys[i]), a, DEV))
        for meth in methods:
            d = agg[meth]
            rows.append(dict(scheme=f"{kind}-{level}", kind=kind, level=level,
                             method=meth, acc=acc,
                             spearman=np.mean(d["sp"]), pearson=np.mean(d["pe"]),
                             iou_self=np.mean(d["iou"]), ssim=np.mean(d["ss"]),
                             gt_iou=np.mean(d["gtiou"]), pointing=np.mean(d["pg"]),
                             deletion=np.mean(d["de"])))
        print(f"{kind}-{level:>4}  acc={acc:.4f}  "
              f"GradCAM-sp={np.mean(agg['GradCAM']['sp']):.3f}  t={time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT + "compression_metrics.csv", index=False)

    # ---- noise stability (quant models) ----
    nrows = []
    sigmas = [0.05, 0.10, 0.20]
    rng = np.random.default_rng(3)
    for kind, level, vm in [("baseline", 32, base)] + \
            [("quant", b, C.quantize_model(base, b)) for b in BITS]:
        vm.eval()
        for sg in sigmas:
            stab = {meth: [] for meth in methods}
            for i in range(min(60, N_EXPL)):
                clean = all_maps(vm, xs[i], int(ys[i]))
                xn = xs[i] + torch.tensor(rng.normal(0, sg, size=xs[i].shape),
                                          dtype=torch.float32)
                noisy = all_maps(vm, xn, int(ys[i]))
                for meth in methods:
                    stab[meth].append(A.spearman(clean[meth], noisy[meth]))
            for meth in methods:
                nrows.append(dict(kind=kind, level=level, sigma=sg, method=meth,
                                  stability=np.mean(stab[meth])))
        print(f"noise {kind}-{level}  t={time.time()-t0:.1f}s")
    pd.DataFrame(nrows).to_csv(OUT + "noise_stability.csv", index=False)

    # ---- EAQ vs uniform ----
    small_loader = DataLoader(
        TensorDataset(torch.tensor(Xn[:512]).permute(0, 3, 1, 2),
                      torch.tensor(Yte[:512])), batch_size=64)
    scores = C.relevance_scores(base, small_loader, DEV)
    erows = []
    for tb in [5, 4, 3]:
        bit_alloc = C.eaq_bit_allocation(scores, tb)
        eaq = C.quantize_model_mixed(base, bit_alloc); eaq.eval()
        uni = C.quantize_model(base, tb); uni.eval()
        for name, mdl in [("EAQ", eaq), ("Uniform", uni)]:
            acc = accuracy(mdl, loader)
            sp = {meth: [] for meth in methods}
            for i in range(N_EXPL):
                vmaps = all_maps(mdl, xs[i], int(ys[i]))
                for meth in methods:
                    sp[meth].append(A.spearman(vmaps[meth], base_maps[i][meth]))
            for meth in methods:
                erows.append(dict(target_bits=tb, method_type=name, xai=meth,
                                  acc=acc, fidelity=np.mean(sp[meth]),
                                  alloc=json.dumps([int(x) for x in bit_alloc])))
        print(f"EAQ target={tb} alloc={list(bit_alloc)} t={time.time()-t0:.1f}s")
    pd.DataFrame(erows).to_csv(OUT + "eaq.csv", index=False)

    # ---- save a few qualitative examples ----
    np.savez(OUT + "qualitative.npz",
             X=Xn[:6], Y=Yte[:6], M=Mte[:6], mean=mean, std=std)
    print("ALL DONE  total", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
