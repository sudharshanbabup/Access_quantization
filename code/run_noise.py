import time, numpy as np, torch, pandas as pd
from dataset import make_dataset, normalise, NUM_CLASSES
from model import SmallResNet
import compress as C
import attributions as A

torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"; OUT = "../results/noise_stability.csv"
BITS = [8, 6, 4, 3]; SIGMAS = [0.05, 0.10, 0.20]; N = 50; IG_STEPS = 20


def all_maps(model, x, target):
    return {"Saliency": A.saliency(model, x, target, DEV),
            "IG": A.integrated_gradients(model, x, target, DEV, steps=IG_STEPS),
            "GradCAM": A.grad_cam(model, x, target, DEV),
            "Occlusion": A.occlusion(model, x, target, DEV, patch=8, stride=8)}


def main():
    t0 = time.time()
    ck = torch.load("baseline.pt", weights_only=False)
    base = SmallResNet(NUM_CLASSES); base.load_state_dict(ck["model"]); base.eval()
    Xte, Yte, Mte = make_dataset(1000, seed=99)
    Xn, _, _ = normalise(Xte, ck["mean"], ck["std"])
    xs = torch.tensor(Xn[:N]).permute(0, 3, 1, 2); ys = Yte[:N]
    methods = list(A.METHODS.keys())
    rng = np.random.default_rng(3)
    rows = []
    variants = [("baseline", 32, base)] + [("quant", b, C.quantize_model(base, b)) for b in BITS]
    for kind, level, vm in variants:
        vm.eval()
        for sg in SIGMAS:
            stab = {m: [] for m in methods}
            for i in range(N):
                clean = all_maps(vm, xs[i], int(ys[i]))
                xn = xs[i] + torch.tensor(rng.normal(0, sg, size=xs[i].shape),
                                          dtype=torch.float32)
                noisy = all_maps(vm, xn, int(ys[i]))
                for m in methods:
                    stab[m].append(A.spearman(clean[m], noisy[m]))
            for m in methods:
                rows.append(dict(kind=kind, level=level, sigma=sg, method=m,
                                 stability=float(np.mean(stab[m]))))
        print(f"noise {kind}-{level} t={time.time()-t0:.1f}")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("noise DONE")


if __name__ == "__main__":
    main()
