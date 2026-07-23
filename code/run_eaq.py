import time, json, numpy as np, torch, pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from dataset import make_dataset, normalise, NUM_CLASSES
from model import SmallResNet
import compress as C
import attributions as A

torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"; OUT = "../results/eaq.csv"; N_EXPL = 90; IG_STEPS = 20


def all_maps(model, x, target):
    return {"Saliency": A.saliency(model, x, target, DEV),
            "IG": A.integrated_gradients(model, x, target, DEV, steps=IG_STEPS),
            "GradCAM": A.grad_cam(model, x, target, DEV),
            "Occlusion": A.occlusion(model, x, target, DEV, patch=8, stride=8)}


def accuracy(model, loader):
    model.eval(); c = t = 0
    with torch.no_grad():
        for xb, yb in loader:
            c += (model(xb).argmax(1) == yb).sum().item(); t += yb.numel()
    return c / t


def main():
    t0 = time.time()
    ck = torch.load("baseline.pt", weights_only=False)
    base = SmallResNet(NUM_CLASSES); base.load_state_dict(ck["model"]); base.eval()
    Xte, Yte, Mte = make_dataset(1000, seed=99)
    Xn, _, _ = normalise(Xte, ck["mean"], ck["std"])
    loader = DataLoader(TensorDataset(torch.tensor(Xn).permute(0, 3, 1, 2),
                        torch.tensor(Yte)), batch_size=128)
    xs = torch.tensor(Xn[:N_EXPL]).permute(0, 3, 1, 2); ys = Yte[:N_EXPL]
    methods = list(A.METHODS.keys())

    bm = np.load("../results/base_maps.npz")
    base_maps = [{m: bm[f"{i}_{m}"] for m in methods} for i in range(N_EXPL)]

    sl = DataLoader(TensorDataset(torch.tensor(Xn[:512]).permute(0, 3, 1, 2),
                    torch.tensor(Yte[:512])), batch_size=64)
    scores = C.relevance_scores(base, sl, DEV)
    print("layer relevance:", np.round(scores, 4))
    rows = []
    for tb in [5, 4, 3]:
        alloc = C.eaq_bit_allocation(scores, tb)
        eaq = C.quantize_model_mixed(base, alloc); eaq.eval()
        uni = C.quantize_model(base, tb); uni.eval()
        for name, mdl in [("EAQ", eaq), ("Uniform", uni)]:
            acc = accuracy(mdl, loader)
            sp = {m: [] for m in methods}
            for i in range(N_EXPL):
                vmaps = all_maps(mdl, xs[i], int(ys[i]))
                for m in methods:
                    sp[m].append(A.spearman(vmaps[m], base_maps[i][m]))
            for m in methods:
                rows.append(dict(target_bits=tb, method_type=name, xai=m, acc=acc,
                                 fidelity=float(np.mean(sp[m])),
                                 alloc=json.dumps([int(x) for x in alloc])))
        print(f"EAQ tb={tb} alloc={list(alloc)} t={time.time()-t0:.1f}")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    np.savez("../results/qualitative.npz", X=Xn[:6], Y=Yte[:6], M=Mte[:6],
             mean=ck["mean"], std=ck["std"])
    print("EAQ DONE")


if __name__ == "__main__":
    main()
