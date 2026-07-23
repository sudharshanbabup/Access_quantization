import time, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dataset import make_dataset, normalise, NUM_CLASSES
from model import SmallResNet, count_params

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(1)
DEV = "cpu"


def loaders(ntr=3000, nte=1000, bs=64):
    Xtr, Ytr, Mtr = make_dataset(ntr, seed=1)
    Xte, Yte, Mte = make_dataset(nte, seed=99)
    Xtr, mean, std = normalise(Xtr)
    Xte, _, _ = normalise(Xte, mean, std)
    def tens(X, Y):
        return TensorDataset(torch.tensor(X).permute(0, 3, 1, 2),
                             torch.tensor(Y))
    tr = DataLoader(tens(Xtr, Ytr), batch_size=bs, shuffle=True)
    te = DataLoader(tens(Xte, Yte), batch_size=bs, shuffle=False)
    return tr, te, (mean, std), (Xte, Yte, Mte)


def evaluate(model, loader):
    model.eval(); correct = tot = 0
    with torch.no_grad():
        for xb, yb in loader:
            p = model(xb).argmax(1)
            correct += (p == yb).sum().item(); tot += yb.numel()
    return correct / tot


def train(epochs=14, lr=1e-3):
    tr, te, stats, testraw = loaders()
    model = SmallResNet(NUM_CLASSES).to(DEV)
    print("params:", count_params(model))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad(); loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        sched.step()
        acc = evaluate(model, te)
        print(f"epoch {ep+1:2d} test_acc {acc:.4f} t {time.time()-t0:.1f}s")
    return model, te, stats, testraw


if __name__ == "__main__":
    model, te, stats, testraw = train()
    acc = evaluate(model, te)
    torch.save({"model": model.state_dict(),
                "mean": stats[0], "std": stats[1], "acc": acc},
               "baseline.pt")
    print("FINAL baseline acc", acc)
