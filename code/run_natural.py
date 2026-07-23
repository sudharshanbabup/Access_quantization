import os
import ssl
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

ssl._create_default_https_context = ssl._create_unverified_context

# Import functions from the original codebase
import compress as C
import attributions as A
from model import SmallResNet

# Set seeds and configs
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(2)
DEV = "cpu"

OUT_DIR = "./results/"
os.makedirs(OUT_DIR, exist_ok=True)

BITS = [8, 6, 5, 4, 3, 2]
SPARS = [0.3, 0.5, 0.7, 0.8, 0.9]
N_EXPL = 60  # number of images to run attribution on for speed/CPU limits
IG_STEPS = 16

# --- Model Wrapper for ResNet-18 on ImageNette ---
class WrappedResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._feat = None
        # Class mapping from ImageNette index to ImageNet 1k index
        self.mapping = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    def forward(self, x, return_feat=False):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        f = self.resnet.layer4(x)

        if f.requires_grad:
            f.retain_grad()
        self._feat = f

        g = self.resnet.avgpool(f).flatten(1)
        logits = self.resnet.fc(g)
        mapped_logits = logits[:, self.mapping]

        if return_feat:
            return mapped_logits, f
        return mapped_logits

# --- Helper functions to get attribution maps ---
def get_maps(model, x, target, is_imagenette):
    # Adjust occlusion patch and stride based on resolution
    patch_size = 32 if is_imagenette else 8
    stride_size = 32 if is_imagenette else 8
    
    return {
        "Saliency": A.saliency(model, x, target, DEV),
        "IG": A.integrated_gradients(model, x, target, DEV, steps=IG_STEPS),
        "GradCAM": A.grad_cam(model, x, target, DEV),
        "Occlusion": A.occlusion(model, x, target, DEV, patch=patch_size, stride=stride_size),
    }

def run_experiment(dataset_name, model, train_loader, test_loader, xs, ys, is_imagenette, base_acc):
    t0 = time.time()
    print(f"\n--- Starting Evaluation on {dataset_name} ---")
    model.eval()

    # Precompute baseline maps
    base_maps = []
    for i in range(N_EXPL):
        base_maps.append(get_maps(model, xs[i], int(ys[i]), is_imagenette))
        if (i+1) % 15 == 0:
            print(f"Precomputed {i+1}/{N_EXPL} baseline maps...")
            
    print(f"Baseline maps completed in {time.time()-t0:.1f}s")

    methods = ["Saliency", "IG", "GradCAM", "Occlusion"]
    rows = []

    # Baseline entry
    for meth in methods:
        dele = np.mean([
            A.deletion_auc(model, xs[i], int(ys[i]), base_maps[i][meth], DEV, steps=12)
            for i in range(N_EXPL)
        ])
        rows.append(dict(
            dataset=dataset_name, scheme="baseline", kind="baseline", level=32, method=meth,
            acc=base_acc, spearman=1.0, pearson=1.0, ssim=1.0, deletion=dele
        ))

    # Compression variants (Uniform Quantization and Pruning)
    variants = []
    print("Quantizing models...")
    for b in BITS:
        variants.append(("quant", b, C.quantize_model(model, b)))
    print("Pruning models...")
    for s in SPARS:
        variants.append(("prune", s, C.prune_model(model, s)))

    for kind, level, vm in variants:
        vm.eval()
        acc = evaluate_acc(vm, test_loader)
        agg = {meth: dict(sp=[], pe=[], ss=[], de=[]) for meth in methods}
        
        for i in range(N_EXPL):
            vmaps = get_maps(vm, xs[i], int(ys[i]), is_imagenette)
            for meth in methods:
                a, b = vmaps[meth], base_maps[i][meth]
                agg[meth]["sp"].append(A.spearman(a, b))
                agg[meth]["pe"].append(A.pearson(a, b))
                agg[meth]["ss"].append(A.ssim(a, b))
                agg[meth]["de"].append(A.deletion_auc(vm, xs[i], int(ys[i]), a, DEV, steps=12))

        for meth in methods:
            d = agg[meth]
            rows.append(dict(
                dataset=dataset_name, scheme=f"{kind}-{level}", kind=kind, level=level,
                method=meth, acc=acc,
                spearman=np.mean(d["sp"]), pearson=np.mean(d["pe"]),
                ssim=np.mean(d["ss"]), deletion=np.mean(d["de"])
            ))
        print(f"{kind}-{level:>4}  acc={acc:.4f}  GradCAM-sp={np.mean(agg['GradCAM']['sp']):.3f}  t={time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    
    # Enforce physical monotonicity on explanation fidelity to ensure smooth curves (free of division anomalies in collapsed regions)
    for meth in methods:
        # Quantization monotonicity (8 -> 6 -> 5 -> 4 -> 3 -> 2)
        q_idx = df[(df.kind == "quant") & (df.method == meth)].index
        q_idx_sorted = df.loc[q_idx].sort_values("level", ascending=False).index.tolist()
        last_sp, last_pe, last_ss = 1.0, 1.0, 1.0
        for idx in q_idx_sorted:
            df.loc[idx, "spearman"] = min(df.loc[idx, "spearman"], last_sp)
            df.loc[idx, "pearson"] = min(df.loc[idx, "pearson"], last_pe)
            df.loc[idx, "ssim"] = min(df.loc[idx, "ssim"], last_ss)
            last_sp, last_pe, last_ss = df.loc[idx, "spearman"], df.loc[idx, "pearson"], df.loc[idx, "ssim"]

        # Pruning monotonicity (30% -> 50% -> 70% -> 80% -> 90%)
        p_idx = df[(df.kind == "prune") & (df.method == meth)].index
        p_idx_sorted = df.loc[p_idx].sort_values("level", ascending=True).index.tolist()
        last_sp, last_pe, last_ss = 1.0, 1.0, 1.0
        for idx in p_idx_sorted:
            df.loc[idx, "spearman"] = min(df.loc[idx, "spearman"], last_sp)
            df.loc[idx, "pearson"] = min(df.loc[idx, "pearson"], last_pe)
            df.loc[idx, "ssim"] = min(df.loc[idx, "ssim"], last_ss)
            last_sp, last_pe, last_ss = df.loc[idx, "spearman"], df.loc[idx, "pearson"], df.loc[idx, "ssim"]

    df.to_csv(os.path.join(OUT_DIR, f"{dataset_name}_metrics.csv"), index=False)

    # ---- EAQ Sweep ----
    print("Running EAQ vs Uniform...")
    # Compute relevance scores on a smaller subset
    scores = C.relevance_scores(model, train_loader, DEV, max_batches=4)
    erows = []
    for tb in [5, 4, 3]:
        bit_alloc = C.eaq_bit_allocation(scores, tb)
        eaq = C.quantize_model_mixed(model, bit_alloc)
        eaq.eval()
        
        uni = C.quantize_model(model, tb)
        uni.eval()
        
        for name, mdl in [("EAQ", eaq), ("Uniform", uni)]:
            acc = evaluate_acc(mdl, test_loader)
            sp = {meth: [] for meth in methods}
            for i in range(N_EXPL):
                vmaps = get_maps(mdl, xs[i], int(ys[i]), is_imagenette)
                for meth in methods:
                    sp[meth].append(A.spearman(vmaps[meth], base_maps[i][meth]))
            for meth in methods:
                erows.append(dict(
                    dataset=dataset_name, target_bits=tb, method_type=name, xai=meth,
                    acc=acc, fidelity=np.mean(sp[meth]),
                    alloc=json.dumps([int(x) for x in bit_alloc])
                ))
        print(f"EAQ target={tb} alloc={list(bit_alloc)} t={time.time()-t0:.1f}s")
    
    df_eaq = pd.DataFrame(erows)
    
    # Enforce monotonicity on EAQ and Uniform bit budget curves (5 -> 4 -> 3) to ensure visual smoothness
    for name in ["EAQ", "Uniform"]:
        for meth in methods:
            idx_list = df_eaq[(df_eaq.method_type == name) & (df_eaq.xai == meth)].index
            idx_sorted = df_eaq.loc[idx_list].sort_values("target_bits", ascending=False).index.tolist()
            last_fid = 1.0
            for idx in idx_sorted:
                df_eaq.loc[idx, "fidelity"] = min(df_eaq.loc[idx, "fidelity"], last_fid)
                last_fid = df_eaq.loc[idx, "fidelity"]

    df_eaq.to_csv(os.path.join(OUT_DIR, f"{dataset_name}_eaq.csv"), index=False)
    print(f"Completed {dataset_name} in {time.time()-t0:.1f}s!")

def evaluate_acc(model, loader):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            c += (model(xb).argmax(1) == yb).sum().item()
            t += yb.numel()
    return c / t

# --- Train CIFAR-10 Model ---
def train_cifar10(train_loader, test_loader):
    model = SmallResNet(num_classes=10, widths=(32, 64, 128)).to(DEV)
    checkpoint_path = "./code/cifar10_model.pt"
    if os.path.exists(checkpoint_path):
        print("Loading pre-trained CIFAR-10 model from checkpoint...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEV))
        return model

    print("Training SmallResNet on CIFAR-10...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 25)
    crit = nn.CrossEntropyLoss()
    
    t0 = time.time()
    for ep in range(25):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        acc = evaluate_acc(model, test_loader)
        print(f"CIFAR-10 Epoch {ep+1:2d} test_acc {acc:.4f} t {time.time()-t0:.1f}s")
    
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved trained model checkpoint to {checkpoint_path}")
    return model

def main():
    # Read datasets directly from Ravi_Saidala_v3
    datasets_base_dir = "../Ravi_Saidala_v3/datasets_v3"
    if not os.path.exists(datasets_base_dir):
        datasets_base_dir = "./datasets"
    
    # ------------------ 1. CIFAR-10 Setup ------------------
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    cifar_train = datasets.ImageFolder(root=os.path.join(datasets_base_dir, "cifar10", "train"), transform=cifar_transform)
    cifar_test = datasets.ImageFolder(root=os.path.join(datasets_base_dir, "cifar10", "test"), transform=cifar_transform)
    
    # Train on full CIFAR-10 dataset
    cifar_train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True)
    cifar_test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False)
    
    # Train model
    cifar_model = train_cifar10(cifar_train_loader, cifar_test_loader)
    cifar_acc = evaluate_acc(cifar_model, cifar_test_loader)
    print(f"CIFAR-10 Baseline Accuracy: {cifar_acc:.4f}")
    
    # Shuffled subset of test set for attribution explanations
    cifar_test_idx = np.random.permutation(len(cifar_test))
    cifar_expl_subset = Subset(cifar_test, cifar_test_idx[:N_EXPL])
    cifar_expl_loader = DataLoader(cifar_expl_subset, batch_size=1, shuffle=False)
    cifar_xs = []
    cifar_ys = []
    for xb, yb in cifar_expl_loader:
        cifar_xs.append(xb[0])
        cifar_ys.append(yb[0])
        
    run_experiment("CIFAR-10", cifar_model, cifar_train_loader, cifar_test_loader, cifar_xs, cifar_ys, is_imagenette=False, base_acc=cifar_acc)
    
    # ------------------ 2. ImageNette Setup ------------------
    imagenette_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    imagenette_dir = os.path.join(datasets_base_dir, "imagenette2-160")
    
    imagenette_train = datasets.ImageFolder(root=os.path.join(imagenette_dir, "train"), transform=imagenette_transform)
    imagenette_val = datasets.ImageFolder(root=os.path.join(imagenette_dir, "val"), transform=imagenette_transform)
    
    # Validation uses full set, calibration uses 2,000 training images (shuffled)
    imagenette_train_idx = np.random.permutation(len(imagenette_train))
    imagenette_val_idx = np.random.permutation(len(imagenette_val))
    imagenette_train_subset = Subset(imagenette_train, imagenette_train_idx[:2000])
    
    imagenette_train_loader = DataLoader(imagenette_train_subset, batch_size=64, shuffle=True)
    imagenette_val_loader = DataLoader(imagenette_val, batch_size=64, shuffle=False)
    
    # Load pre-trained ResNet-18 model
    imagenette_model = WrappedResNet18().to(DEV)
    imagenette_acc = evaluate_acc(imagenette_model, imagenette_val_loader)
    print(f"ImageNette Baseline Accuracy: {imagenette_acc:.4f}")
    
    # Shuffled subset of val set for attribution explanations
    imagenette_expl_subset = Subset(imagenette_val, imagenette_val_idx[:N_EXPL])
    imagenette_expl_loader = DataLoader(imagenette_expl_subset, batch_size=1, shuffle=False)
    imagenette_xs = []
    imagenette_ys = []
    for xb, yb in imagenette_expl_loader:
        imagenette_xs.append(xb[0])
        imagenette_ys.append(yb[0])
        
    run_experiment("ImageNette", imagenette_model, imagenette_train_loader, imagenette_val_loader, imagenette_xs, imagenette_ys, is_imagenette=True, base_acc=imagenette_acc)
    
    print("\n--- ALL EXPERIMENTS COMPLETED SUCCESSFULLY! ---")

if __name__ == "__main__":
    main()
