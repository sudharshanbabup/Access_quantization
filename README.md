# Explanation-Aware Quantization (EAQ)

This repository contains the pure, lightweight Python codebase to reproduce the empirical results, evaluation metrics, and figures for the paper: **"Explanation Fidelity Under Neural Network Compression: Theory, Bounds, and Explanation-Aware Quantization"**.

---

## Project Structure

The project has been cleaned to contain only the necessary code execution scripts and outputs:

```text
├── code/
│   ├── model.py                     # Scaled ResNet backbone architectures
│   ├── compress.py                  # Quantization and pruning operators
│   ├── attributions.py              # Attribution explainers (Saliency, IG, Grad-CAM, Occlusion) and AUC metrics
│   ├── train.py                     # Standard PyTorch model training loops
│   ├── run_natural.py               # Main evaluation sweep for CIFAR-10 and ImageNette
│   ├── run_runtime_analysis.py      # Profiles latency, FLOPs, parameter size, and bootstrap confidence intervals
│   ├── download_datasets.py         # Helper script to download and extract standard datasets
│   ├── make_figs.py                 # Plots figures for the SynthShapes-32 synthetic benchmark
│   ├── make_figs_natural.py         # Plots figures for CIFAR-10 and ImageNette natural image benchmarks
│   ├── cifar10_model.pt             # Pre-trained model weights (81.42% CIFAR-10 test accuracy)
│   └── baseline.pt                  # Pre-trained baseline weights
├── results/
│   ├── CIFAR-10_metrics.csv         # Raw metrics for CIFAR-10 evaluations
│   ├── ImageNette_metrics.csv       # Raw metrics for ImageNette evaluations
│   ├── runtime_metrics.csv          # CPU latency, size (KB), and energy consumption values
│   ├── bootstrap_confidence_intervals.csv # 95% confidence intervals on explanation gains
│   ├── ablation_study.csv           # mixed-precision bit-allocation ablation data
│   └── insertion_deletion_auc.csv   # Insertion/Deletion AUC metrics
└── .gitignore
```

---

## Getting Started

### 1. Requirements
Install the required packages using your package manager (e.g. `pip` or `conda`):
```bash
pip install torch torchvision numpy pandas matplotlib
```

### 2. Set Up Datasets
Run the included download script to fetch and set up the CIFAR-10 and ImageNette datasets in the default workspace location (`../Ravi_Saidala_v3/datasets_v3`):
```bash
python3 code/download_datasets.py
```
*(Note: The synthetic `SynthShapes-32` dataset is generated procedurally and dynamically at runtime, so it requires no download.)*

---

## Running Evaluations & Reproducing Figures

### Step 1: Run Metrics Sweeps
To run the full evaluation sweep and measure model sizes, latencies, energy metrics, bootstrap confidence intervals, and ablation results:
```bash
python3 code/run_runtime_analysis.py
```
This script will instantly load the pre-trained weights from `code/cifar10_model.pt` and output all compiled data directly into the `results/` folder as CSV files.

### Step 2: Plot Paper Figures
To plot all the charts and figures presented in the paper from the compiled CSV files, run:
```bash
# Plots synthetic dataset charts
python3 code/make_figs.py

# Plots CIFAR-10 and ImageNette charts
python3 code/make_figs_natural.py
```
The output charts (e.g. accuracy curves, explainability degradation profiles, and knees) will be saved cleanly as PDF files under the newly generated **`results/figs/`** directory.
