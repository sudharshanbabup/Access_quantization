# Datasets Deliverable Package

To comply with GitHub's 100MB file size limits, the CIFAR-10 and ImageNette datasets are split into 50MB parts.

## Extraction Instructions

To combine and extract the datasets, run the following commands in the root of the cloned repository:

### 1. Reconstruct and Extract CIFAR-10
```bash
cat datasets/cifar10.zip.part.* > datasets/cifar10.zip
unzip -q datasets/cifar10.zip -d ../Ravi_Saidala_v3/datasets_v3/
rm datasets/cifar10.zip
```

### 2. Reconstruct and Extract ImageNette
```bash
cat datasets/imagenette2-160.zip.part.* > datasets/imagenette2-160.zip
unzip -q datasets/imagenette2-160.zip -d ../Ravi_Saidala_v3/datasets_v3/
rm datasets/imagenette2-160.zip
```

The code scripts (`run_natural.py`, `run_runtime_analysis.py`) look for datasets at `../Ravi_Saidala_v3/datasets_v3/` by default. Extracting to that location will allow the code to find them immediately.
