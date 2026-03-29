# NYCU Visual Recognition using Deep Learning 2026 — Homework 1

* **Student ID**: 313553049
* **Name**: 劉怡妏

## Introduction

This repository contains the implementation for HW1.

Our approach uses a pretrained ResNet-101 backbone with a two-phase fine-tuning strategy:

* **Phase 1**: Warm-up — only the classification head is trained (5 epochs)
* **Phase 2**: Full fine-tuning with Cosine Annealing LR decay (25 epochs)

Key techniques include WeightedRandomSampler for class imbalance, Mixup/CutMix augmentation, label smoothing, and weighted ensemble inference with 8-pass Test Time Augmentation (TTA).

## Environment Setup

### Requirements

* Python 3.9+
* CUDA-compatible GPU (recommended)

### Installation

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm scikit-learn pandas
```

## Usage

### 1\. Exploratory Data Analysis (EDA)

Run EDA to visualize class distribution and image size distribution before training:

```bash
python preprocessing.py --data\\\_root ./cv\\\_hw1\\\_data/data --mode eda --split train
python preprocessing.py --data\\\_root ./cv\\\_hw1\\\_data/data --mode eda --split val
```

Output plots are saved to `./eda\\\_outputs/`.

### 2\. Training

**Normal mode** (with validation monitoring and early stopping):

```bash
python train.py --data\\\_root ./cv\\\_hw1\\\_data/data
```

**Final mode** (merge train + val, fixed epochs, no early stopping):

```bash
# Model A (seed 42)
python train.py --data\_root ./cv\_hw1\_data/data --final\_mode --phase2\_epochs 25 --seed 42 --run\_name final\_A


# Model B (seed 7)
python train.py --data\_root ./cv\_hw1\_data/data --final\_mode --phase2\_epochs 25 --seed 7 --run\_name final\_B 

```

Checkpoints are saved to `./checkpoints/`. Training curves are saved to `./plots/`.

### 3\. Inference

**Single model with TTA:**

```bash
python ensemble\\\_inference.py --data\\\_root ./cv\\\_hw1\\\_data/data --ckpts checkpoints/final\\\_A.pth --sizes 384 --tta --tta\\\_n 8 --out\\\_dir ./submission/single
```

**Weighted ensemble (recommended):**

```bash
python ensemble\\\_inference.py --data\\\_root ./cv\\\_hw1\\\_data/data --ckpts checkpoints/final\\\_A.pth checkpoints/final\\\_B.pth checkpoints/final\\\_resnet101.pth --sizes 384 384 256 --weights 0.4 0.4 0.2 --tta --tta\\\_n 8 --out\\\_dir ./submission/ensemble
```

The output `submission.zip` (containing `prediction.csv`) is saved to the specified `--out\\\_dir`. Upload `submission.zip` directly to CodaBench.

## Performance Snapshot

### Leaderboard

!\[Leaderboard Screenshot](assets/leaderboard.png)

