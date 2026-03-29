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
python preprocessing.py --data_root ./cv_hw1_data/data --mode eda --split train
python preprocessing.py --data_root ./cv_hw1_data/data --mode eda --split val
```

Output plots are saved to `./eda_outputs/`.

### 2\. Training

**Normal mode** (with validation monitoring and early stopping):

```bash
python train.py --data_root ./cv_hw1_data/data
```

**Final mode** (merge train + val, fixed epochs, no early stopping):

```bash
# Model A (seed 42)
python train.py --data_root ./cv_hw1_data/data --final_mode --phase2_epochs 25 --seed 42 --run_name final_A


# Model B (seed 7)
python train.py --data_root ./cv_hw1_data/data --final_mode --phase2_epochs 25 --seed 7 --run_name final_B 

```

Checkpoints are saved to `./checkpoints/`. Training curves are saved to `./plots/`.

### 3\. Inference

**Single model with TTA:**

```bash
python ensemble_inference.py --data_root ./cv_hw1_data/data --ckpts checkpoints/final_A.pth --sizes 384 --tta --tta_n 8 --out_dir ./submission/single
```

**Weighted ensemble (recommended):**

```bash
python ensemble_inference.py --data_root ./cv_hw1_data/data --ckpts checkpoints/final_A.pth checkpoints/final_B.pth checkpoints/final_resnet101.pth --sizes 384 384 256 --weights 0.4 0.4 0.2 --tta --tta_n 8 --out_dir ./submission/ensemble
```

The output `submission.zip` (containing `prediction.csv`) is saved to the specified `--out_dir`. Upload `submission.zip` directly to CodaBench.

## Performance Snapshot

### Leaderboard

![Leaderboard Screenshot](assets/leaderboard.png)

