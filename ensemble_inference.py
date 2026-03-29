"""
ensemble_inference.py
============
Visual Recognition using Deep Learning - HW1
Weighted ensemble inference with TTA.
Supports per-checkpoint image sizes.
"""

import os
import argparse
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from preprocessing import ImageFolderDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="HW1 Weighted Ensemble Inference with TTA"
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--ckpts", nargs="+", required=True,
        help="Checkpoint paths. e.g. --ckpts a.pth b.pth c.pth",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=None,
        help="Input image size per checkpoint (must match --ckpts count). "
             "Defaults to 224 for all. e.g. --sizes 224 224 256",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="Ensemble weight per checkpoint. Defaults to equal weights. "
             "Will be normalised to sum=1.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta", action="store_true",
                        help="Enable Test Time Augmentation")
    parser.add_argument("--tta_n", type=int, default=8,
                        help="Number of TTA passes (default: 8)")
    parser.add_argument("--out_dir", type=str, default="./submission")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loader — auto-detects ResNet-50 vs ResNet-101
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, num_classes: int = 100) -> nn.Module:
    state = torch.load(ckpt_path, map_location="cpu")
    total_params = sum(p.numel() for p in state.values())

    if total_params > 35_000_000:
        print(f"  → ResNet-101  ({total_params/1e6:.1f}M params)")
        model = models.resnet101(weights=None)
    else:
        print(f"  → ResNet-50   ({total_params/1e6:.1f}M params)")
        model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# TTA transforms — built per image_size
# ---------------------------------------------------------------------------
def get_tta_transforms(n: int, image_size: int) -> list:
    """
    8-strategy TTA pool (cycles if n > 8):
      0: centre crop
      1: centre crop + H-flip
      2: random resized crop (0.8-1.0)
      3: random resized crop + H-flip
      4: tighter crop (0.7-0.9)
      5: tighter crop + H-flip
      6: centre crop + colour jitter
      7: centre crop + colour jitter + H-flip
    """
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    resize_to = int(image_size * 256 / 224)

    pool = [
        # 0
        transforms.Compose([
            transforms.Resize(resize_to), transforms.CenterCrop(image_size),
            transforms.ToTensor(), norm,
        ]),
        # 1
        transforms.Compose([
            transforms.Resize(resize_to), transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), norm,
        ]),
        # 2
        transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(), norm,
        ]),
        # 3
        transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), norm,
        ]),
        # 4
        transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 0.9)),
            transforms.ToTensor(), norm,
        ]),
        # 5
        transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 0.9)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), norm,
        ]),
        # 6
        transforms.Compose([
            transforms.Resize(resize_to), transforms.CenterCrop(image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(), norm,
        ]),
        # 7
        transforms.Compose([
            transforms.Resize(resize_to), transforms.CenterCrop(image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), norm,
        ]),
    ]
    return [pool[i % len(pool)] for i in range(n)]


# ---------------------------------------------------------------------------
# TTA inference for one model
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_tta(model, data_root, batch_size, num_workers,
            device, tta_transforms) -> tuple:
    """Run all TTA passes and return averaged softmax probs + filenames."""
    probs_sum = None
    fnames_ref = None
    n = len(tta_transforms)

    for i, tfm in enumerate(tta_transforms):
        ds = ImageFolderDataset(
            root=os.path.join(data_root, "test"),
            transform=tfm,
            is_test=True,
        )
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        pass_probs, pass_fnames = [], []
        for images, fnames in tqdm(
                loader, desc=f"  TTA {i+1}/{n}", leave=False):
            logits = model(images.to(device))
            pass_probs.append(F.softmax(logits, dim=1).cpu())
            pass_fnames.extend(fnames)

        pass_probs = torch.cat(pass_probs, dim=0)
        if probs_sum is None:
            probs_sum = pass_probs
            fnames_ref = pass_fnames
        else:
            probs_sum += pass_probs

    return fnames_ref, probs_sum / n


# ---------------------------------------------------------------------------
# Save submission
# ---------------------------------------------------------------------------
def save_submission(fnames: list, preds: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "prediction.csv")
    zip_path = os.path.join(out_dir, "submission.zip")

    pd.DataFrame({
        "image_name": [f.split('.')[0] for f in fnames],
        "pred_label": preds
    }).to_csv(csv_path, index=False)
    print(f"\n[Output] {len(fnames)} predictions → {csv_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="prediction.csv")
    print(f"[Output] Zipped  → {zip_path}  ← upload this to CodaBench")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    n_ckpts = len(args.ckpts)

    # ---- Validate & normalise sizes ----
    if args.sizes is None:
        sizes = [224] * n_ckpts
    else:
        if len(args.sizes) != n_ckpts:
            raise ValueError(
                f"--sizes count ({len(args.sizes)}) must match "
                f"--ckpts count ({n_ckpts})"
            )
        sizes = args.sizes

    # ---- Validate & normalise weights ----
    if args.weights is None:
        weights = [1.0 / n_ckpts] * n_ckpts
    else:
        if len(args.weights) != n_ckpts:
            raise ValueError(
                f"--weights count ({len(args.weights)}) must match "
                f"--ckpts count ({n_ckpts})"
            )
        total = sum(args.weights)
        weights = [w / total for w in args.weights]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device : {device}")
    status = f"ON (n={args.tta_n})" if args.tta else "OFF"
    print(f"[Info] TTA    : {status}")
    print(f"[Info] Models : {n_ckpts}")
    for ck, sz, w in zip(args.ckpts, sizes, weights):
        print(f"         size={sz}  weight={w:.3f}  {ck}")

    # ---- Run each model ----
    ensemble_probs = None
    fnames_ref = None

    for ckpt_path, image_size, weight in zip(args.ckpts, sizes, weights):
        print(
            f"\n[Model] {ckpt_path}  (size={image_size}, weight={weight:.3f})")
        model = load_model(ckpt_path).to(device)

        n_passes = args.tta_n if args.tta else 1
        tta_tfms = get_tta_transforms(n_passes, image_size)
        fnames, probs = run_tta(
            model, args.data_root, args.batch_size,
            args.num_workers, device, tta_tfms,
        )

        if ensemble_probs is None:
            ensemble_probs = probs * weight
            fnames_ref = fnames
        else:
            ensemble_probs += probs * weight

        del model
        torch.cuda.empty_cache()

    preds = torch.argmax(ensemble_probs, dim=1).tolist()
    save_submission(fnames_ref, preds, args.out_dir)


if __name__ == "__main__":
    main()
