"""
preprocessing.py
================
Visual Recognition using Deep Learning - HW1
Handles dataset EDA, augmentation, and DataLoader construction.

Usage (standalone EDA):
    python preprocessing.py --data_root ./data --mode eda

Usage (import in train.py):
    from preprocessing import get_dataloaders, get_dataloaders_final
"""

import os
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
from torch.utils.data import (Dataset, DataLoader,
                              WeightedRandomSampler, ConcatDataset)
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 384


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------
class ImageFolderDataset(Dataset):
    """Custom Dataset that reads class-organised folders.

    Expected structure:
        root/
            0/  img1.jpg  img2.jpg ...
            1/  ...
            99/ ...

    For the test set (no sub-folders), pass ``is_test=True``.
    """

    def __init__(self, root: str, transform=None, is_test: bool = False):
        self.root = Path(root)
        self.transform = transform
        self.is_test = is_test
        self.samples = []
        self.classes = []

        if is_test:
            self._load_test()
        else:
            self._load_train_val()

    def _load_train_val(self):
        class_dirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir()],
            key=lambda d: int(d.name),
        )
        self.classes = [d.name for d in class_dirs]
        for class_dir in class_dirs:
            label = int(class_dir.name)
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() in {
                        ".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((img_path, label))

    def _load_test(self):
        for img_path in sorted(self.root.glob("*")):
            if img_path.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                self.samples.append((img_path, img_path.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transform(strong: bool = False) -> transforms.Compose:
    """Return training augmentation pipeline."""
    base = [
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1),
        transforms.RandomRotation(degrees=15),
    ]

    if strong:
        base += [
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        ]

    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if strong:
        base.append(transforms.RandomErasing(p=0.3))

    return transforms.Compose(base)


def get_val_transform() -> transforms.Compose:
    """Return deterministic validation / test transform."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# WeightedRandomSampler
# ---------------------------------------------------------------------------
def get_class_weights(dataset) -> torch.Tensor:
    """Compute per-class weights (inverse frequency) for loss reweighting."""
    labels = [label for _, label in dataset.samples]
    count = Counter(labels)
    num_cls = len(count)
    weights = torch.zeros(num_cls)
    for cls, cnt in count.items():
        weights[cls] = 1.0 / cnt
    return weights / weights.sum()


def get_sampler(dataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler so every class is sampled equally."""
    labels = [label for _, label in dataset.samples]
    count = Counter(labels)
    sample_weights = [1.0 / count[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def _get_combined_sampler(datasets: list) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler for a ConcatDataset."""
    all_labels = []
    for ds in datasets:
        all_labels.extend([label for _, label in ds.samples])
    count = Counter(all_labels)
    sample_weights = [1.0 / count[label] for label in all_labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# DataLoader — standard (train / val split)
# ---------------------------------------------------------------------------
def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_sampler: bool = True,
    strong_aug: bool = False,
) -> dict:
    """
    Build train / val / test DataLoaders
    (uses separate train & val folders).
    """
    root = Path(data_root)

    train_ds = ImageFolderDataset(root / "train",
                                  transform=get_train_transform(strong_aug))
    val_ds = ImageFolderDataset(root / "val", transform=get_val_transform())
    test_ds = ImageFolderDataset(
        root / "test",
        transform=get_val_transform(),
        is_test=True)

    if use_sampler:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=get_sampler(train_ds),
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(
        f"[Dataset] Train: {len(train_ds)} | "
        f"Val: {len(val_ds)} | "
        f"Test: {len(test_ds)}")
    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ---------------------------------------------------------------------------
# DataLoader — final mode (train + val merged, no val split)
# ---------------------------------------------------------------------------
def get_dataloaders_final(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    strong_aug: bool = True,
) -> dict:
    """
    Build DataLoaders for final training using ALL labelled data (train + val).

    Val folder is added to training data with the same augmentation pipeline.
    No validation loader is returned — use 'test' loader for sanity check only.
    """
    root = Path(data_root)

    train_ds = ImageFolderDataset(root / "train",
                                  transform=get_train_transform(strong_aug))
    val_ds = ImageFolderDataset(root / "val",
                                transform=get_train_transform(strong_aug))
    test_ds = ImageFolderDataset(
        root / "test",
        transform=get_val_transform(),
        is_test=True)

    combined_ds = ConcatDataset([train_ds, val_ds])
    sampler = _get_combined_sampler([train_ds, val_ds])

    train_loader = DataLoader(
        combined_ds, batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    total = len(train_ds) + len(val_ds)
    print(f"[Dataset - Final Mode] Train+Val: {total} | Test: {len(test_ds)}")
    return {"train": train_loader, "test": test_loader}


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def analyze_class_distribution(root: str, split: str = "train") -> dict:
    split_dir = Path(root) / split
    counts = {}
    for class_dir in sorted(split_dir.iterdir(), key=lambda d: int(d.name)):
        if class_dir.is_dir():
            n = sum(
                1 for f in class_dir.glob("*")
                if f.suffix.lower() in {".jpg",
                                        ".jpeg",
                                        ".png",
                                        ".bmp",
                                        ".webp"}
            )
            counts[int(class_dir.name)] = n
    return counts


def analyze_image_sizes(root: str, split: str = "train",
                        sample_n: int = 500) -> list:
    split_dir = Path(root) / split
    images = (
        list(split_dir.rglob("*.jpg"))
        + list(split_dir.rglob("*.jpeg"))
        + list(split_dir.rglob("*.png"))
    )
    np.random.shuffle(images)
    sizes = []
    for img_path in images[:sample_n]:
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
        except Exception:
            continue
    return sizes


def plot_class_distribution(
        counts: dict, split: str = "train", save_path: str = None):
    classes = list(counts.keys())
    values = list(counts.values())
    mean_val = np.mean(values)

    colors = []
    for v in values:
        ratio = v / mean_val
        if ratio < 0.5:
            colors.append("#e74c3c")
        elif ratio < 0.8:
            colors.append("#f39c12")
        else:
            colors.append("#2ecc71")

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    ax = axes[0]
    ax.bar(classes, values, color=colors, edgecolor="none", width=0.85)
    ax.axhline(mean_val, color="#3498db", linewidth=1.5, linestyle="--",
               label=f"Mean = {mean_val:.1f}")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Number of Images")
    ax.set_title(f"Class Distribution — {split} set", fontweight="bold")
    red_p = mpatches.Patch(color="#e74c3c", label="< 50% of mean (severe)")
    amber_p = mpatches.Patch(color="#f39c12", label="50–80% of mean")
    green_p = mpatches.Patch(color="#2ecc71", label=">= 80% of mean")
    ax.legend(handles=[red_p, amber_p, green_p], fontsize=9)

    ax2 = axes[1]
    sorted_items = sorted(counts.items(), key=lambda x: x[1])
    s_classes = [str(c) for c, _ in sorted_items]
    s_values = [v for _, v in sorted_items]
    s_colors = []
    for v in s_values:
        ratio = v / mean_val
        if ratio < 0.5:
            s_colors.append("#e74c3c")
        elif ratio < 0.8:
            s_colors.append("#f39c12")
        else:
            s_colors.append("#2ecc71")
    ax2.barh(range(len(s_classes)), s_values, color=s_colors, edgecolor="none")
    ax2.set_yticks(range(len(s_classes)))
    ax2.set_yticklabels(s_classes, fontsize=6)
    ax2.axvline(mean_val, color="#3498db", linewidth=1.5, linestyle="--",
                label=f"Mean = {mean_val:.1f}")
    ax2.set_title(f"Sorted by Count — {split} set", fontweight="bold")
    ax2.set_xlabel("Number of Images")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[EDA] Saved distribution plot → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_size_distribution(sizes: list, save_path: str = None):
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(widths, heights, alpha=0.3, s=10, color="#8e44ad")
    axes[0].axvline(
        224,
        color="red",
        linestyle="--",
        linewidth=1,
        label="224px")
    axes[0].axhline(224, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Image W × H Scatter")
    axes[0].legend()

    axes[1].hist(widths, bins=40, color="#2980b9", edgecolor="white")
    axes[1].axvline(np.median(widths), color="red", linestyle="--",
                    label=f"Median={np.median(widths):.0f}")
    axes[1].set_title("Width Distribution")
    axes[1].legend()

    axes[2].hist(heights, bins=40, color="#27ae60", edgecolor="white")
    axes[2].axvline(np.median(heights), color="red", linestyle="--",
                    label=f"Median={np.median(heights):.0f}")
    axes[2].set_title("Height Distribution")
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[EDA] Saved size plot → {save_path}")
    else:
        plt.show()
    plt.close()


def print_summary(counts: dict, split: str = "train"):
    values = list(counts.values())
    mean_v = np.mean(values)
    print(f"\n{'='*50}")
    print(f"  EDA Summary — {split} set")
    print(f"{'='*50}")
    print(f"  Total images   : {sum(values)}")
    print(f"  Num classes    : {len(counts)}")
    print(
        f"  Min per class  : {min(values)}  "
        f"(class {min(counts, key=counts.get)})")
    print(
        f"  Max per class  : {max(values)}  "
        f"(class {max(counts, key=counts.get)})")
    print(f"  Mean per class : {mean_v:.1f}")
    print(f"  Std per class  : {np.std(values):.1f}")
    severe = [c for c, v in counts.items() if v / mean_v < 0.5]
    print(f"  Severe minority classes (<50% mean): {len(severe)} → {severe}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="HW1 Preprocessing & EDA")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--mode", type=str, default="eda",
                        choices=["eda", "test_loader"])
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"])
    parser.add_argument("--sample_n", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="./eda_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "eda":
        print(
            f"\n[EDA] Analysing '{args.split}' split under: {args.data_root}")
        counts = analyze_class_distribution(args.data_root, args.split)
        print_summary(counts, args.split)
        plot_class_distribution(
            counts, split=args.split,
            save_path=os.path.join(
                args.save_dir,
                f"{args.split}_class_distribution.png"),
        )
        print(f"[EDA] Sampling {args.sample_n} images for size analysis …")
        sizes = analyze_image_sizes(args.data_root, args.split, args.sample_n)
        plot_size_distribution(
            sizes,
            save_path=os.path.join(
                args.save_dir,
                f"{args.split}_size_distribution.png"),
        )
        print("[EDA] Done. Check the eda_outputs/ folder for plots.\n")

    elif args.mode == "test_loader":
        loaders = get_dataloaders(args.data_root, batch_size=32, num_workers=0)
        imgs, labels = next(iter(loaders["train"]))
        print(
            f"[Loader] Train batch — "
            f"images: {imgs.shape}, labels: {labels.shape}")
        imgs, labels = next(iter(loaders["val"]))
        print(
            f"[Loader] Val   batch — "
            f"images: {imgs.shape}, labels: {labels.shape}")
        imgs, fnames = next(iter(loaders["test"]))
        print(
            f"[Loader] Test  batch — "
            f"images: {imgs.shape}, filenames[0]: {fnames[0]}")


if __name__ == "__main__":
    main()
