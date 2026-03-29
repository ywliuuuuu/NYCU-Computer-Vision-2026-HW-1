"""
train.py
========
Visual Recognition using Deep Learning - HW1
ResNet-101 two-phase training with Mixup / CutMix augmentation.

Usage (normal, with val monitoring):
    python train.py --data_root ./data

Usage (final model, all data, no early stopping):
    python train.py --data_root ./data --final_mode --phase2_epochs 20
"""

import os
import argparse
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for all environments
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet101_Weights
from torch.cuda.amp import autocast, GradScaler

from preprocessing import get_dataloaders, get_dataloaders_final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="HW1 Training – ResNet-101 Two-Phase + Mixup/CutMix"
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--phase1_epochs", type=int, default=5)
    parser.add_argument("--phase2_epochs", type=int, default=50,
                        help="Set to 20 when using --final_mode")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (ignored in final_mode)")
    parser.add_argument("--phase2_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--mixup_prob", type=float, default=0.3)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--final_mode", action="store_true",
        help="Merge train+val, disable early stopping, save last checkpoint. "
             "Use with --phase2_epochs 20",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Checkpoint filename stem (e.g. 'final_A' -> final_A.pth). "
                             "Defaults to 'final_resnet101' or 'best_resnet101'.")
    parser.add_argument("--plot_dir", type=str, default="./plots",
                        help="Directory to save training curve and confusion matrix.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int = 100) -> nn.Module:
    """ResNet-101 with Dropout + Linear classifier head."""
    print("[Model] Loading Pretrained ResNet-101 (IMAGENET1K_V2)...")
    model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Mixup / CutMix 
# ---------------------------------------------------------------------------
def mixup_data(inputs, labels, alpha):
    lam        = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    rand_index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed      = lam * inputs + (1 - lam) * inputs[rand_index]
    return mixed, labels, labels[rand_index], lam


def cutmix_data(inputs, labels, alpha):
    lam         = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    b, _, h, w  = inputs.size()
    rand_index  = torch.randperm(b, device=inputs.device)
    cut_h, cut_w = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
    cx, cy      = np.random.randint(w), np.random.randint(h)
    x1, x2     = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    y1, y2     = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    mixed       = inputs.clone()
    mixed[:, :, y1:y2, x1:x2] = inputs[rand_index, :, y1:y2, x1:x2]
    lam         = 1 - (x2 - x1) * (y2 - y1) / (w * h)
    return mixed, labels, labels[rand_index], lam


def mixup_criterion(criterion, outputs, la, lb, lam):
    return lam * criterion(outputs, la) + (1 - lam) * criterion(outputs, lb)


# ---------------------------------------------------------------------------
# Visualization 
# ---------------------------------------------------------------------------
def plot_training_curve(history: dict, save_path: str):
    """Plot train/val loss and accuracy curves and save to file.

    Args:
        history: dict with keys 'train_loss', 'train_acc',
                 and optionally 'val_loss', 'val_acc'.
        save_path: path to save the PNG.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    has_val = "val_loss" in history and len(history["val_loss"]) > 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Loss ----
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2980b9")
    if has_val:
        axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#e74c3c")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- Accuracy ----
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#2980b9")
    if has_val:
        axes[1].plot(epochs, history["val_acc"], "s--", label="Val Acc", color="#e74c3c")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training / Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curve saved → {save_path}")


def plot_confusion_matrix(model, dataloader, device, num_classes: int,
                          save_path: str):
    """Run inference on dataloader and save a confusion matrix heatmap.

    Args:
        model:       trained model (will be set to eval mode).
        dataloader:  DataLoader yielding (images, labels).
        device:      torch device.
        num_classes: number of classes (100).
        save_path:   path to save the PNG.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Confusion Matrix", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (100 Classes)", fontsize=14, fontweight="bold")

    # Show tick labels every 10 classes to avoid clutter
    tick_step = 10
    ticks = list(range(0, num_classes, tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_yticklabels(ticks, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------
def train_one_epoch(
    model, dataloader, criterion, optimizer, device, scaler,
    use_mix=False, mixup_alpha=0.4, cutmix_alpha=1.0, mix_prob=0.3,
):
    model.train()
    running_loss, corrects, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        mixed = False
        if use_mix and np.random.rand() < mix_prob:
            if np.random.rand() < 0.5 and mixup_alpha > 0:
                inputs, la, lb, lam = mixup_data(inputs, labels, mixup_alpha)
            else:
                inputs, la, lb, lam = cutmix_data(inputs, labels, cutmix_alpha)
            mixed = True

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss    = (
                mixup_criterion(criterion, outputs, la, lb, lam)
                if mixed else criterion(outputs, labels)
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        if mixed:
            corrects += (
                lam * torch.sum(preds == la).item()
                + (1 - lam) * torch.sum(preds == lb).item()
            )
        else:
            corrects += torch.sum(preds == labels).item()
        total += inputs.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return running_loss / total, corrects / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, corrects, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)
    return running_loss / total, corrects / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # ---- Reproducibility ----
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"[Info] Seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    if args.final_mode:
        print("[Info] FINAL MODE — train+val merged, no early stopping")

    # ---- DataLoaders ----
    if args.final_mode:
        loaders     = get_dataloaders_final(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            strong_aug=True,
        )
        train_loader = loaders["train"]
        val_loader   = None
        default_name = "final_resnet101"
        save_path    = os.path.join(args.save_dir,
                                    f"{args.run_name or default_name}.pth")
    else:
        loaders      = get_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_sampler=True,
            strong_aug=True,
        )
        train_loader = loaders["train"]
        val_loader   = loaders["val"]
        default_name = "best_resnet101"
        save_path    = os.path.join(args.save_dir,
                                    f"{args.run_name or default_name}.pth")

    # ---- Model & Loss ----
    model     = build_model(num_classes=100).to(device)
    scaler = GradScaler()

    # ======== ✨ 在這裡加入計算參數量的代碼 ✨ ========
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total Parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
    
    if total_params >= 100_000_000:
        print("[Warning] Model size exceeds 100M limits!")
    else:
        print("[Info] Model size is within the 100M limit. Safe to proceed!")
    # ==================================================

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc     = 0.0
    patience_counter = 0
    val_acc_window: deque = deque(maxlen=3)

    # History for training curve (Phase 2 only — Phase 1 is warm-up)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # =========================================================
    # Phase 1: Warm-up — FC head only, backbone frozen
    # =========================================================
    print(f"\n{'='*44}")
    print(f" Phase 1: Warm-up FC Head ({args.phase1_epochs} Epochs)")
    print(f"{'='*44}")

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer1 = optim.Adam(model.fc.parameters(), lr=1e-3)

    for epoch in range(args.phase1_epochs):
        print(f"Epoch {epoch + 1}/{args.phase1_epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer1, device, scaler, use_mix=False
        )
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            )
        else:
            print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")

    # =========================================================
    # Phase 2: Full Fine-tuning
    # =========================================================
    print(f"\n{'='*44}")
    print(f" Phase 2: Full Fine-tuning ({args.phase2_epochs} Epochs)")
    print(f" LR={args.phase2_lr}  WD={args.weight_decay}  "
          f"Mixup α={args.mixup_alpha}  CutMix α={args.cutmix_alpha}  "
          f"p={args.mixup_prob}")
    if args.final_mode:
        print(" [Final Mode] No early stopping — will save last epoch")
    print(f"{'='*44}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = optim.SGD(
        model.parameters(),
        lr=args.phase2_lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=args.phase2_epochs, eta_min=1e-6
    )

    for epoch in range(args.phase2_epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{args.phase2_epochs}  (LR: {current_lr:.6f})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer2, device, scaler,
            use_mix=True,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mixup_prob,
        )
        scheduler.step()

        if val_loader:
            # ---- Normal mode: validate + early stopping ----
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            )

            # Record history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                print(f" => Val acc improved ({best_val_acc:.4f} → {val_acc:.4f}). Saving...")
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f" => EarlyStopping counter: {patience_counter} / {args.patience}")

            val_acc_window.append(val_acc)
            if len(val_acc_window) == 3:
                print(f"    Rolling-avg val acc (last 3): {sum(val_acc_window)/3:.4f}")

            if patience_counter >= args.patience:
                print("\n[Early Stopping] Triggered!")
                break

        else:
            # ---- Final mode: no val, save every epoch (overwrite) ----
            print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
            torch.save(model.state_dict(), save_path)
            print(f" => Checkpoint saved (epoch {epoch + 1})")

            # Record history (train only)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

    # =========================================================
    # Post-training: save plots
    # =========================================================
    run_tag = args.run_name or ("final_resnet101" if args.final_mode else "best_resnet101")

    # Training curve
    plot_training_curve(
        history,
        save_path=os.path.join(args.plot_dir, f"{run_tag}_training_curve.png"),
    )

    # Confusion matrix — use val_loader if available, otherwise skip
    if val_loader:
        plot_confusion_matrix(
            model, val_loader, device,
            num_classes=100,
            save_path=os.path.join(args.plot_dir, f"{run_tag}_confusion_matrix.png"),
        )
    else:
        print("[Plot] Final mode — no val_loader available for confusion matrix. "
              "Re-run without --final_mode to generate one, or use inference.py on val set.")

    if args.final_mode:
        print(f"\nFinal Training Complete.")
        print(f"Final model saved → {save_path}")
    else:
        print(f"\nTraining Complete.  Best Val Acc: {best_val_acc:.4f}")
        print(f"Best model saved → {save_path}")


if __name__ == "__main__":
    main()