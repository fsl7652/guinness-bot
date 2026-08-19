"""
train_glass.py

Fine-tunes MobileNetV3-Small on the glass classifier dataset.
4 classes: guinness_tulip, guinness_midsip, wrong_glass, not_glass

Usage:
    python training/train_glass.py --data data/datasets/glass_classifier --out models/glass
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# ── Config ────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 32
LR_HEAD    = 1e-3   # higher lr for new classification head
LR_BACKBONE = 1e-4  # lower lr for pretrained backbone
IMG_SIZE   = 224
PATIENCE   = 7      # early stopping patience


def get_transforms(train=True):
    if train:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            T.RandomRotation(15),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(num_classes):
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    # Replace classifier head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to glass_classifier dataset')
    parser.add_argument('--out',  required=True, help='Output folder for model')
    args   = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Datasets
    train_ds = ImageFolder(data_dir / 'train', transform=get_transforms(train=True))
    val_ds   = ImageFolder(data_dir / 'val',   transform=get_transforms(train=False))
    test_ds  = ImageFolder(data_dir / 'test',  transform=get_transforms(train=False))

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # Save class mapping
    with open(out_dir / 'classes.json', 'w') as f:
        json.dump(train_ds.class_to_idx, f, indent=2)

    # Class weights to handle imbalance
    class_counts = [0] * len(train_ds.classes)
    for _, label in train_ds.samples:
        class_counts[label] += 1
    weights = [1.0 / c for c in class_counts]
    class_weights = torch.tensor(weights).to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    model     = build_model(len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Two param groups: backbone gets lower lr
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    head_params     = list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params,     'lr': LR_HEAD},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0
    patience_counter = 0
    history = []

    print(f"\nTraining for up to {EPOCHS} epochs (early stop patience={PATIENCE})...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
              f"{elapsed:.1f}s")

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / 'best.pt')
            print(f"  ✓ New best val acc: {val_acc:.3f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Test evaluation
    model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.3f}")

    with open(out_dir / 'history.json', 'w') as f:
        json.dump({'history': history, 'best_val_acc': best_val_acc,
                   'test_acc': test_acc}, f, indent=2)

    print(f"\n✓ Model saved to {out_dir}/best.pt")
    print(f"✓ Class mapping saved to {out_dir}/classes.json")


if __name__ == '__main__':
    main()