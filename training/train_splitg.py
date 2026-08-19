"""
train_splitg.py

Trains a MobileNetV3-Small ordinal classifier for split-the-G scoring.
5 classes: perfect / close / partial / near_miss / missed

Usage:
    python training/train_splitg.py --data data/datasets/splitg --out models/splitg
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGE_SIZE  = 224
BATCH_SIZE  = 16
EPOCHS      = 40   # more epochs — harder task, less data
LR_BACKBONE = 5e-5
LR_HEAD     = 5e-4
PATIENCE    = 10

# Ordinal class order — used for adjacent-class accuracy metric
ORDINAL_ORDER = ['perfect', 'close', 'partial', 'near_miss', 'missed']


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    transforms.RandomRotation(15),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes, device):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(device)


def adjacent_accuracy(preds, labels, class_names):
    """
    Ordinal metric: correct if prediction is within 1 class of true label.
    More meaningful than strict accuracy for ordinal tasks.
    """
    order = {c: i for i, c in enumerate(ORDINAL_ORDER)}
    correct = 0
    for pred, label in zip(preds, labels):
        pred_name  = class_names[pred]
        label_name = class_names[label]
        if pred_name in order and label_name in order:
            if abs(order[pred_name] - order[label_name]) <= 1:
                correct += 1
    return correct / len(preds) if preds else 0


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device, class_names):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs     = model(images)
            loss        = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    adj_acc = adjacent_accuracy(all_preds, all_labels, class_names)
    return total_loss / total, correct / total, adj_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='splitg dataset root')
    parser.add_argument('--out',  required=True, help='output folder for model')
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = datasets.ImageFolder(data_dir / 'train', transform=train_transform)
    val_ds   = datasets.ImageFolder(data_dir / 'val',   transform=val_transform)
    test_ds  = datasets.ImageFolder(data_dir / 'test',  transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    if len(train_ds) < 20:
        print("[warn] Very few training examples — consider merging partial into near_miss")

    # Heavy class weighting — partial has very few examples
    class_counts  = torch.tensor([len(list((data_dir / 'train' / c).glob('*')))
                                   for c in class_names], dtype=torch.float)
    class_weights = (1.0 / class_counts).to(device)
    class_weights = class_weights / class_weights.sum() * num_classes

    model     = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n],
         'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0
    patience_ctr = 0
    history      = []
    model_path   = out_dir / 'splitg_best.pt'

    print(f"\nTraining for up to {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_adj = eval_epoch(model, val_loader, criterion, device, class_names)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train {train_acc:.3f}  "
              f"val {val_acc:.3f} (adj {val_adj:.3f})  "
              f"({time.time()-t0:.1f}s)")

        history.append({
            'epoch': epoch,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_adj_acc': val_adj,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ saved (val_acc={val_acc:.3f} adj={val_adj:.3f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path))
    _, test_acc, test_adj = eval_epoch(model, test_loader, criterion, device, class_names)
    print(f"\nTest accuracy: {test_acc:.3f}  Adjacent accuracy: {test_adj:.3f}")

    meta = {
        'classes':       class_names,
        'num_classes':   num_classes,
        'ordinal_order': ORDINAL_ORDER,
        'best_val_acc':  best_val_acc,
        'test_acc':      test_acc,
        'test_adj_acc':  test_adj,
        'image_size':    IMAGE_SIZE,
        'history':       history,
    }
    with open(out_dir / 'splitg_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Model saved to {model_path}")


if __name__ == '__main__':
    main()