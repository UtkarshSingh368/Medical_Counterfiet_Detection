"""
STEP 3: Train 4 independent ViT binary classifiers.

Models: front_vit, back_vit, text_vit, logo_vit
Each learns REAL (0) vs FAKE (1) classification.

Label mapping (enforced): real=0, fake=1
This is consistent with extract_features.py and infer.py.

Usage:
    python src/train_vit.py
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights

# ---------------------------------
# CONFIG
# ---------------------------------

ROOT = Path(__file__).resolve().parent.parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
FREEZE_BACKBONE = True

FEATURES = ["front", "back", "text", "logo"]

# Enforced label mapping (alphabetical would give fake=0, real=1)
DESIRED_CLASS_TO_IDX = {"real": 0, "fake": 1}

# ---------------------------------
# VIT FEATURE EXTRACTOR
# ---------------------------------

class ViTFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(
            weights=ViT_B_16_Weights.DEFAULT
        )
        self.embedding_dim = (
            self.vit.heads.head.in_features
        )
        self.vit.heads = nn.Identity()

    def forward(self, x):
        return self.vit(x)

# ---------------------------------
# TRANSFORMS
# ---------------------------------

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ---------------------------------
# FIX IMAGEFOLDER LABEL MAPPING
# ---------------------------------

def fix_label_mapping(dataset):
    """
    ImageFolder assigns labels alphabetically: fake=0, real=1.
    We need: real=0, fake=1. This function remaps targets.
    """
    old_to_new = {}
    for cls_name, old_idx in dataset.class_to_idx.items():
        old_to_new[old_idx] = DESIRED_CLASS_TO_IDX[cls_name]

    dataset.targets = [old_to_new[t] for t in dataset.targets]
    dataset.samples = [
        (path, old_to_new[label])
        for path, label in dataset.samples
    ]
    dataset.class_to_idx = DESIRED_CLASS_TO_IDX.copy()
    return dataset

# ---------------------------------
# TRAIN ONE FEATURE
# ---------------------------------

def train_feature(feature_name):

    print(f"\n{'='*40}")
    print(f"Training: {feature_name}")
    print(f"{'='*40}")

    train_dir = str(
        ROOT / "data" / "crops" / "train" / feature_name
    )
    val_dir = str(
        ROOT / "data" / "crops" / "val" / feature_name
    )

    # Check directories exist and have images
    for d, name in [(train_dir, "train"), (val_dir, "val")]:
        if not os.path.exists(d):
            print(f"  WARNING: {name} dir missing: {d}")
            return

    try:
        train_dataset = datasets.ImageFolder(
            train_dir, transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            val_dir, transform=val_transform
        )
    except FileNotFoundError:
        print(f"  WARNING: No images found for {feature_name}")
        return

    if len(train_dataset) == 0:
        print(f"  WARNING: No training images for {feature_name}")
        return

    # Fix label mapping: real=0, fake=1
    train_dataset = fix_label_mapping(train_dataset)
    val_dataset = fix_label_mapping(val_dataset)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Class mapping: {train_dataset.class_to_idx}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2,
    )

    # -------------------------
    # Model Setup
    # -------------------------

    backbone = ViTFeatureExtractor()

    # Freeze backbone to prevent overfitting on small datasets
    if FREEZE_BACKBONE:
        for param in backbone.vit.parameters():
            param.requires_grad = False
        # Unfreeze last encoder block + layernorm
        for param in backbone.vit.encoder.layers[-1].parameters():
            param.requires_grad = True
        for param in backbone.vit.encoder.ln.parameters():
            param.requires_grad = True

    classifier = nn.Linear(backbone.embedding_dim, 2)

    backbone = backbone.to(DEVICE)
    classifier = classifier.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Only optimize trainable parameters
    trainable_params = [
        p for p in backbone.parameters() if p.requires_grad
    ] + list(classifier.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=LR)

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    # -------------------------
    # EPOCH LOOP
    # -------------------------

    for epoch in range(EPOCHS):

        backbone.train()
        classifier.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            features = backbone(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # -------------------------
        # Validation
        # -------------------------

        backbone.eval()
        classifier.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                features = backbone(images)
                outputs = classifier(features)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = (100.0 * correct / total) if total > 0 else 0.0

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(acc)

        print(
            f"  Epoch {epoch+1}/{EPOCHS}"
            f" | Loss={avg_loss:.4f}"
            f" | Val Acc={acc:.2f}%"
        )

        # -------------------------
        # SAVE BEST
        # -------------------------

        if acc > best_acc:
            best_acc = acc

            save_dir = ROOT / "models" / "vit"
            os.makedirs(save_dir, exist_ok=True)

            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                str(save_dir / f"{feature_name}_vit.pth"),
            )

    # -------------------------
    # SAVE HISTORY
    # -------------------------

    history_path = str(
        ROOT / "models" / "vit" / f"{feature_name}_history.json"
    )
    with open(history_path, "w") as f:
        json.dump(history, f)

    print(f"  {feature_name} Done | Best Val Acc: {best_acc:.2f}%")

# ---------------------------------
# MAIN
# ---------------------------------

if __name__ == "__main__":

    print(f"Using Device: {DEVICE}")
    print(f"Freeze Backbone: {FREEZE_BACKBONE}")

    for feature in FEATURES:
        train_feature(feature)

    print("\nAll ViT Training Complete!")