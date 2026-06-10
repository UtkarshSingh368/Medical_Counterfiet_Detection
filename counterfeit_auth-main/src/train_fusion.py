"""
STEP 5: Train the Fusion classifier on concatenated ViT features.

Input:  3072-dim feature vector (768 * 4)
Output: REAL (0) or FAKE (1)

Architecture (MUST match infer.py and plots.py):
    feature_layer: 3072 -> 1024 -> 256
    classifier:    256 -> 2

Saves:
    models/fusion/fusion_model.pth
    models/fusion/fusion_loss.png
    models/fusion/fusion_accuracy.png

Usage:
    python src/train_fusion.py
"""

import os
import torch
import pandas as pd
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

# ----------------------------------
# CONFIG
# ----------------------------------

ROOT = Path(__file__).resolve().parent.parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4

# ----------------------------------
# DATASET
# ----------------------------------

class FusionDataset(Dataset):

    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self.labels = torch.tensor(
            data.iloc[:, 0].values, dtype=torch.long
        )
        self.features = torch.tensor(
            data.iloc[:, 1:].values, dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ----------------------------------
# FUSIONNET (identical in all files)
# ----------------------------------

class FusionNet(nn.Module):
    """
    Fusion classifier. Architecture MUST be identical in:
      - train_fusion.py
      - infer.py
      - plots.py
    """

    def __init__(self):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(256, 2)

    def forward(self, x):
        features = self.feature_layer(x)
        return self.classifier(features)

    def get_features(self, x):
        """Extract 256-dim intermediate features (for t-SNE)."""
        return self.feature_layer(x)

# ----------------------------------
# MAIN
# ----------------------------------

def main():

    print(f"Using Device: {DEVICE}")

    # Load data
    train_csv = str(ROOT / "data" / "train_fusion.csv")
    val_csv = str(ROOT / "data" / "val_fusion.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Not found: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Not found: {val_csv}")

    train_dataset = FusionDataset(train_csv)
    val_dataset = FusionDataset(val_csv)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    )

    # Model
    model = FusionNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR,
    )

    best_acc = 0.0
    loss_history = []
    acc_history = []

    save_dir = ROOT / "models" / "fusion"
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------
    # TRAINING LOOP
    # ----------------------------------

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0.0

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(features)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = (100.0 * correct / total) if total > 0 else 0.0

        loss_history.append(avg_loss)
        acc_history.append(acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS}"
            f" | Loss={avg_loss:.4f}"
            f" | Acc={acc:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                str(save_dir / "fusion_model.pth"),
            )

    print(f"\nBest Val Accuracy: {best_acc:.2f}%")

    # ----------------------------------
    # LOSS PLOT
    # ----------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.title("Fusion Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(str(save_dir / "fusion_loss.png"), dpi=150)
    plt.close()

    # ----------------------------------
    # ACCURACY PLOT
    # ----------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(acc_history)
    plt.title("Fusion Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid()
    plt.savefig(str(save_dir / "fusion_accuracy.png"), dpi=150)
    plt.close()

    print("Fusion Training Done!")


if __name__ == "__main__":
    main()