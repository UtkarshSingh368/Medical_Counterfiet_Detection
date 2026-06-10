"""
STEP 7: Generate evaluation plots from the validation fusion CSV.

Produces:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
  - t-SNE Visualization

Usage:
    python src/plots.py
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
)
from sklearn.manifold import TSNE

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# ---------------------------------
# CONFIG
# ---------------------------------

ROOT = Path(__file__).resolve().parent.parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULT_DIR = ROOT / "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------
# DATASET
# ---------------------------------

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

# ---------------------------------
# FUSIONNET (identical to train_fusion.py)
# ---------------------------------

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

# ---------------------------------
# MAIN
# ---------------------------------

def main():

    print(f"Using Device: {DEVICE}")

    # Load data
    val_csv = str(ROOT / "data" / "val_fusion.csv")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Not found: {val_csv}")

    dataset = FusionDataset(val_csv)
    print(f"Val samples: {len(dataset)}")

    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
    )

    # Load model
    model_path = str(
        ROOT / "models" / "fusion" / "fusion_model.pth"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Not found: {model_path}")

    model = FusionNet()
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    # ---------------------------------
    # COLLECT PREDICTIONS
    # ---------------------------------

    all_labels = []
    all_preds = []
    all_probs = []
    all_features = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)

            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            tsne_features = model.get_features(features)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_features.extend(tsne_features.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_features = np.array(all_features)

    # ---------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------

    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Real", "Fake"],
    ))

    # ---------------------------------
    # CONFUSION MATRIX
    # ---------------------------------

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real", "Fake"],
    )
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(
        str(RESULT_DIR / "confusion_matrix.png"), dpi=150
    )
    plt.close()
    print("Saved: confusion_matrix.png")

    # ---------------------------------
    # ROC CURVE
    # ---------------------------------

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(
        str(RESULT_DIR / "roc_curve.png"), dpi=150
    )
    plt.close()
    print(f"Saved: roc_curve.png (AUC={roc_auc:.4f})")

    # ---------------------------------
    # PRECISION RECALL CURVE
    # ---------------------------------

    precision, recall, _ = precision_recall_curve(
        all_labels, all_probs
    )

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.savefig(
        str(RESULT_DIR / "precision_recall_curve.png"), dpi=150
    )
    plt.close()
    print("Saved: precision_recall_curve.png")

    # ---------------------------------
    # t-SNE VISUALIZATION
    # ---------------------------------

    n_samples = len(all_features)

    if n_samples < 3:
        print("WARNING: Too few samples for t-SNE, skipping.")
    else:
        perplexity = min(20, n_samples - 1)
        print(
            f"Running t-SNE (n={n_samples}, "
            f"perplexity={perplexity})..."
        )

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
        )
        tsne_result = tsne.fit_transform(all_features)

        plt.figure(figsize=(8, 6))

        real_idx = all_labels == 0
        fake_idx = all_labels == 1

        plt.scatter(
            tsne_result[real_idx, 0],
            tsne_result[real_idx, 1],
            label="Real", alpha=0.7, c="tab:blue",
        )
        plt.scatter(
            tsne_result[fake_idx, 0],
            tsne_result[fake_idx, 1],
            label="Fake", alpha=0.7, c="tab:red",
        )

        plt.title("t-SNE Visualization")
        plt.legend()
        plt.grid()
        plt.savefig(
            str(RESULT_DIR / "tsne_visualization.png"), dpi=150
        )
        plt.close()
        print("Saved: tsne_visualization.png")

    print("\nAll Evaluation Plots Saved!")


if __name__ == "__main__":
    main()