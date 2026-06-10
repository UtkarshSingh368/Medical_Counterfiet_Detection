"""
STEP 4: Extract 768-dim ViT embeddings and build fusion CSVs.

For each medicine (identified by front image):
  - front embedding  (768)
  - back embedding   (768)  — zeros if missing
  - text avg pooling (768)  — average of all text crops, zeros if none
  - logo avg pooling (768)  — average of all logo crops, zeros if none

Final vector: 3072 dimensions + label column.

Creates:
  data/train_fusion.csv
  data/val_fusion.csv
  data/test_fusion.csv

Usage:
    python src/extract_features.py
"""

import os
import re
import csv
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

# ---------------------------------
# CONFIG
# ---------------------------------

ROOT = Path(__file__).resolve().parent.parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIM = 768

FEATURES = ["front", "back", "text", "logo"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ---------------------------------
# VIT FEATURE EXTRACTOR
# ---------------------------------

class ViTFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(
            weights=ViT_B_16_Weights.DEFAULT
        )
        self.vit.heads = nn.Identity()

    def forward(self, x):
        return self.vit(x)

# ---------------------------------
# LOAD ALL 4 VIT MODELS
# ---------------------------------

models = {}

for feature in FEATURES:

    model_path = ROOT / "models" / "vit" / f"{feature}_vit.pth"

    if not model_path.exists():
        print(f"Warning: ViT model not found: {model_path}. Using zeros.")
        continue

    model = ViTFeatureExtractor()

    checkpoint = torch.load(
        str(model_path), map_location=DEVICE
    )

    model.load_state_dict(checkpoint["backbone"])
    model = model.to(DEVICE)
    model.eval()
    models[feature] = model

print(f"Loaded 4 ViT models on {DEVICE}")

# ---------------------------------
# HELPER: extract medicine ID
# ---------------------------------

def get_medicine_id(filename):
    """
    Extract the unique medicine identifier from a filename.

    Examples:
        'Calpol_front.jpg'          -> 'calpol'
        'Calpol_front_fake.jpg'     -> 'calpol'
        'Calpol_back_text_1.jpg'    -> 'calpol'
        'ELDOSIZ-M_FRONT.jpeg'      -> 'eldosiz-m'
    """
    name = os.path.splitext(filename)[0].lower()
    # Remove crop suffixes: _text_N, _logo_N
    name = re.sub(r'_(text|logo)_\d+$', '', name)
    # Remove _fake suffix
    name = re.sub(r'_fake$', '', name)
    # Remove _front or _back
    name = re.sub(r'_(front|back)$', '', name)
    return name

# ---------------------------------
# HELPER: get single embedding
# ---------------------------------

def get_embedding(image_path, model):
    """Extract 768-dim embedding from a single image."""
    img = Image.open(image_path).convert("RGB")
    img = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = model(img)

    return feat.cpu().numpy().flatten()

# ---------------------------------
# HELPER: average embeddings
# ---------------------------------

def get_avg_embedding(file_list, model):
    """Average embeddings from a list of image paths."""
    if not file_list:
        return np.zeros(EMB_DIM)

    feats = [get_embedding(f, model) for f in file_list]
    return np.mean(feats, axis=0)

# ---------------------------------
# PROCESS A SPLIT
# ---------------------------------

def process_split(split):

    crops_dir = ROOT / "data" / "crops" / split
    output_csv = ROOT / "data" / f"{split}_fusion.csv"

    front_dir = crops_dir / "front"
    back_dir = crops_dir / "back"
    text_dir = crops_dir / "text"
    logo_dir = crops_dir / "logo"

    # Collect all front images (they define the samples)
    all_front = []
    for cls in ["real", "fake"]:
        cls_dir = front_dir / cls
        if cls_dir.exists():
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_front.append(
                        (str(cls_dir / f), cls)
                    )

    if not all_front:
        print(f"  No front images found for {split}")
        return

    rows = []

    for front_path, cls_label in sorted(all_front):

        filename = os.path.basename(front_path)
        med_id = get_medicine_id(filename)

        # Label: real=0, fake=1 (consistent with train_vit.py)
        label = 1 if cls_label == "fake" else 0

        # ----- FRONT FEATURE -----
        front_feat = get_embedding(
            front_path, models["front"]
        )

        # ----- BACK FEATURE -----
        # Look for matching back image in SAME class folder
        back_feat = np.zeros(EMB_DIM)
        back_cls_dir = back_dir / cls_label
        if back_cls_dir.exists():
            for f in os.listdir(back_cls_dir):
                if (f.lower().endswith((".jpg", ".jpeg", ".png"))
                        and get_medicine_id(f) == med_id):
                    back_feat = get_embedding(
                        str(back_cls_dir / f), models["back"]
                    )
                    break

        # ----- TEXT FEATURES (average) -----
        text_files = []
        text_cls_dir = text_dir / cls_label
        if text_cls_dir.exists():
            for f in os.listdir(text_cls_dir):
                if (f.lower().endswith((".jpg", ".jpeg", ".png"))
                        and get_medicine_id(f) == med_id):
                    text_files.append(str(text_cls_dir / f))
        
        if "text" in models:
            text_feat = get_avg_embedding(text_files, models["text"])
        else:
            text_feat = np.zeros(EMB_DIM)

        # ----- LOGO FEATURES (average) -----
        logo_files = []
        logo_cls_dir = logo_dir / cls_label
        if logo_cls_dir.exists():
            for f in os.listdir(logo_cls_dir):
                if (f.lower().endswith((".jpg", ".jpeg", ".png"))
                        and get_medicine_id(f) == med_id):
                    logo_files.append(str(logo_cls_dir / f))
        
        if "logo" in models:
            logo_feat = get_avg_embedding(logo_files, models["logo"])
        else:
            logo_feat = np.zeros(EMB_DIM)

        # ----- CONCAT -----
        final_vector = np.concatenate([
            front_feat, back_feat, text_feat, logo_feat
        ])

        row = [label] + final_vector.tolist()
        rows.append(row)

        print(
            f"  {filename} | med={med_id} | label={label} "
            f"| text_crops={len(text_files)} "
            f"| logo_crops={len(logo_files)}"
        )

    # ----- SAVE CSV -----
    with open(str(output_csv), "w", newline="") as f:
        writer = csv.writer(f)
        # Header row
        header = ["label"] + [f"f{i}" for i in range(EMB_DIM * 4)]
        writer.writerow(header)
        writer.writerows(rows)

    print(f"  Saved {output_csv} ({len(rows)} samples)")

# ---------------------------------
# MAIN
# ---------------------------------

if __name__ == "__main__":

    print(f"Using Device: {DEVICE}\n")

    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split}...")
        process_split(split)

    print("\nFeature Extraction Done!")