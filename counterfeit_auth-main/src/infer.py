"""
STEP 6: Inference — predict REAL or FAKE for a single medicine image.

Pipeline: Image -> YOLO (text/logo crops) -> ViT features -> Fusion model

The input image is treated as the FRONT side.
Back features default to zeros (single-image mode).
For best accuracy, provide --back_image as well.

Usage:
    python src/infer.py --image path/to/medicine.jpg
    python src/infer.py --image front.jpg --back_image back.jpg
"""

import os
import sys
import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

# -----------------------------------
# CONFIG
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_MODEL = str(
    ROOT / "runs" / "detect" / "medicine_detector"
    / "weights" / "best.pt"
)

EMB_DIM = 768

# -----------------------------------
# TRANSFORM (identical to extract_features.py)
# -----------------------------------

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# -----------------------------------
# VIT FEATURE EXTRACTOR (identical)
# -----------------------------------

class ViTFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(
            weights=ViT_B_16_Weights.DEFAULT
        )
        self.vit.heads = nn.Identity()

    def forward(self, x):
        return self.vit(x)

# -----------------------------------
# FUSIONNET (identical to train_fusion.py)
# -----------------------------------

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
        return self.feature_layer(x)

# -----------------------------------
# HELPER: image to embedding
# -----------------------------------

def get_embedding(img_bgr, model):
    """Extract embedding from a BGR numpy array."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = model(tensor)

    return feat.cpu().numpy().flatten()

# -----------------------------------
# MAIN
# -----------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Medicine Counterfeit Detection"
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to front medicine image",
    )
    parser.add_argument(
        "--back_image", default=None,
        help="Optional path to back medicine image",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # -----------------------------------
    # LOAD VIT MODELS
    # -----------------------------------

    features_list = ["front", "back", "text", "logo"]
    vit_models = {}

    for feature in features_list:
        model_path = ROOT / "models" / "vit" / f"{feature}_vit.pth"

        if not model_path.exists():
            print(f"Error: ViT model not found: {model_path}")
            sys.exit(1)

        m = ViTFeatureExtractor()
        ckpt = torch.load(str(model_path), map_location=DEVICE)
        m.load_state_dict(ckpt["backbone"])
        m.to(DEVICE)
        m.eval()
        vit_models[feature] = m

    # -----------------------------------
    # LOAD FUSION MODEL
    # -----------------------------------

    fusion_path = ROOT / "models" / "fusion" / "fusion_model.pth"

    if not fusion_path.exists():
        print(f"Error: Fusion model not found: {fusion_path}")
        sys.exit(1)

    fusion_model = FusionNet()
    fusion_model.load_state_dict(
        torch.load(str(fusion_path), map_location=DEVICE)
    )
    fusion_model.to(DEVICE)
    fusion_model.eval()

    # -----------------------------------
    # FRONT FEATURE
    # -----------------------------------

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Cannot read image: {args.image}")
        sys.exit(1)

    front_feat = get_embedding(image, vit_models["front"])

    # -----------------------------------
    # BACK FEATURE
    # -----------------------------------

    if args.back_image and os.path.exists(args.back_image):
        back_img = cv2.imread(args.back_image)
        if back_img is not None:
            back_feat = get_embedding(
                back_img, vit_models["back"]
            )
        else:
            print("Warning: Cannot read back image, using zeros")
            back_feat = np.zeros(EMB_DIM)
    else:
        back_feat = np.zeros(EMB_DIM)

    # -----------------------------------
    # YOLO DETECTION (text/logo crops)
    # -----------------------------------

    yolo = YOLO(YOLO_MODEL)

    # Detect on front image
    text_feats = []
    logo_feats = []

    for img_to_detect in [image]:
        results = yolo.predict(
            source=img_to_detect,
            conf=0.25,
            verbose=False,
        )

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                crop = img_to_detect[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                if cls_id == 0:  # Text
                    text_feats.append(
                        get_embedding(crop, vit_models["text"])
                    )
                elif cls_id == 1:  # Logo
                    logo_feats.append(
                        get_embedding(crop, vit_models["logo"])
                    )

    # Also detect on back image if provided
    if args.back_image and os.path.exists(args.back_image):
        back_img = cv2.imread(args.back_image)
        if back_img is not None:
            results = yolo.predict(
                source=back_img, conf=0.25, verbose=False,
            )
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = back_img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    if cls_id == 0:
                        text_feats.append(
                            get_embedding(crop, vit_models["text"])
                        )
                    elif cls_id == 1:
                        logo_feats.append(
                            get_embedding(crop, vit_models["logo"])
                        )

    # -----------------------------------
    # AVERAGE POOLING
    # -----------------------------------

    text_feat = (
        np.mean(text_feats, axis=0)
        if text_feats else np.zeros(EMB_DIM)
    )
    logo_feat = (
        np.mean(logo_feats, axis=0)
        if logo_feats else np.zeros(EMB_DIM)
    )

    # -----------------------------------
    # CONCAT & PREDICT
    # -----------------------------------

    final_vector = np.concatenate([
        front_feat, back_feat, text_feat, logo_feat,
    ])

    tensor = torch.tensor(
        final_vector, dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = fusion_model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    # Label mapping: real=0, fake=1 (consistent with training)
    label = "FAKE" if pred.item() == 1 else "REAL"

    print("\n" + "=" * 30)
    print(f"  Prediction : {label}")
    print(f"  Confidence : {conf.item() * 100:.2f}%")
    print(f"  P(Real)    : {probs[0][0].item() * 100:.2f}%")
    print(f"  P(Fake)    : {probs[0][1].item() * 100:.2f}%")
    print("=" * 30)


if __name__ == "__main__":
    main()