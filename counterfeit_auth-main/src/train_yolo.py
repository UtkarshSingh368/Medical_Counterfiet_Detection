"""
STEP 1: Train YOLOv8s for Text and Logo detection on medicine packages.

Usage:
    python src/train_yolo.py

Expects (after manual flattening):
    data/raw/train/images/  <- all images flat (real + fake)
    data/raw/train/labels/  <- YOLO .txt annotations
    data/raw/val/images/
    data/raw/val/labels/
    data/yolo_meds.yaml
"""

import os
import torch
from pathlib import Path
from ultralytics import YOLO

# -----------------------------------
# CONFIG
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "yolov8n.pt"
DATA_YAML = str(ROOT / "data" / "yolo_meds.yaml")
EPOCHS = 40
IMG_SIZE = 640
BATCH_SIZE = 8
PROJECT_NAME = str(ROOT / "runs" / "detect")
RUN_NAME = "medicine_detector"

# -----------------------------------
# TRAIN
# -----------------------------------

def main():

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\nUsing Device: {device}")

    # Verify YAML exists
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"YAML not found: {DATA_YAML}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=2,
        device=device,
        pretrained=True,
        cache=True,
        amp=True,
        patience=20,
        project=PROJECT_NAME,
        name=RUN_NAME,
        save=True,
        plots=True,
        verbose=True,
    )

    # ----------------------------
    # VALIDATION
    # ----------------------------

    print("\nRunning Validation...")

    metrics = model.val(
        data=DATA_YAML,
        split="val",
        plots=True,
    )

    print("\nValidation Complete")
    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")

    best_path = os.path.join(
        PROJECT_NAME, RUN_NAME, "weights", "best.pt"
    )
    print(f"\nBest Model Saved At: {best_path}")


if __name__ == "__main__":
    main()