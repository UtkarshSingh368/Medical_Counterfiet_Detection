"""
STEP 2: Generate crops from raw images using trained YOLO model.

Creates 4 crop types per split (train/val/test):
  - front/  : full front-side images
  - back/   : full back-side images
  - text/   : YOLO-detected text regions
  - logo/   : YOLO-detected logo regions

Each crop type has real/ and fake/ subdirectories.
Real/Fake is determined by '_fake' in the filename.
Front/Back is determined by '_front' or '_back' in the filename.

Usage:
    python src/make_crops.py
"""

import os
import cv2
from pathlib import Path
from ultralytics import YOLO

# -----------------------------------
# CONFIG
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

YOLO_MODEL = str(
    ROOT / "runs" / "detect" / "medicine_detector"
    / "weights" / "best.pt"
)

SPLITS = ["train", "val", "test"]
FEATURES = ["front", "back", "text", "logo"]
CLASSES = ["real", "fake"]
MIN_CROP_SIZE = 20

# -----------------------------------
# LOAD MODEL
# -----------------------------------

print(f"Loading YOLO model: {YOLO_MODEL}")
model = YOLO(YOLO_MODEL)

# -----------------------------------
# CREATE OUTPUT FOLDERS
# -----------------------------------

for split in SPLITS:
    for feature in FEATURES:
        for cls in CLASSES:
            os.makedirs(
                ROOT / "data" / "crops" / split / feature / cls,
                exist_ok=True,
            )

# -----------------------------------
# HELPERS
# -----------------------------------

def get_label(filename):
    """Return 'fake' if filename contains '_fake', else 'real'."""
    return "fake" if "_fake" in filename.lower() else "real"


def get_side(filename):
    """Return 'front', 'back', or None."""
    lower = filename.lower()
    if "_front" in lower:
        return "front"
    elif "_back" in lower:
        return "back"
    return None

# -----------------------------------
# PROCESS ALL SPLITS
# -----------------------------------

for split in SPLITS:

    image_dir = ROOT / "data" / "raw" / split / "images"

    if not image_dir.exists():
        print(f"\nSkipping {split}: directory not found")
        continue

    print(f"\nProcessing {split}")

    # Collect image files from FLAT directory
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print(f"  No images found in {image_dir}")
        continue

    for image_name in sorted(image_files):

        image_path = str(image_dir / image_name)
        img = cv2.imread(image_path)

        if img is None:
            print(f"  Skipped (unreadable): {image_name}")
            continue

        label = get_label(image_name)
        side = get_side(image_name)

        # --------------------------
        # FRONT / BACK (full image)
        # --------------------------

        if side in ("front", "back"):
            save_path = str(
                ROOT / "data" / "crops" / split
                / side / label / image_name
            )
            cv2.imwrite(save_path, img)

        # --------------------------
        # YOLO DETECTION
        # --------------------------

        results = model.predict(
            source=image_path,
            conf=0.15,
            verbose=False,
        )

        text_counter = 0
        logo_counter = 0
        stem = os.path.splitext(image_name)[0]

        for result in results:
            for box in result.boxes:

                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                h, w = crop.shape[:2]
                if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
                    continue

                # TEXT (class 0)
                if cls_id == 0:
                    text_counter += 1
                    save_name = f"{stem}_text_{text_counter}.jpg"
                    save_path = str(
                        ROOT / "data" / "crops" / split
                        / "text" / label / save_name
                    )
                    cv2.imwrite(save_path, crop)

                # LOGO (class 1)
                elif cls_id == 1:
                    logo_counter += 1
                    save_name = f"{stem}_logo_{logo_counter}.jpg"
                    save_path = str(
                        ROOT / "data" / "crops" / split
                        / "logo" / label / save_name
                    )
                    cv2.imwrite(save_path, crop)

        print(
            f"  {image_name}: side={side}, label={label}, "
            f"text={text_counter}, logo={logo_counter}"
        )

print("\nCropping Completed!")