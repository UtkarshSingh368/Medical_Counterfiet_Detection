"""
Utility script to clean and prep CVAT exports for YOLO training.

CVAT often exports nested folders or leaves orphan labels if images are deleted.
This script performs the following guarantees:
1. Flattens any subdirectories inside images/ and labels/
2. Converts image extensions to lowercase (.JPEG -> .jpeg)
3. Deletes any label (.txt) that does not have a matching image (Orphan cleanup).
4. Deletes any image that does not have a matching label.

Usage:
    python src/prep_dataset.py
"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

def flatten_directory(base_dir):
    """Move all files from subdirectories into the base directory, then delete subdirectories."""
    if not base_dir.exists():
        return

    for item in base_dir.rglob("*"):
        if item.is_file():
            # If the file is already in the base_dir, skip
            if item.parent == base_dir:
                continue
            
            # Move to base_dir
            dest = base_dir / item.name
            
            # Handle duplicates if they exist
            counter = 1
            while dest.exists():
                dest = base_dir / f"{item.stem}_{counter}{item.suffix}"
                counter += 1
                
            shutil.move(str(item), str(dest))
            
    # Remove empty subdirectories
    for item in list(base_dir.rglob("*"))[::-1]:
        if item.is_dir() and not os.listdir(item):
            item.rmdir()


def clean_split(split):
    print(f"\n--- Processing {split} split ---")
    img_dir = ROOT / split / "images"
    lbl_dir = ROOT / split / "labels"
    
    if not img_dir.exists():
        print(f"Directory missing: {img_dir}")
        return
        
    if not lbl_dir.exists():
        print(f"Directory missing: {lbl_dir}")
        return

    # 1. Flatten nested folders (e.g. real/ fake/)
    flatten_directory(img_dir)
    flatten_directory(lbl_dir)
    print("Flattened directories.")

    # 2. Normalize extensions and build stem sets
    img_stems = {}
    for f in img_dir.glob("*"):
        if f.is_file():
            # Standardize extensions to lowercase to avoid Windows mismatches
            if f.suffix != f.suffix.lower():
                new_path = f.with_suffix(f.suffix.lower())
                f.rename(new_path)
                f = new_path
            
            # YOLO matches stems case-insensitively, but it's safest to match exact stems.
            # We map lowercase stem to the actual Path object
            img_stems[f.stem.lower()] = f

    lbl_stems = {}
    for f in lbl_dir.glob("*.txt"):
        if f.is_file():
            lbl_stems[f.stem.lower()] = f

    # 3. Purge Orphan Labels (labels with no image)
    orphans_removed = 0
    for lower_stem, lbl_path in lbl_stems.items():
        if lower_stem not in img_stems:
            print(f"Removing orphan label: {lbl_path.name}")
            lbl_path.unlink()
            orphans_removed += 1

    # 4. Purge Images without labels (Optional, YOLO treats them as background, but good for purity)
    missing_labels = 0
    for lower_stem, img_path in img_stems.items():
        if lower_stem not in lbl_stems:
            print(f"Removing image without label: {img_path.name}")
            img_path.unlink()
            missing_labels += 1
            
    print(f"Summary for {split}:")
    print(f"  Removed {orphans_removed} orphan labels.")
    print(f"  Removed {missing_labels} images missing labels.")


if __name__ == "__main__":
    print(f"Dataset Prep Script\nTarget: {ROOT}")
    for s in ["train", "val", "test"]:
        clean_split(s)
    print("\nDataset preparation complete. Safe for YOLO training.")
