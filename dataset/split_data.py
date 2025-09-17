#!/usr/bin/env python3
"""
Dataset split script (autosplit, no file moves).
It creates the following files at the dataset root datasets/firedetn:
- autosplit_train.txt
- autosplit_val.txt
- autosplit_test.txt

Note: fire.yaml is already configured to use these list files.
"""
from pathlib import Path
from ultralytics.data.utils import autosplit

DATASET_ROOT = Path("/root/yolov8fire/datasets/firedetn")
IMAGES_DIR = DATASET_ROOT / "images"


def main():
    if not IMAGES_DIR.exists():
        raise SystemExit(f"Images directory not found: {IMAGES_DIR}")

    print("🚀 Starting autosplit ...")
    print(f"Images dir: {IMAGES_DIR}")

    # 80% train, 10% val, 10% test; use only images with corresponding labels
    autosplit(path=IMAGES_DIR, weights=(0.8, 0.1, 0.1), annotated_only=True)

    ok = True
    for split in ("train", "val", "test"):
        f = DATASET_ROOT / f"autosplit_{split}.txt"
        status = "✅" if f.exists() else "❌"
        if not f.exists():
            ok = False
        print(f"{status} {f}")

    if ok:
        print("🎉 Autosplit completed! Ensure fire.yaml points to autosplit_*.txt files.")
    else:
        print("⚠️ Autosplit list not found. Please check paths and labels.")


if __name__ == "__main__":
    main()
