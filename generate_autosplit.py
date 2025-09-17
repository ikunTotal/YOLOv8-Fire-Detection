#!/usr/bin/env python3
"""
Generate autosplit train/val/test lists for firedetn without moving files.
This will create autosplit_train.txt, autosplit_val.txt, autosplit_test.txt
under /root/yolov8fire/datasets/firedetn
"""
from pathlib import Path
from ultralytics.data.utils import autosplit

IMAGES_DIR = Path("/root/yolov8fire/datasets/firedetn/images")


def main():
    if not IMAGES_DIR.exists():
        raise SystemExit(f"Images directory not found: {IMAGES_DIR}")

    print("🚀 Running autosplit ...")
    print(f"Images dir: {IMAGES_DIR}")
    # 80% train, 10% val, 10% test; only include images with labels
    autosplit(path=IMAGES_DIR, weights=(0.8, 0.1, 0.1), annotated_only=True)

    for split in ("train", "val", "test"):
        f = IMAGES_DIR.parent / f"autosplit_{split}.txt"
        status = "✅" if f.exists() else "❌"
        print(f"{status} {f}")

    print("Done. Files are created at the dataset root. If all files show ✅, you can start training.")


if __name__ == "__main__":
    main()

