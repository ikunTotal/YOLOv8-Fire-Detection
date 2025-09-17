#!/usr/bin/env python3
"""
Dataset setup script
Copy the firedetn dataset into the project's datasets directory.
"""

import os
import shutil
from pathlib import Path

def setup_dataset():
    """Place the dataset in the correct location"""
    # Source and target paths
    source_path = Path("/root/lanyun-tmp/firedetn")
    target_path = Path("/root/yolov8fire/datasets/firedetn")

    print("🚀 Starting dataset setup...")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")

    # Check source path exists
    if not source_path.exists():
        print(f"❌ Source dataset path does not exist: {source_path}")
        return False

    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)

    # Confirm overwrite if target exists and not empty
    if target_path.exists() and any(target_path.iterdir()):
        response = input(f"Target path {target_path} already exists and is not empty. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operation cancelled by user")
            return False

    try:
        # Copy dataset
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)

        print("✅ Dataset copy completed!")

        # Verify dataset structure
        images_dir = target_path / "images"
        labels_dir = target_path / "labels"

        if images_dir.exists() and labels_dir.exists():
            image_count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
            label_count = len(list(labels_dir.glob("*.txt")))

            print("📊 Dataset stats:")
            print(f"  - Images: {image_count}")
            print(f"  - Labels: {label_count}")
            print(f"  - Images dir: {images_dir}")
            print(f"  - Labels dir: {labels_dir}")

            if image_count > 0 and label_count > 0:
                print("✅ Dataset structure verified!")
                return True
            else:
                print("❌ Dataset structure verification failed!")
                return False
        else:
            print("❌ Dataset structure incomplete!")
            return False

    except Exception as e:
        print(f"❌ Error while copying dataset: {e}")
        return False

def verify_training_setup():
    """Verify training configuration"""
    print("\n🔍 Verifying training configuration...")

    # Check fire.yaml
    yaml_path = Path("/root/yolov8fire/fire.yaml")
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            content = f.read()
            if "datasets/firedetn" in content:
                print("✅ fire.yaml is configured correctly")
            else:
                print("❌ fire.yaml configuration might be incorrect")
    else:
        print("❌ fire.yaml does not exist")

    # Check model config file
    model_cfg = Path("/root/yolov8fire/ultralytics/cfg/models/yolov8s-ircb-BiLevelRoutingAttention.yaml")
    if model_cfg.exists():
        print("✅ Model config file exists")
    else:
        print("❌ Model config file does not exist")

    # Check train.py
    train_script = Path("/root/yolov8fire/train.py")
    if train_script.exists():
        print("✅ Training script exists")
    else:
        print("❌ Training script does not exist")

if __name__ == "__main__":
    print("=" * 60)
    print("🔥 YOLOv8 Fire Detection Dataset Setup")
    print("=" * 60)

    # Setup dataset
    if setup_dataset():
        print("\n🎉 Dataset setup completed!")

        # Verify training config
        verify_training_setup()

        print("\n📝 Next steps:")
        print("1. Run training: python train.py")
        print("2. Run validation: python val.py")
        print("3. Check results in: runs/detect/")

    else:
        print("\n❌ Dataset setup failed!")
        print("Please check if the source dataset path is correct.")

    print("=" * 60)
