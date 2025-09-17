import warnings
warnings.filterwarnings('ignore')
import time
import torch
import os
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path


def find_latest_best(base_dir: str = 'runs/detect') -> str:
    """Return the path to the most recently modified best.pt under runs like runs/detect/train*/weights/best.pt"""
    base = Path(base_dir)
    candidates = []
    for p in base.glob('train*/weights/best.pt'):
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0
        candidates.append((mtime, p))
    if not candidates:
        return ''
    candidates.sort(key=lambda x: x[0], reverse=True)
    return str(candidates[0][1])

def calculate_model_params_mb(model):
    """Compute model parameter count (MB)"""
    total_params = sum(p.numel() for p in model.model.parameters())
    params_mb = total_params  / (1024 * 1024)
    return params_mb

def calculate_fps(model, data_path, imgsz=640, num_samples=100, split='val'):
    """Compute FPS on given split ('val' or 'test'), supporting autosplit .txt lists"""
    import cv2
    import numpy as np
    from pathlib import Path
    import random

    data_yaml = Path(data_path)
    if not data_yaml.exists():
        print("Warning: Data config file not found for FPS calculation")
        return 0.0

    import yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    dataset_root = Path(data_config.get('path', '.'))
    split_entry = data_config.get(split, 'images')

    image_files = []
    split_path = Path(split_entry)
    if not split_path.is_absolute():
        split_path = (dataset_root / split_entry)

    # Case A: autosplit .txt list
    if split_path.suffix.lower() == '.txt':
        if not split_path.exists():
            # try alongside YAML
            alt = data_yaml.parent / split_entry
            if alt.exists():
                split_path = alt
        if split_path.exists():
            try:
                with open(split_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        p = Path(line)
                        if not p.is_absolute():
                            cand1 = dataset_root / p
                            cand2 = data_yaml.parent / p
                            if cand1.exists():
                                p = cand1
                            elif cand2.exists():
                                p = cand2
                            else:
                                p = cand1
                        if p.exists():
                            image_files.append(p)
            except Exception:
                pass
    else:
        # Case B: directory path
        if not split_path.exists():
            split_path = dataset_root / 'images'
        if split_path.exists() and split_path.is_dir():
            image_files = list(split_path.glob('*.jpg')) + \
                          list(split_path.glob('*.jpeg')) + \
                          list(split_path.glob('*.png'))

    if len(image_files) == 0:
        print(f"Warning: No {split} images found for FPS calculation")
        return 0.0

    # Randomly sample images for testing
    random.shuffle(image_files)
    test_images = image_files[:min(num_samples, len(image_files))]

    # Warmup
    dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(10):
        _ = model(dummy_img, verbose=False)

    # Timed runs
    times = []
    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            start_time = time.time()
            _ = model(img, verbose=False)
            end_time = time.time()
            times.append(end_time - start_time)

    if times:
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        return fps
    else:
        return 0.0

if __name__ == '__main__':
    print("🚀 Loading model...")
    weights_path = find_latest_best('runs/detect')
    if not weights_path:
        print("⚠️ No trained model found under runs/detect/*/weights/best.pt. Please train a model first.")
        raise SystemExit(1)
    print(f"Using weights: {weights_path}")
    model = YOLO(weights_path)

    # Compute model parameters
    print("📊 Calculating model parameters...")
    params_mb = calculate_model_params_mb(model)
    print(f"Model Parameters: {params_mb:.2f} MB")
    
    print("\n" + "="*50)
    print("📈 MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Model Parameters: {params_mb:.2f} MB")
    print("="*50)

    # Run validation
    print("\n🔍 Running validation...")
    results = model.val(data='./fire.yaml',
                        split='val',
                        imgsz=640,
                        batch=16,
                        # iou=0.7,
                        # rect=False,
                        # save_json=True, # if you need to cal coco metrice
                        project='',
                        name='',
                        )
    
    # Final results
    print("\n" + "="*50)
    print("📊 FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Model Parameters: {params_mb:.2f} MB")
    if hasattr(results, 'box'):
        print(f"mAP50: {results.box.map50:.3f}")
        print(f"mAP50-95: {results.box.map:.3f}")
        print(f"Precision: {results.box.mp:.3f}")
        print(f"Recall: {results.box.mr:.3f}")
    print("="*50)
