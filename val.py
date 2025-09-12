import warnings
warnings.filterwarnings('ignore')
import time
import torch
import os
from datetime import datetime
from ultralytics import YOLO

def calculate_model_params_mb(model):
    """计算模型参数量(MB)"""
    total_params = sum(p.numel() for p in model.model.parameters())
    params_mb = total_params  / (1024 * 1024)
    return params_mb

def calculate_fps(model, data_path, imgsz=640, num_samples=100):
    """计算FPS"""
    import cv2
    import numpy as np
    from pathlib import Path
    
    # 获取测试图像路径
    data_yaml = Path(data_path)
    if data_yaml.exists():
        import yaml
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        test_path = Path(data_config['path']) / data_config.get('test', 'images')
        if not test_path.exists():
            test_path = Path(data_config['path']) / 'images'
        
        # 获取图像文件列表
        image_files = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
        if len(image_files) == 0:
            print("Warning: No test images found for FPS calculation")
            return 0.0
        
        # 随机选择样本进行测试
        import random
        test_images = random.sample(image_files, min(num_samples, len(image_files)))
        
        # 预热
        dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(10):
            _ = model(dummy_img, verbose=False)
        
        # 计时测试
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
    else:
        print("Warning: Data config file not found for FPS calculation")
        return 0.0

if __name__ == '__main__':
    print("🚀 Loading model...")
    model = YOLO('./runs/detect/train10/weights/best.pt')
    
    # 计算模型参数量
    print("📊 Calculating model parameters...")
    params_mb = calculate_model_params_mb(model)
    print(f"Model Parameters: {params_mb:.2f} MB")
    
    # 计算FPS
    print("⚡ Calculating FPS...")
    fps = calculate_fps(model, './fire.yaml', imgsz=640, num_samples=50)
    print(f"FPS: {fps:.2f}")
    
    print("\n" + "="*50)
    print("📈 MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Model Parameters: {params_mb:.2f} MB")
    print(f"FPS: {fps:.2f}")
    print("="*50)
    
    # 运行验证
    print("\n🔍 Running validation...")
    results = model.val(data='./fire.yaml',
                        split='test',
                        imgsz=640,
                        batch=16,
                        # iou=0.7,
                        # rect=False,
                        # save_json=True, # if you need to cal coco metrice
                        project='',
                        name='',
                        )
    
    # 输出最终结果
    print("\n" + "="*50)
    print("📊 FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Model Parameters: {params_mb:.2f} MB")
    print(f"FPS: {fps:.2f}")
    if hasattr(results, 'box'):
        print(f"mAP50: {results.box.map50:.3f}")
        print(f"mAP50-95: {results.box.map:.3f}")
        print(f"Precision: {results.box.mp:.3f}")
        print(f"Recall: {results.box.mr:.3f}")
    print("="*50)
