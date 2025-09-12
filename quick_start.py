#!/usr/bin/env python3
"""
快速启动脚本
自动设置数据集并开始训练
"""

import os
import subprocess
import sys
from pathlib import Path

def check_dataset():
    """检查数据集是否存在"""
    dataset_path = Path("/root/yolov8fire/datasets/firedetn")
    images_path = dataset_path / "images"
    labels_path = dataset_path / "labels"
    
    if dataset_path.exists() and images_path.exists() and labels_path.exists():
        image_count = len(list(images_path.glob("*.jpg"))) + len(list(images_path.glob("*.png")))
        label_count = len(list(labels_path.glob("*.txt")))
        
        if image_count > 0 and label_count > 0:
            print(f"✅ 数据集已就绪: {image_count} 张图像, {label_count} 个标签")
            return True
    
    print("❌ 数据集未就绪，请先运行: python setup_dataset.py")
    return False

def run_training():
    """运行训练"""
    print("🚀 开始训练...")
    try:
        # 切换到项目目录
        os.chdir("/root/yolov8fire")
        
        # 运行训练脚本
        result = subprocess.run([sys.executable, "train.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 训练完成！")
            print("📊 查看结果: runs/detect/ 目录")
        else:
            print("❌ 训练失败！")
            print("错误信息:", result.stderr)
            
    except Exception as e:
        print(f"❌ 运行训练时出错: {e}")

def run_validation():
    """运行验证"""
    print("🔍 开始验证...")
    try:
        result = subprocess.run([sys.executable, "val.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 验证完成！")
        else:
            print("❌ 验证失败！")
            print("错误信息:", result.stderr)
            
    except Exception as e:
        print(f"❌ 运行验证时出错: {e}")

def main():
    print("=" * 60)
    print("🔥 YOLOv8 Fire Detection Quick Start")
    print("=" * 60)
    
    # 检查数据集
    if not check_dataset():
        return
    
    # 询问用户要执行的操作
    print("\n请选择操作:")
    print("1. 开始训练")
    print("2. 运行验证")
    print("3. 训练 + 验证")
    print("4. 退出")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == "1":
        run_training()
    elif choice == "2":
        run_validation()
    elif choice == "3":
        run_training()
        print("\n" + "="*40)
        run_validation()
    elif choice == "4":
        print("👋 再见！")
    else:
        print("❌ 无效选择！")

if __name__ == "__main__":
    main()
