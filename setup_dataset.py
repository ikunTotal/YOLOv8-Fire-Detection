#!/usr/bin/env python3
"""
数据集设置脚本
将firedetn数据集移动到项目datasets目录
"""

import os
import shutil
from pathlib import Path

def setup_dataset():
    """设置数据集到正确位置"""
    # 源路径和目标路径
    source_path = Path("/root/lanyun-tmp/firedetn")
    target_path = Path("/root/yolov8fire/datasets/firedetn")
    
    print("🚀 开始设置数据集...")
    print(f"源路径: {source_path}")
    print(f"目标路径: {target_path}")
    
    # 检查源路径是否存在
    if not source_path.exists():
        print(f"❌ 源数据集路径不存在: {source_path}")
        return False
    
    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 如果目标路径已存在且不为空，询问是否覆盖
    if target_path.exists() and any(target_path.iterdir()):
        response = input(f"目标路径 {target_path} 已存在且不为空，是否覆盖？(y/N): ")
        if response.lower() != 'y':
            print("❌ 用户取消操作")
            return False
    
    try:
        # 复制数据集
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)
        
        print("✅ 数据集复制完成！")
        
        # 验证数据集结构
        images_dir = target_path / "images"
        labels_dir = target_path / "labels"
        
        if images_dir.exists() and labels_dir.exists():
            image_count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
            label_count = len(list(labels_dir.glob("*.txt")))
            
            print(f"📊 数据集统计:")
            print(f"  - 图像数量: {image_count}")
            print(f"  - 标签数量: {label_count}")
            print(f"  - 图像目录: {images_dir}")
            print(f"  - 标签目录: {labels_dir}")
            
            if image_count > 0 and label_count > 0:
                print("✅ 数据集结构验证通过！")
                return True
            else:
                print("❌ 数据集结构验证失败！")
                return False
        else:
            print("❌ 数据集结构不完整！")
            return False
            
    except Exception as e:
        print(f"❌ 复制数据集时出错: {e}")
        return False

def verify_training_setup():
    """验证训练配置"""
    print("\n🔍 验证训练配置...")
    
    # 检查fire.yaml
    yaml_path = Path("/root/yolov8fire/fire.yaml")
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            content = f.read()
            if "datasets/firedetn" in content:
                print("✅ fire.yaml 配置正确")
            else:
                print("❌ fire.yaml 配置可能有问题")
    else:
        print("❌ fire.yaml 文件不存在")
    
    # 检查模型配置文件
    model_cfg = Path("/root/yolov8fire/ultralytics/cfg/models/yolov8s-ircb-BiLevelRoutingAttention.yaml")
    if model_cfg.exists():
        print("✅ 模型配置文件存在")
    else:
        print("❌ 模型配置文件不存在")
    
    # 检查train.py
    train_script = Path("/root/yolov8fire/train.py")
    if train_script.exists():
        print("✅ 训练脚本存在")
    else:
        print("❌ 训练脚本不存在")

if __name__ == "__main__":
    print("=" * 60)
    print("🔥 YOLOv8 Fire Detection Dataset Setup")
    print("=" * 60)
    
    # 设置数据集
    if setup_dataset():
        print("\n🎉 数据集设置完成！")
        
        # 验证训练配置
        verify_training_setup()
        
        print("\n📝 下一步:")
        print("1. 运行训练: python train.py")
        print("2. 运行验证: python val.py")
        print("3. 查看结果: 检查 runs/detect/ 目录")
        
    else:
        print("\n❌ 数据集设置失败！")
        print("请检查源数据集路径是否正确。")
    
    print("=" * 60)
