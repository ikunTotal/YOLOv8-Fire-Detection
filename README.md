# 🔥 YOLOv8 Fire Detection with iRMB and BiLevel Routing Attention

A lightweight and efficient real-time forest fire detection system based on YOLOv8, enhanced with **Inverted Residual Mobile Block (iRMB)** and **BiLevel Routing Attention** mechanisms for improved accuracy and computational efficiency.

## ✨ Key Features

- 🚀 **Real-time Detection**: Optimized for edge deployment with high FPS
- 🔧 **Lightweight Architecture**: iRMB blocks for efficient feature extraction
- 🎯 **Enhanced Attention**: BiLevel Routing Attention for better feature representation
- 📊 **Comprehensive Evaluation**: Built-in performance metrics and visualization tools
- 🛠️ **Easy Deployment**: Simple training and inference scripts

## 🏗️ Architecture Improvements

### Inverted Residual Mobile Block (iRMB)
- Replaces standard C2f blocks in backbone
- Cascaded design for better feature flow
- Reduced computational complexity while maintaining accuracy

### BiLevel Routing Attention
- Applied at the end of backbone (layer 10)
- Efficient attention mechanism for global context
- Parameters: `[8, 7]` for optimal performance

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Parameters** | 10.05MB |
| **FPS** | 29.2 (RTX 3090) |
| **mAP50** | 0.917 |
| **mAP50-95** | 0.637 |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ikunTotal/YOLOv8-Fire-Detection

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

```bash
# Python 3.8.16
# PyTorch 1.13.1+cu117
# TorchVision 0.14.1+cu117

pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0
```

### Training

```bash
# Quick start (recommended)
python quick_start.py

# Or direct training
python train.py

# Custom training parameters
python train.py --epochs 100 --batch 16 --imgsz 640 --device 0
```

### Validation & Testing

```bash
# Run validation with performance metrics
python val.py

# Custom validation
python val.py --data fire.yaml --imgsz 640 --batch 16
```

## 📁 Dataset

This project uses the **FireDetn** dataset:
- **Images**: 4,603 fire detection images
- **Classes**: 1 (fire)
- **Format**: YOLO format with bounding box annotations
- **Source**: [FireDetn Dataset](https://github.com/SuperXxts/FireDetn)

### Dataset Structure
```
datasets/firedetn/
├── images/
│   ├── 0000.jpg
│   ├── 0001.jpg
│   └── ...
├── labels/
│   ├── 0000.txt
│   ├── 0001.txt
│   └── ...
└── labels.cache
```

## 🔧 Model Configuration

The model uses a custom YOLOv8 configuration with the following key components:

```yaml
# Backbone with iRMB blocks
backbone:
  - [-1, 3, C2f_iRMB_Cascaded, [128, True]]  # P2/4
  - [-1, 6, C2f_iRMB_Cascaded, [256, True]]  # P3/8
  - [-1, 6, C2f_iRMB_Cascaded, [512, True]]  # P4/16
  - [-1, 3, C2f_iRMB_Cascaded, [1024, True]] # P5/32
  - [-1, 1, BiLevelRoutingAttention, [8, 7]] # Attention layer

# Standard YOLOv8 head
head:
  - [[16, 19, 22], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```

## 🔬 Technical Details

### Architecture Modifications
1. **iRMB Integration**: Replaced standard C2f blocks with Inverted Residual Mobile Blocks
2. **Attention Mechanism**: Added BiLevel Routing Attention at backbone output
3. **Cascaded Design**: Improved feature flow through cascaded iRMB blocks

### Training Configuration
- **Optimizer**: SGD
- **Learning Rate**: Adaptive with warmup
- **Batch Size**: 16
- **Image Size**: 640x640
- **Epochs**: 100
- **Mosaic Augmentation**: Disabled in last 10 epochs

## 📝 File Structure

```
yolov8fire/
├── datasets/                    # Dataset directory
│   └── firedetn/               # Fire detection dataset
├── ultralytics/                # YOLOv8 framework
│   └── cfg/models/             # Model configurations
├── runs/                       # Training results
├── fire.yaml                   # Dataset configuration
├── train.py                    # Training script
├── val.py                      # Validation script
├── setup_dataset.py            # Dataset setup script
├── quick_start.py              # Quick start script
├── plot_result.py              # Visualization tools
└── transform_PGI.py            # Model conversion
```



---

⭐ **Star this repository if you find it helpful!**
