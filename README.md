# YOLOv8 Fire Detection (iRMB + BiLevel Routing Attention)

A lightweight and efficient fire detection model built on Ultralytics YOLOv8, enhanced with Inverted Residual Mobile Blocks (iRMB) and BiLevel Routing Attention.

This README explains exactly how to set up the dataset paths (YAML), train, and evaluate so that anyone can reproduce results after cloning the repo.
## Quickstart

1. Put your dataset at: `/root/yolov8fire/datasets/firedetn` (or edit the three absolute lines in `fire.yaml`).
2. Generate split lists (train/val/test) using Ultralytics autosplit:

```bash
python -m dataset.split_data
```

3. Train:

```bash
python train.py
```

4. Evaluate (uses validation split, loads latest best.pt):

```bash
python val.py
```

---


## Install
```bash
pip install -r requirements.txt
```

## Dataset
- Option 1: Use the pre-split dataset at `/root/yolov8fire/datasets/firedetn`.
- Option 2: Download and split yourself:
  - Download: https://drive.google.com/file/d/11YwVAph_-b8Ew25zM-MYGQO-TqIbu-zE/view?usp=drive_link
  - Extract to `./datasets/firedetn`
  - Generate splits:
```bash
python -m dataset.split_data
```

## Configure (fire.yaml)
```yaml
train: /root/yolov8fire/datasets/firedetn/autosplit_train.txt
val:   /root/yolov8fire/datasets/firedetn/autosplit_val.txt
test:  /root/yolov8fire/datasets/firedetn/autosplit_test.txt
```

## Train
```bash
python train.py
```

## Evaluate
```bash
python val.py
```

## Outputs
- Weights: `runs/detect/trainX/weights/best.pt`, `runs/detect/trainX/weights/last.pt`
- Logs/curves/metrics saved under `runs/detect/trainX/`

