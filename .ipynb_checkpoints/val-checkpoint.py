import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./runs/detect/train10/weights/best.pt')
    model.val(data='/sjxy004/datasets/fire/fire.yaml',
              split='test',
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='',
              name='',
              )
