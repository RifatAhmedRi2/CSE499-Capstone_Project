from ultralytics import YOLO
import torch

import gc
# Load a model
model = YOLO('yolov8x-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8x-seg.yaml').load('yolov8x.pt')  # build from YAML and transfer weights

if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
# Train the model
    results = model.train(data=r'E:\xview_base\data.yml', epochs=100, imgsz=768, device='cuda', batch=1)