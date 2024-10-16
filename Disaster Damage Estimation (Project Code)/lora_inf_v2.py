import torch
import time
import numpy as np
import transformers
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoImageProcessor
from transformers import AutoModelForSemanticSegmentation, TrainingArguments

from peft import PeftConfig, PeftModel

from ultralytics import YOLO


img_path = r'hurricane-florence_00000173_post_disaster.png'
# Load a model
yolo_model = YOLO(r'D:\AI Models\SAT PROJECT (YOLO Cls)\best.pt')  # pretrained YOLOv8n model
result = yolo_model.predict(img_path, save=False, conf=0.1, imgsz=1024, boxes=True)[0]
boxes = result.boxes  # Boxes object for bbox outputs
classes  = boxes.cls
print(boxes.conf)
cords = boxes.xywh


# Red: (255, 0, 0)
# Green: (0, 255, 0)
# Blue: (0, 0, 255)
# Yellow: (255, 255, 0)
# Purple: (128, 0, 128)

checkpoint = "nvidia/mit-b0"

id2label = {0: "background", 1: "building"}
label2id = {label: idx for idx, label in id2label.items()}
config = PeftConfig.from_pretrained('mit-b0-building-damage-lora')
model = AutoModelForSemanticSegmentation.from_pretrained(
    checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

inference_model = PeftModel.from_pretrained(model, 'mit-b0-building-damage-lora')

image = Image.open(img_path)

image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=False)
encoding = image_processor(image.convert("RGB"), return_tensors="pt")

with torch.no_grad():
    outputs = inference_model(pixel_values=encoding.pixel_values)
    logits = outputs.logits

upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]

for i,cord in enumerate(cords):
    print(cord)
    for x in range(int(cord[0]), int(cord[0]) + int(cord[2])):
        if int(cord[0]) + int(cord[2]) >= 1024:
            continue 
        for y in range(int(cord[1]), int(cord[1]) + int(cord[3])):
            if int(cord[1]) + int(cord[3]) >=1024:
                continue
            if pred_seg[x][y] == 1:
                pred_seg[x][y] = classes[i]


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.imshow(image)
# ax1.set_title('Image')
# ax2.imshow(pred_seg)
# ax2.set_title('Segmentation')
# plt.show()

color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)


# color_seg[pred_seg == 1, :] = np.array([255, 0, 0])

for i in range(1024):
    for j in range(1024):
        if pred_seg[i][j] == 0:
            color_seg[i][j] = [255, 0, 0]
        elif pred_seg[i][j] == 1:
            color_seg[i][j] = [0, 255, 0]
        elif pred_seg[i][j] == 2:
            color_seg[i][j] = [0, 0, 255]
        elif pred_seg[i][j] == 3:
            color_seg[i][j] = [255, 255, 0]
        elif pred_seg[i][j] == 4:
            color_seg[i][j] = [128, 0, 128]
color_seg = color_seg[..., ::-1]  # convert to BGR


img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
img = img.astype(np.uint8)
print(img)

# plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()