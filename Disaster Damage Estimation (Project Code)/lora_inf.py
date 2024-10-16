import torch
import numpy as np
import transformers
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoImageProcessor
from transformers import AutoModelForSemanticSegmentation, TrainingArguments

from peft import PeftConfig, PeftModel

checkpoint = "nvidia/mit-b0"

id2label = {0: "background", 1: "building"}
label2id = {label: idx for idx, label in id2label.items()}
config = PeftConfig.from_pretrained('mit-b0-building-damage-lora')
model = AutoModelForSemanticSegmentation.from_pretrained(
    checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

inference_model = PeftModel.from_pretrained(model, 'mit-b0-building-damage-lora')

image = Image.open("hurricane-matthew_00000273_post_disaster.png")

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

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(image)
ax1.set_title('Image')
ax2.imshow(pred_seg)
ax2.set_title('Segmentation')
plt.show()

color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)

color_seg[pred_seg == 1, :] = np.array([255, 0, 0])
color_seg = color_seg[..., ::-1]  # convert to BGR

img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()