from ultralytics import YOLO

# Load a model
model = YOLO(r'E:\xview_base\runs\detect\train7\weights\best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
# results = model([r'dataset\test\all_images\guatemala-volcano_00000009_pre_disaster.png'])  # return a list of Results objects

results = model.predict(r'dataset\train\images\hurricane-matthew_00000215_post_disaster.png', save=False, conf=0.1, imgsz=1024, boxes=True)
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs
    print(probs)
    print("*"*100)
    print("Printing Boxes -------------------------------------->")
    print(boxes.cls)
    print(boxes.conf)
    print(boxes.xywhn)
    print("*"*100)
