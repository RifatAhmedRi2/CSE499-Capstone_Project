import cv2
import os
import numpy as np
import torch
import torchvision as tv
from PIL import Image
from torchvision import transforms 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from matplotlib import pyplot as plt


class CustomDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = os.listdir(os.path.join(root_dir, 'images'))
        self.image_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')


    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_name = self.images[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
        width, height = image.size
        # image = cv2.imread(os.path.join(self.image_dir, image_name), cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image = cv2.resize(image, (1024, 1024))
        # print(image.shape)
        # image = image / 255.0
        with open(os.path.join(self.labels_dir, image_name[:-4] + '.txt')) as f:
            lines = f.readlines()
        boxes, targets = [], []
        for line in lines:
            target, x, y, w, h = line.split()
            targets.append(int(target) + 1)
            boxes.append([(float(x) - float(w)/2)*width, (float(y) - float(h)/2)*height, (float(x) + float(w)/2)*width , (float(y) + float(h)/2)*height])

        if not boxes:
            return None, None
        target = {}
        target['boxes'] = torch.tensor(boxes)
        target['labels'] = torch.tensor(targets)
        return transforms.ToTensor()(image), target
    
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def custom_collate(data):
    return tuple(zip(*data))


custom_dataset_train_path, custom_dataset_val_path, batch_size = r'selected\train', r'selected\val', 1
custom_dataset_train = CustomDataset(root_dir=custom_dataset_train_path)
custom_dataset_val = CustomDataset(root_dir=custom_dataset_val_path)
data_loader_train = DataLoader(custom_dataset_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, pin_memory=True if torch.cuda.is_available else False)
data_loader_val = DataLoader(custom_dataset_val, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, pin_memory=True if torch.cuda.is_available else False)


model = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
num_classes = 6
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
device = 'cuda' if torch.cuda.is_available else 'cpu'
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# lr_scheduler = None

num_epochs = 150
loss_hist = Averager()
itr = 1

for epoch in range(num_epochs):
    loss_hist.reset()
    for images, target in data_loader_train:
        images = list(image.to(device) for image in images)
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        loss_dict = model(images, target)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1
    
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")

torch.save(model.state_dict(), 'test.pb')