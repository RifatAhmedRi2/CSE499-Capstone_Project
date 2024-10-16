import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class AutoencoderModel(nn.Module):
    def __init__(self):
        super(AutoencoderModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(262144, 512),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 262144),  # Adjusted output size
            nn.Unflatten(1, (16, 128, 128)),  # Reverse Flatten operation
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1),  # Transpose Convolution
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # Transpose Convolution
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
autoencoder_model = AutoencoderModel()
autoencoder_model.load_state_dict(torch.load('autoencoder_model_2.pth'))

# Assuming you have your data loaded in PyTorch DataLoader
img_path = 'guatemala-volcano\images\guatemala-volcano_00000004_pre_disaster.png'
img = cv2.imread(img_path, 0)
img = np.float32(img)/255.
img = cv2.resize(img, (512, 512))
img = img.reshape((1, 1 , 512, 512)) #torch.unsqueeze(torch.tensor(img), 0)
img = torch.tensor(img)
print('img shape', img.shape)
autoencoder_model.eval()
with torch.no_grad():
    outputs = autoencoder_model(img)
print('output shape', outputs.shape)

img = outputs.numpy().reshape(512,512)*255
# img = img.resize(img , (64, 64))
cv2.imwrite('tmp3.jpg', img)