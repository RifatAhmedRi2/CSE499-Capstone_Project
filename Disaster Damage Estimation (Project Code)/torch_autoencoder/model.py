import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.cuda.empty_cache()

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = os.listdir(os.path.join(root_dir, 'images'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.images[idx])
        img = cv2.imread(img_name, 0)
        img = np.float32(img)/255.
        img = cv2.resize(img, (512, 512))
        label_name = os.path.join(self.root_dir, 'labels', self.images[idx])
        label = cv2.imread(label_name, 0)
        label = np.float32(label)/255.
        label = cv2.resize(label, (512, 512))
        label = label.reshape((1, 512,512))
        img = img.reshape((1, 512, 512))


        return img, label

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

# Define dataset paths and batch size
custom_dataset_train_path = 'spacenet_gt'
custom_dataset_val_path = 'guatemala-volcano'
batch_size = 4

# Create datasets and data loaders
custom_dataset_train = CustomDataset(root_dir=custom_dataset_train_path)
# custom_dataset_val = CustomDataset(root_dir=custom_dataset_val_path)
data_loader_train = DataLoader(custom_dataset_train, batch_size=batch_size, shuffle=True)
# data_loader_val = DataLoader(custom_dataset_val, batch_size=batch_size)


device = 'cuda'
# Initialize the model
autoencoder_model = AutoencoderModel()
autoencoder_model.to(device)

# Initialize optimizer
optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=1e-3)



# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    autoencoder_model.train()
    for i,batch in enumerate(data_loader_train):
        optimizer.zero_grad()
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
        loss = autoencoder_model.training_step((inputs, targets))
        loss.backward()
        optimizer.step()
        print(f'\rEpoch: {epoch+1} | Iteration: {i+1}/{len(data_loader_train)} | Loss: {loss.item():.6f}', end='')

    # autoencoder_model.eval()
    # val_losses = []
    # with torch.no_grad():
    #     for batch in data_loader_val:
    #         inputs, targets = batch
    #         inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
    #         val_loss = autoencoder_model.validation_step((inputs, targets))
    #         val_losses.append(val_loss)
    # avg_val_loss = torch.stack(val_losses).mean()
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {avg_val_loss.item():.4f}')

# Save the model
torch.save(autoencoder_model.state_dict(), 'autoencoder_model_2.pth')