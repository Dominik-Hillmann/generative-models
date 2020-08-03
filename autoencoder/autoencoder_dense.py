# Python libraries
import os
import math
# External
import numpy as np
import torchvision
from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
# from torch.util.data import TensorDataset
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 32
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

from PIL import Image

import matplotlib.pyplot as plt


class AutoencoderDense(nn.Module):
    """An autoencoder that only uses fully connected layers."""

    def __init__(self, num_latent_dims: int, device: str):
        super(AutoencoderDense, self).__init__()
        self.device = device
        
        self.encode_1 = nn.Linear(28 * 28, 256)
        self.encode_2 = nn.Linear(256, num_latent_dims)
        self.decode_1 = nn.Linear(num_latent_dims, 256)
        self.decode_2 = nn.Linear(256, 28 * 28)


    def _flatten(self, x: torch.Tensor):
        return x.view(-1, 28 * 28)


    def forward(self, x: torch.Tensor):
        x.to(self.device)
        x = self._flatten(x)
        x = func.relu(self.encode_1(x))
        x = func.relu(self.encode_2(x))
        x = func.relu(self.decode_1(x))
        x = self.decode_2(x)
        
        return x

def train(model: nn.Module, data: DataLoader, lr: float, device: str):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.MSELoss() # Measures squared difference between each element of the prediction and input.
    
    for epoch in range(num_epochs):
        for idx, (batch_X, _) in tqdm(enumerate(data)):
            # print(f'Berechne batch der Elemente {idx} bis {idx + BATCH_SIZE}.')
            # print(batch_X.shape)
            batch_X = batch_X.to(device)
            batch_X = batch_X.view(-1, 28 * 28)

            model.zero_grad()
            pred_X = model(batch_X)
            # print(pred_X.shape)
            loss = loss_fn(pred_X, batch_X)
            loss.backward()
            optimizer.step()

            # print(loss.item())
            # print()

def get_device():
    if torch.cuda.is_available():
        print('Using GPU...')
        device = torch.device('cuda:0')
    else:
        print('Using CPU...')
        device = torch.device('cpu')

    return device


def main():
    device = get_device()

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081]) # Over all of the dataset
    ])
    train_data = torchvision.datasets.MNIST(os.path.join('.', 'data'), train = True, download = True, transform = img_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE)
    # test = torchvision.datasets.MNIST(os.path.join('.', 'data'), train = False, download = True, transform = img_transforms)
    # zu Dataset
    # zu DataLoader
    model = AutoencoderDense(16, device)
    train(model, train_loader, 0.0001, device)
    
    


if __name__ == '__main__':
    main()