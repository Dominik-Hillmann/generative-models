# Python libraries
import os
import math
# Internal imports
from utils import train, get_device, save_from_flat_tensor, travel_2d_latent_space
# External imports
import numpy as np
import torchvision
from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary
# Typing
from typing import Callable, List, Tuple
# Constants
BATCH_SIZE = 32
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
SEED = 69
IMGS = os.path.join('autoencoder', 'img')
torch.manual_seed(SEED)


class EncoderConv(nn.Module):
    """Converts the img to its latent space representation."""

    def __init__(self, n_latent_dims, device: str):
        super(EncoderConv, self).__init__()
        self.device = device

        self.conv_1 = nn.Conv2d(1, 8, (3, 3), padding = 1)
        self.max_pool_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(8, 32, (3, 3), padding = 1)
        self.max_pool_2 = nn.MaxPool2d((2, 2))
        self.dense_1 = nn.Linear(1568, 256)
        self.dense_2 = nn.Linear(256, 64)
        self.dense_3 = nn.Linear(64, n_latent_dims)

    
    def _flatten(self, batch_X: torch.Tensor) -> torch.Tensor:
        return batch_X.view(-1, 32 * 7 * 7)


    def forward(self, batch_X: torch.Tensor) -> torch.Tensor:
        batch_X = func.leaky_relu(self.conv_1(batch_X))
        batch_X = self.max_pool_1(batch_X)
        batch_X = func.leaky_relu(self.conv_2(batch_X))
        batch_X = self.max_pool_2(batch_X)
        batch_X = self._flatten(batch_X)
        batch_X = func.leaky_relu(self.dense_1(batch_X))
        batch_X = func.leaky_relu(self.dense_2(batch_X))
        batch_X = func.leaky_relu(self.dense_3(batch_X))
        
        return batch_X


class DecoderConv(nn.Module):
    """Converts latent space representation to image."""

    def __init__(self, n_latent_dims: int, device: str):
        super(DecoderConv, self).__init__()
        self.device = device

        self.dense_1 = nn.Linear(n_latent_dims, 64)
        self.dense_2 = nn.Linear(64, 256)
        self.dense_3 = nn.Linear(256, 1568)
        self.deconv_1 = nn.ConvTranspose2d(32, 8, 2, stride = 2)
        self.deconv_2 = nn.ConvTranspose2d(8, 1, 2, stride = 2)


    def _reconstruct_from_1d(self, batch_X: torch.Tensor) -> torch.Tensor:
        return batch_X.view(-1, 32, 7, 7)


    def forward(self, batch_X: torch.Tensor) -> torch.Tensor:
        batch_X = func.leaky_relu(self.dense_1(batch_X))
        batch_X = func.leaky_relu(self.dense_2(batch_X))
        batch_X = func.leaky_relu(self.dense_3(batch_X))
        batch_X = self._reconstruct_from_1d(batch_X)
        batch_X = self.deconv_1(batch_X)
        batch_X = self.deconv_2(batch_X)

        return batch_X


class AutoencoderConv(nn.Module):
    """An autoencoder that only uses fully connected layers."""

    def __init__(self, n_latent_dims: int, device: str):
        super(AutoencoderConv, self).__init__()
        self.device = device
        
        self.encoder = EncoderConv(n_latent_dims, device)
        self.decoder = DecoderConv(n_latent_dims, device)


    def forward(self, batch_X: torch.Tensor):
        batch_X = self.encoder(batch_X)
        batch_X = self.decoder(batch_X)
        
        return batch_X


    def get_decoder(self) -> nn.Module:
        return list(self.children())[1]

    
    def pred_from_latent_space(self, batch_Z: torch.Tensor) -> torch.Tensor:
        batch_Z = batch_Z.to(self.device)
        with torch.no_grad():
            pred_reconstruction = self.decoder(batch_Z).view(28, 28)
        
        return pred_reconstruction


class GaussianNoise(object):
    """Adds Gaussian noise to the image. Values have to be integers, for pixel values
    range from integer 0 to integer 255."""

    def __init__(self, mean: int, std: int):
        self.std = std
        self.mean = mean
        

    def __call__(self, batch_X: torch.Tensor) -> torch.Tensor:
        noise = (torch.randn(tensor.size()) * self.std) + self.mean
        noise = noise.type(torch.int16)
        batch_X_noisy = batch_X + noise
        # No values above 255 or below 0 wanted
        batch_X_noisy = torch.where(batch_X_noisy > 255, torch.ones(batch_X_noisy.shape) * 255, batch_X_noisy)
        batch_X_noisy = torch.where(batch_X_noisy < 0, torch.zeros(batch_X_noisy.shape), batch_X_noisy)

        return batch_X_noisy


    def __repr__(self):
        return self.__class__.__name__ + f'(mean = {self.mean}, std = {self.std})'


def main():
    device = get_device()

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([MNIST_MEAN], [MNIST_STD]) # Over all of the dataset
    ])
    train_data = torchvision.datasets.MNIST(os.path.join('.', 'data'), train = True, download = True, transform = normalize)
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)

    test_data = torchvision.datasets.MNIST(os.path.join('.', 'data'), train = False, download = True, transform = normalize)
    test_loader = DataLoader(test_data, batch_size = int(len(test_data) / 2), shuffle = True)

    n_latent_dims = 2
    autoencoder = AutoencoderConv(n_latent_dims, device)
    summary(autoencoder.to(device), input_size = (1, 28, 28))
    train(
        autoencoder, 
        train_loader, 
        BATCH_SIZE, 
        device, 
        test_data = test_loader,
        n_epochs = 10, 
        img_save_path = os.path.join(IMGS, 'losses-conv.png'), 
        flatten = False
    )

    travel_2d_latent_space(autoencoder, n_latent_dims, (0, 1), os.path.join(IMGS, 'latent-space-travel-conv.png'), latent_z_start = 0.0, step_size = 1.0)


if __name__ == '__main__':
    main()