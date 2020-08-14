"""This script trains a very simple autoencoder only using dense layers and outputs
an image of the outputs of various latent space inputs."""

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


class EncoderDense(nn.Module):
    """Converts the img to its latent space representation."""

    def __init__(self, n_latent_dims, device: str):
        super(EncoderDense, self).__init__()
        self.device = device

        self.dense_1 = nn.Linear(28 * 28, 256)
        self.dense_2 = nn.Linear(256, n_latent_dims)
        

    def forward(self, batch_X: torch.Tensor) -> torch.Tensor:
        batch_X = func.leaky_relu(self.dense_1(batch_X))
        batch_X = func.leaky_relu(self.dense_2(batch_X))
        
        return batch_X


class DecoderDense(nn.Module):
    """Converts latent space representation to image."""

    def __init__(self, n_latent_dims: int, device: str):
        super(DecoderDense, self).__init__()
        self.device = device

        self.dense_1 = nn.Linear(n_latent_dims, 256)
        self.dense_2 = nn.Linear(256, 28 * 28)


    def forward(self, batch_X: torch.Tensor) -> torch.Tensor:
        batch_X = func.leaky_relu(self.dense_1(batch_X))
        batch_X = self.dense_2(batch_X)

        return batch_X


class AutoencoderDense(nn.Module):
    """An autoencoder that only uses fully connected layers."""

    def __init__(self, n_latent_dims: int, device: str):
        super(AutoencoderDense, self).__init__()
        self.device = device
        
        self.encoder = EncoderDense(n_latent_dims, device)
        self.decoder = DecoderDense(n_latent_dims, device)


    def _flatten(self, batch_X: torch.Tensor) -> torch.Tensor:
        return batch_X.view(-1, 28 * 28)


    def forward(self, batch_X: torch.Tensor) -> torch.Tensor:
        batch_X = self._flatten(batch_X)
        batch_X = self.encoder(batch_X)
        batch_X = self.decoder(batch_X)
        
        return batch_X


    def get_decoder(self) -> nn.Module:
        return list(self.children())[1]

    
    def pred_from_latent_space(self, batch_Z: torch.Tensor) -> torch.Tensor:
        batch_Z = batch_Z.to(self.device)
        with torch.no_grad():
            pred_X_reconstruction = self.decoder(batch_Z).view(28, 28)
        
        return pred_X_reconstruction


def main() -> None:
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
    autoencoder = AutoencoderDense(n_latent_dims, device)
    summary(autoencoder.to(device), input_size = (BATCH_SIZE, 28, 28))
    train(autoencoder, train_loader, BATCH_SIZE, device, test_data = test_loader, n_epochs = 10, img_save_path = os.path.join(IMGS, 'losses-dense.png'))

    travel_2d_latent_space(autoencoder, n_latent_dims, (0, 1), os.path.join(IMGS, 'latent-space-travel-dense.png'), latent_z_start = -1.0, step_size = 0.5)


if __name__ == '__main__':
    main()