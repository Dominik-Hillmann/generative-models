# Python libraries
import os
import math
# Internal imports
from utils import train, get_device, save_from_flat_tensor
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
# Typing
from typing import Callable, List
# Constants
BATCH_SIZE = 32
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

from PIL import Image


class EncoderDense(nn.Module):
    def __init__(self, n_latent_dims, device: str):
        super(EncoderDense, self).__init__()
        self.device = device

        self.dense_1 = nn.Linear(28 * 28, 256)
        self.dense_2 = nn.Linear(256, n_latent_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = func.leaky_relu(self.dense_1(x))
        x = func.leaky_relu(self.dense_2(x))
        
        return x


class DecoderDense(nn.Module):
    def __init__(self, n_latent_dims: int, device: str):
        super(DecoderDense, self).__init__()
        self.device = device

        self.dense_1 = nn.Linear(n_latent_dims, 256)
        self.dense_2 = nn.Linear(256, 28 * 28)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = func.leaky_relu(self.dense_1(x))
        x = self.dense_2(x)

        return x


class AutoencoderDense(nn.Module):
    """An autoencoder that only uses fully connected layers."""

    def __init__(self, n_latent_dims: int, device: str):
        super(AutoencoderDense, self).__init__()
        self.device = device
        
        self.encoder = EncoderDense(n_latent_dims, device)
        self.decoder = DecoderDense(n_latent_dims, device)


    def _flatten(self, x: torch.Tensor):
        return x.view(-1, 28 * 28)


    def forward(self, x: torch.Tensor):
        x = self._flatten(x)
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x

    def get_decoder(self) -> nn.Module:
        return list(self.children())[1]

    
    def pred_from_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        with torch.no_grad():
            pred_reconstruction = self.decoder(x).view(28, 28)
        
        return pred_reconstruction


def main():
    device = get_device()

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([MNIST_MEAN], [MNIST_STD]) # Over all of the dataset
    ])
    train_data = torchvision.datasets.MNIST(os.path.join('.', 'data'), train = True, download = True, transform = normalize)
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE)

    test_data = torchvision.datasets.MNIST(os.path.join('.', 'data'), train = False, download = True, transform = normalize)
    test_loader = DataLoader(test_data, batch_size = int(len(test_data) / 2))

    n_latent_dims = 64
    autoencoder = AutoencoderDense(n_latent_dims, device)
    train(autoencoder, train_loader, BATCH_SIZE, device, test_data = test_loader, n_epochs = 1)

    latent_X = torch.Tensor(np.array([.5] * n_latent_dims))
    reconstruction = autoencoder.pred_from_latent_space(latent_X)
    save_from_flat_tensor(reconstruction, 'latent-space-trial.png')

    latent_space = [0.0] * n_latent_dims
    # fig = plt.figure()#figsize = (5, 5))
    fig, ax = plt.subplots(5, 5)
    # plt.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    latent_idx1 = 5
    latent_idx2 = 28
    x1 = -0.5
    x2 = -0.5
    i = 0
    for x1_step in range(5):
        for x2_step in range(5):
            fig.add_subplot(5, 5, i + 1)

            latent_space[latent_idx1] = x1
            latent_space[latent_idx2] = x2
            reconstruction = autoencoder.pred_from_latent_space(torch.Tensor(np.array(latent_space))).cpu()
            plt.imshow(reconstruction, cmap = 'Greys')

            i += 1
            x2 += 0.2
        x1 += 0.2
    plt.savefig('trial.png')

    # columns = 4
    # rows = 5
    # for i in range(1, columns*rows +1):
    #     img = 
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(img)
    # plt.show()


if __name__ == '__main__':
    main()