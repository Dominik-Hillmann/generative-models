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
from PIL import Image
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = func.leaky_relu(self.dense_1(x))
        x = func.leaky_relu(self.dense_2(x))
        
        return x


class DecoderDense(nn.Module):
    """Converts latent space representation to image."""

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


def travel_2d_latent_space(
    model: AutoencoderDense,
    latent_space_size: int,
    latent_idxs: Tuple[int, int],
    save_path: str,
    n_steps: Tuple[int, int] = (5, 5),
    step_size: float = 1.5,
    latent_x_start: float = -1.0,
    unused_latent_dim_val: float = 0.0
) -> None:
    latent_space = [unused_latent_dim_val] * latent_space_size
    fig, ax = plt.subplots(*n_steps, sharex = True, sharey = True)
    fig.tight_layout(pad = 1.0)
    latent_idx1, latent_idx2 = latent_idxs
    x1 = latent_x_start
    x2 = latent_x_start
    i = 0
    for x1_step in range(n_steps[0]):
        x2 = latent_x_start
        for x2_step in range(n_steps[1]):
            fig.add_subplot(*n_steps, i + 1)

            latent_space[latent_idx1] = x1
            latent_space[latent_idx2] = x2
            latent_X = torch.Tensor(np.array(latent_space))
            reconstruction = model.pred_from_latent_space(latent_X).cpu()
            
            ax[x1_step, x2_step].set_yticklabels([])
            ax[x1_step, x2_step].set_xticklabels([])
            ax[x1_step, x2_step].axis('off')
            plt.title(f'$(x_1={round(x1, 1)}, x_2={round(x2, 1)})$', fontdict = {'fontsize': 7})
            plt.imshow(reconstruction, cmap = 'Greys')
            plt.axis('off')

            i += 1
            x2 += step_size
        x1 += step_size
    # fig.suptitle('Reconstruction from latent space with values $(x_1, x_2)$', y = -0.01)
    plt.savefig(save_path)


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

    n_latent_dims = 2
    autoencoder = AutoencoderDense(n_latent_dims, device)
    train(autoencoder, train_loader, BATCH_SIZE, device, test_data = test_loader, n_epochs = 10, img_save_path = os.path.join(IMGS, 'losses-dense.png'))

    travel_2d_latent_space(autoencoder, n_latent_dims, (0, 1), os.path.join(IMGS, 'latent-space-travel.png'))


if __name__ == '__main__':
    main()