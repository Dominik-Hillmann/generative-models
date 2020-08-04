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
from tqdm import tqdm

import matplotlib.pyplot as plt

# Typing
from typing import Callable, List

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
        x = self._flatten(x)
        x = func.relu(self.encode_1(x))
        x = func.relu(self.encode_2(x))
        x = func.relu(self.decode_1(x))
        x = self.decode_2(x)
        
        return x


def train(
    model: nn.Module, 
    train_data: DataLoader,
    device: str,
    test_data: DataLoader = None,
    n_epochs: int = 5, 
    lr: float = 0.001,
    n_batches_till_test: int = 10,
    save: bool = False
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.MSELoss() # Measures squared difference between each element of the prediction and ground truth.
    avg_test_losses = []
    avg_train_losses = []

    for epoch in range(n_epochs):
        print(f'Running epoch {epoch + 1} of {n_epochs}.')
        for batch_idx, (train_batch_X, _) in tqdm(enumerate(train_data)):
            batch_train_losses = []
            train_batch_X = train_batch_X.to(device)
            train_batch_X = train_batch_X.view(-1, 28 * 28)

            model.zero_grad()
            pred_X = model(train_batch_X)
            loss = loss_fn(pred_X, train_batch_X)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())

            if batch_idx % (BATCH_SIZE * n_batches_till_test) == 0:
                if test_data is not None:
                    test_loss = validate(model, test_data, loss_fn, device)
                    avg_test_losses.append(test_loss)

                avg_train_losses.append(np.mean(np.array(batch_train_losses)))                
                batch_train_losses.clear()
    
    if test_data is not None:
        draw_losses(avg_train_losses, n_batches_till_test, os.path.join('autoencoder', 'losses.png'), val_losses = avg_test_losses)
    else:
        draw_losses(avg_train_losses, n_batches_till_test, os.path.join('autoencoder', 'losses.png'), val_losses = avg_test_losses)

    if save:
        pass


def validate(model: nn.Module, test_data: DataLoader, loss_fn: Callable, device: str) -> float:
    test_losses = []
    with torch.no_grad():
        for test_batch_X, _ in test_data:
            test_batch_X = test_batch_X.to(device)
            test_batch_X = test_batch_X.view(-1, 28 * 28)

            pred_X = model(test_batch_X)
            loss = loss_fn(pred_X, test_batch_X)
            test_losses.append(loss.item())

    return np.mean(np.array(test_losses))


def draw_losses(train_losses: List[float], interval_size: int, save_path: str, val_losses: List[float] = None) -> None:
        assert len(train_losses) == len(val_losses)
        batches = [0]
        while len(batches) != len(train_losses):
            batches.append(batches[len(batches) - 1] + interval_size)
                
        plt.plot(batches, train_losses, 'r-', label = 'Training loss')
        plt.plot(batches, val_losses, 'b-', label = 'Validation loss')
        plt.title(f'Average MSE of the last {interval_size} batches of {BATCH_SIZE} images')
        plt.xlabel('Batch')
        plt.ylabel('Loss (MSE)')
        plt.legend(loc = 'upper right')
        plt.savefig(save_path)


def get_device() -> str:
    if torch.cuda.is_available():
        print('Using GPU...')
        device = torch.device('cuda:0')
    else:
        print('Using CPU...')
        device = torch.device('cpu')

    print()
    return device


def reconstruct_from_flat_img(flat_img: torch.Tensor):
    pass


def main():
    device = get_device()

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081]) # Over all of the dataset
    ])
    train_data = torchvision.datasets.MNIST(os.path.join('.', 'data'), train = True, download = True, transform = normalize)
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE)

    test_data = torchvision.datasets.MNIST(os.path.join('.', 'data'), train = False, download = True, transform = normalize)
    test_loader = DataLoader(test_data, batch_size = int(len(test_data) / 2))

    model = AutoencoderDense(16, device)
    train(model, train_loader, device, test_data = test_loader)


if __name__ == '__main__':
    main()