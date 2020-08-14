# Python libraries
import os
import math
# Internal imports
from utils import train, get_device, save_from_flat_tensor, travel_2d_latent_space, save_from_tensor, draw_losses
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
from typing import Callable, List, Tuple, Union
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
        noise = (torch.randn(batch_X.shape) * self.std) + self.mean
        noise = noise.type(torch.int16)
        batch_X_noisy = batch_X + noise
        # No values above 255 or below 0 wanted
        batch_X_noisy = torch.where(batch_X_noisy > 255, torch.ones(batch_X_noisy.shape) * 255, batch_X_noisy)
        batch_X_noisy = torch.where(batch_X_noisy < 0, torch.zeros(batch_X_noisy.shape), batch_X_noisy)

        return batch_X_noisy


    def __repr__(self):
        return self.__class__.__name__ + f'(mean = {self.mean}, std = {self.std})'


class NoisyXDataset(torch.utils.data.Dataset):
    """A dataset that returns a noisy version of the data as input data and a clean
    version as the labels."""

    def __init__(self, X: torch.Tensor, clean_transforms: List[Callable] = None, noisy_transforms: List[Callable] = None):
        X = X.type(torch.float32)

        self.clean_X = X.clone()
        if clean_transforms is not None:
            for transform_fn in clean_transforms:
                self.clean_X = transform_fn(self.clean_X)

        self.noisy_X = X.clone()
        if noisy_transforms is not None:
            for transform_fn in noisy_transforms:
                self.noisy_X = transform_fn(self.noisy_X)


    def __len__(self) -> int:
        return len(self.clean_X)


    def __getitem__(self, idx: Union[torch.Tensor, int]) -> torch.Tensor:
        batch_X = self.noisy_X[idx]
        batch_Y = self.clean_X[idx]

        return batch_X, batch_Y


def train_denoiser(
    model: nn.Module, 
    train_data: DataLoader,
    batch_size: int, 
    device: str,
    test_data: DataLoader = None,
    n_epochs: int = 5, 
    lr: float = 0.001,
    n_batches_till_test: int = 10,
    save: bool = False,
    img_save_path: str = None, 
    flatten: bool = True
) -> None:
    """This function trains the model on the ```train_data``` and validates it
    on ```test_data```, if it is given.

    Args
    ----
        model (nn.Module): The model that will be trained.
        train_data (DataLoader): The data the model will be trained on.
        batch_size (int): The mini-batch size.
        device (str): The device PyTorch uses.
        test_data (DataLoader, optional): The validation data. Defaults to None.
        n_epochs (int, optional): Number of epochs. Defaults to 5.
        lr (float, optional): The learning rate alpha. Defaults to 0.001.
        n_batches_till_test (int, optional): The number of batches that will be trained until the validation set is run. Defaults to 10.
        save (bool, optional): Whether the model will be serialized after training. Defaults to False.
    """

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.MSELoss() # Measures squared difference between each element of the prediction and ground truth.
    avg_test_losses = []
    avg_train_losses = []

    for epoch in range(n_epochs):
        print(f'Running epoch {epoch + 1} of {n_epochs}.')
        for batch_idx, (train_batch_X, train_batch_Y) in tqdm(enumerate(train_data)):
            batch_train_losses = []

            train_batch_X = train_batch_X.to(device).view(-1, 1, 28, 28)
            train_batch_Y = train_batch_Y.to(device).view(-1, 1, 28, 28)
            if flatten:
                train_batch_X = train_batch_X.view(-1, 28 * 28)

            model.zero_grad()
            pred_Y = model(train_batch_X)
            loss = loss_fn(pred_Y, train_batch_Y)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())

            if batch_idx % (batch_size * n_batches_till_test) == 0:
                if test_data is not None:
                    test_loss = validate_denoiser(model, test_data, loss_fn, device, flatten = flatten)
                    avg_test_losses.append(test_loss)

                avg_train_losses.append(np.mean(np.array(batch_train_losses)))                
                batch_train_losses.clear()
    
    if test_data is not None:
        draw_losses(avg_train_losses, n_batches_till_test, img_save_path, val_losses = avg_test_losses)
    else:
        draw_losses(avg_train_losses, n_batches_till_test, img_save_path, val_losses = avg_test_losses)

    if save:
        pass


def validate_denoiser(model: nn.Module, test_data: DataLoader, loss_fn: Callable, device: str, flatten: bool = True) -> float:
    """Validates ```model``` on ```test_data```.

    Args
    ----
        model (nn.Module): The model to be validated.
        test_data (DataLoader): The data with which the model will be validated.
        loss_fn (Callable): The loss function used for training.
        device (str): The device PyTorch currently uses.

    Returns
    -------
        float: The loss over all of the test_data.
    """

    test_losses = []
    with torch.no_grad():
        for test_batch_X, test_batch_Y in test_data:
            test_batch_X = test_batch_X.to(device).view(-1, 1, 28, 28)
            test_batch_Y = test_batch_Y.to(device).view(-1, 1, 28, 28)

            if flatten:
                test_batch_X = test_batch_X.view(-1, 28 * 28)

            pred_Y = model(test_batch_X)
            loss = loss_fn(pred_Y, test_batch_Y)
            test_losses.append(loss.item())

    return np.mean(np.array(test_losses))



def main() -> None:
    device = get_device()

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([MNIST_MEAN], [MNIST_STD]) # Over all of the dataset
    ])
    train_data = torchvision.datasets.MNIST(os.path.join('.', 'data'), train = True, download = True)#, transform = normalize)
    # train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)

    # test_loader = DataLoader(test_data, batch_size = int(len(test_data) / 2), shuffle = True)

    # print(type(train_data.data))
    noisy_train_data = NoisyXDataset(
        train_data.data,
        clean_transforms = [
            transforms.Normalize([MNIST_MEAN], [MNIST_STD])
        ],
        noisy_transforms = [
            # GaussianNoise(0, 5),
            transforms.Normalize([MNIST_MEAN], [MNIST_STD])
        ]
    )
    noisy_train_data_loader = DataLoader(noisy_train_data, batch_size = BATCH_SIZE, shuffle = True)
    
    test_data = torchvision.datasets.MNIST(os.path.join('.', 'data'), train = False, download = True)
    noisy_test_data = NoisyXDataset(
        test_data.data,
        clean_transforms = [
            transforms.Normalize([MNIST_MEAN], [MNIST_STD])
        ],
        noisy_transforms = [
            GaussianNoise(0, 20),
            transforms.Normalize([MNIST_MEAN], [MNIST_STD])
        ]
    )
    noisy_test_data_loader = DataLoader(noisy_test_data, batch_size = BATCH_SIZE, shuffle = True)

    n_latent_dims = 256
    autoencoder = AutoencoderConv(n_latent_dims, device)
    summary(autoencoder.to(device), input_size = (1, 28, 28))
    train_denoiser(
        autoencoder, 
        noisy_train_data_loader, 
        BATCH_SIZE, 
        device, 
        test_data = noisy_test_data_loader,
        n_epochs = 1, 
        img_save_path = os.path.join(IMGS, 'losses-conv-test.png'), 
        flatten = False
    )

    save_denoising_img(autoencoder, noisy_test_data_loader, '', device)
    # travel_2d_latent_space(autoencoder, n_latent_dims, (0, 1), os.path.join(IMGS, 'latent-space-travel-conv.png'), latent_z_start = 0.0, step_size = 1.0)
def save_denoising_img(model: nn.Module, data: DataLoader, save_path: str, device: str, n: int = 10):
    for batch_X, batch_Y in data:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        # print(batch_X[0])
        # print(model(batch_X[0].view(-1, 1, 28, 28)))
        # print(batch_Y[0])
        save_from_tensor(batch_X[1], os.path.join('autoencoder', 'img', 'noisy_input.png'))
        save_from_tensor(model(batch_X[1].view(-1, 1, 28, 28)), os.path.join('autoencoder', 'img', 'pred_Y.png'))
        save_from_tensor(batch_Y[1], os.path.join('autoencoder', 'img', 'target_Y.png'))

        break

if __name__ == '__main__':
    main()