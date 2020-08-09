"""This file contains all utility function common across all autoencoders."""

# Python libraries
import os
# External imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
# Typing
from typing import Callable, List
# Constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def train(
    model: nn.Module, 
    train_data: DataLoader,
    batch_size: int, 
    device: str,
    test_data: DataLoader = None,
    n_epochs: int = 5, 
    lr: float = 0.001,
    n_batches_till_test: int = 10,
    save: bool = False
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

            if batch_idx % (batch_size * n_batches_till_test) == 0:
                if test_data is not None:
                    test_loss = validate(model, test_data, loss_fn, device)
                    avg_test_losses.append(test_loss)

                avg_train_losses.append(np.mean(np.array(batch_train_losses)))                
                batch_train_losses.clear()
    
    if test_data is not None:
        draw_losses(avg_train_losses, n_batches_till_test, os.path.join('autoencoder', 'losses-dense.png'), val_losses = avg_test_losses)
    else:
        draw_losses(avg_train_losses, n_batches_till_test, os.path.join('autoencoder', 'losses-dense.png'), val_losses = avg_test_losses)

    with torch.no_grad():
        for test_batch_X, _ in test_data:
            test_batch_X = test_batch_X.to(device)
            test_batch_X = test_batch_X.view(-1, 28 * 28)
            pred_X = model(test_batch_X)
            save_from_flat_tensor(test_batch_X[28], 'ground.png')
            save_from_flat_tensor(pred_X[28], 'pred.png')

            break


    if save:
        pass


def validate(model: nn.Module, test_data: DataLoader, loss_fn: Callable, device: str) -> float:
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
        for test_batch_X, _ in test_data:
            test_batch_X = test_batch_X.to(device)
            test_batch_X = test_batch_X.view(-1, 28 * 28)

            pred_X = model(test_batch_X)
            loss = loss_fn(pred_X, test_batch_X)
            test_losses.append(loss.item())

    return np.mean(np.array(test_losses))


def draw_losses(
    train_losses: List[float], 
    interval_size: int, 
    save_path: str,
    val_losses: List[float] = None
) -> None:

    assert len(train_losses) == len(val_losses)
    batches = [0]
    while len(batches) != len(train_losses):
        batches.append(batches[len(batches) - 1] + interval_size)
            
    plt.plot(batches, train_losses, 'r-', label = 'Training loss')
    plt.plot(batches, val_losses, 'b-', label = 'Validation loss')
    plt.title(f'Average MSE of the last {interval_size} batches.')
    plt.xlabel('Batch')
    plt.ylabel('Loss (MSE)')
    plt.legend(loc = 'upper right')
    plt.savefig(save_path)


def get_device() -> str:
    """Get the GPU as device if available, otherwise
    the CPU.

    Returns
    -------
        str: Indication whether GPU or CPU is used.
    """

    if torch.cuda.is_available():
        print('Using GPU...')
        device = torch.device('cuda:0')
    else:
        print('Using CPU...')
        device = torch.device('cpu')

    print()
    return device


def save_from_flat_tensor(flat_img: torch.Tensor, name: str) -> torch.Tensor:
    img = flat_img.view(28, 28)
    img *= MNIST_STD
    img += MNIST_MEAN
    img *= 255
    img = img.type(torch.uint8)
    print(img)
    plt.imsave(os.path.join(name), np.array(img.cpu()), cmap = 'Greys')