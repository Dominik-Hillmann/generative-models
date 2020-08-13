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
from typing import Callable, List, Tuple
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
        for batch_idx, (train_batch_X, _) in tqdm(enumerate(train_data)):
            batch_train_losses = []
            train_batch_X = train_batch_X.to(device)
            if flatten:
                train_batch_X = train_batch_X.view(-1, 28 * 28)

            model.zero_grad()
            pred_X = model(train_batch_X)
            loss = loss_fn(pred_X, train_batch_X)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())

            if batch_idx % (batch_size * n_batches_till_test) == 0:
                if test_data is not None:
                    test_loss = validate(model, test_data, loss_fn, device, flatten = flatten)
                    avg_test_losses.append(test_loss)

                avg_train_losses.append(np.mean(np.array(batch_train_losses)))                
                batch_train_losses.clear()
    
    if test_data is not None:
        draw_losses(avg_train_losses, n_batches_till_test, img_save_path, val_losses = avg_test_losses)
    else:
        draw_losses(avg_train_losses, n_batches_till_test, img_save_path, val_losses = avg_test_losses)

    if save:
        pass


def validate(model: nn.Module, test_data: DataLoader, loss_fn: Callable, device: str, flatten: bool = True) -> float:
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
            if flatten:
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
    """Saves an image of the training and validation losses.

    Args
    ----
        train_losses (List[float]): The losses of the training data.
        interval_size (int): How many training batches are forwarded until the test data is run.
        save_path (str): The path the image will be saved to.
        val_losses (List[float], optional): The losses of the validation data. Defaults to None.
    """

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


def save_from_flat_tensor(flat_img: torch.Tensor, save_path: str) -> None:
    """Transforms a flat ```(-1, 28 * 28)``` back to ```(-1, 28, 28)``` and 
    saves it as an image.

    Args
    ----
        flat_img (torch.Tensor): The flattened tensor.
        save_path (str): The path the image will be saved to.
    """

    img = flat_img.view(28, 28)
    img *= MNIST_STD
    img += MNIST_MEAN
    img *= 255
    img = img.type(torch.uint8)
    plt.imsave(save_path, np.array(img.cpu()), cmap = 'Greys')


def travel_2d_latent_space(
    model: nn.Module,
    latent_space_size: int,
    latent_idxs: Tuple[int, int],
    save_path: str,
    n_steps: Tuple[int, int] = (5, 5),
    step_size: float = 1.5,
    latent_z_start: float = -1.0,
    unused_latent_dim_val: float = 0.0
) -> None:
    """Creates an image with outputs of the decoder using a range of values
    in the latent space.

    Args
    ----
        model (nn.Module): The decoder which has to implmenent predict_from_latent_space.
        latent_space_size (int): Number of dimensions in the latent space.
        latent_idxs (Tuple[int, int]): The selection of two dimensions in the latent space that will be changed.
        save_path (str): The path the image of the travel will be saved to.
        n_steps (Tuple[int, int], optional): Number of steps in both selected dimensions. Defaults to (5, 5).
        step_size (float, optional): The size of each step in the selected latent dimensions. Defaults to 1.5.
        latent_z_start (float, optional): Start value of the selected latent dimensions. Defaults to -1.0.
        unused_latent_dim_val (float, optional): The value all latent dimensions will be set to while the two selected are being traveled. Defaults to 0.0.
    """

    latent_space = [unused_latent_dim_val] * latent_space_size
    fig, ax = plt.subplots(*n_steps, sharex = True, sharey = True)
    fig.tight_layout(pad = 1.0)
    latent_idx1, latent_idx2 = latent_idxs
    z1 = latent_z_start
    z2 = latent_z_start
    i = 0
    for z1_step in range(n_steps[0]):
        z2 = latent_z_start
        for z2_step in range(n_steps[1]):
            fig.add_subplot(*n_steps, i + 1)

            latent_space[latent_idx1] = z1
            latent_space[latent_idx2] = z2
            latent_X = torch.Tensor(latent_space)
            reconstruction = model.pred_from_latent_space(latent_X).cpu()
            
            ax[z1_step, z2_step].set_yticklabels([])
            ax[z1_step, z2_step].set_xticklabels([])
            ax[z1_step, z2_step].axis('off')
            plt.title(f'$(z_1={round(z1, 1)}, z_2={round(z2, 1)})$', fontdict = {'fontsize': 7})
            plt.imshow(reconstruction, cmap = 'Greys')
            plt.axis('off')

            i += 1
            z2 += step_size
        z1 += step_size
    # fig.suptitle('Reconstruction from latent space with values $(x_1, x_2)$', y = -0.01)
    plt.savefig(save_path)