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
    """The function that trains a model.

    Args
    ----
        model (nn.Module): The model.
        train_data (DataLoader): The data as PyTorch DataLoader.
        device (str): The device.
        test_data (DataLoader, optional): The validation data as PyTorch DataLoader. Defaults to None.
        n_epochs (int, optional): Number of epochs. Defaults to 5.
        lr (float, optional): Learning rate. Defaults to 0.001.
        n_batches_till_test (int, optional): Controls at what interval of batches the test data is run. Defaults to 10.
        save (bool, optional): Whether the model should be saved after training. Defaults to False.
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
    """Runs the validation data and returns the average loss.

    Args
    ----
        model (nn.Module): The model on which the data will be tested.
        test_data (DataLoader): Validation data as PyTorch DataLoader.
        loss_fn (Callable): The loss function.
        device (str): The device.

    Returns
    -------
        float: The average loss value.
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