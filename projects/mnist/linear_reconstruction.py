# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import tqdm

import base_autoencoder
import representation_splitting as rs


def get_encodings(model, dataloader):
    with torch.no_grad():
        encodings = []
        for batch in tqdm.tqdm(dataloader):
            img, labels = [d.to(device) for d in batch]
            encoding_batch = model(img)[-1].cpu()
            encodings.append(encoding_batch)
        encodings = torch.cat(encodings).numpy()
    return encodings


def get_reconstruction_mappings(list_of_encodings, add_intercept: bool):
    """
    Find the error of optimal linear reconstruction error
    of one encoding from another
    """
    num_models = len(list_of_encodings)
    d_model = list_of_encodings[0].shape[-1]
    loss = np.zeros((num_models, num_models))
    mappings = np.empty((num_models, num_models, d_model + add_intercept, d_model))

    for i, X in enumerate(tqdm.tqdm(list_of_encodings)):
        n = X.shape[0]
        if add_intercept:
            X = np.concatenate((np.ones((n, 1)), X), axis=1)  # n x d+1
        E_XXT = (X.T @ X) / n  # d+1 x d+1
        for j, Y in enumerate(list_of_encodings):
            E_XYT = X.T @ Y / n  # d+1 x n @ n x d = d+1 x n
            optimal = np.linalg.pinv(E_XXT) @ E_XYT  # d+1 x d
            loss[i, j] = ((X @ optimal - Y) ** 2).sum(axis=1).mean()
            mappings[i, j] = optimal
    return mappings, loss


def visualize_images(model, dataloader, indexes):
    n_images = len(indexes)
    with torch.no_grad():
        batch = next(iter(dataloader))
        img, labels = [d[indexes].to(device) for d in batch]
        pred, *_ = model(img)

    fig, axes = plt.subplots(nrows=2, ncols=n_images, figsize=(n_images, 2.5))
    for i in range(n_images):
        axes[0, i].imshow(img.squeeze()[i].cpu())
        axes[1, i].imshow(pred.squeeze()[i].cpu())
        axes[0, i].axis("off")
        axes[1, i].axis("off")

    axes[0, n_images // 2].set_title("Input", fontsize=7)
    axes[1, n_images // 2].set_title("Prediction", fontsize=7)

    plt.show()


if __name__ == "__main__":
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Download and load the training data
    trainset = datasets.MNIST(
        "~/.pytorch/MNIST_data/", download=True, train=True, transform=transform
    )
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=2048, shuffle=True)
    trainloader = rs.PreloadedDataLoader(
        trainset, batch_size=2048, shuffle=True, device=device
    )

    # Download and load the test data
    testset = datasets.MNIST(
        "~/.pytorch/MNIST_data/", download=True, train=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    hidden_layer_sizes = [2048, 256]

    num_models_per_type = 1
    num_epochs = 2

    print("Training regular autoencoders...")
    models = []
    for _ in range(num_models_per_type):
        model = base_autoencoder.Autoencoder(hidden_layer_sizes, hidden_size=32).to(
            device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
        logger = rs.LossLogger(
            loss_weights={
                "Decoder": 1,
                "Encoder L1 penalty": 1e-3,
            }
        )

        rs.train(
            model,
            trainloader,
            optimizer,
            num_epochs,
            base_autoencoder.calculate_loss,
            logger,
            routing_pct=lambda ep: 1,
        )
        models.append(model)
    logger.plot()

    print("Training split autoencoders...")
    for _ in range(num_models_per_type):
        model = rs.SplitAutoencoder(
            hidden_layer_sizes,
            hidden_size=32,
            split_decoders=True,
            use_split_encoding=True,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
        logger = rs.LossLogger(
            loss_weights={
                "Decoder A": 1,
                "Decoder B": 1,
                "Target Decoder": 1,
                "Encoder L1 penalty": 1e-3,
                "Correlation penalty": 0.1,
            }
        )

        rs.train(
            model,
            trainloader,
            optimizer,
            num_epochs,
            rs.calculate_split_losses,
            logger,
            routing_pct=lambda ep: 1,
        )
        models.append(model)
    logger.plot()

    trainloader.shuffle = False
    encodings_list = [get_encodings(model, trainloader) for model in models]
    mappings, loss = get_reconstruction_mappings(encodings_list, add_intercept=True)

    mapping_norms = np.linalg.norm(mappings, axis=(-1, -2))

    # %%
    fig, ax = plt.subplots()
    pos = ax.imshow(loss)
    ax.set_title("Optimal linear reconstruction loss")
    ax.set_ylabel("Source")
    ax.set_xlabel("Target")
    fig.colorbar(pos, ax=ax)
    labels = ["Base autoencoders", "Split autoencoder"]
    x_min, x_max = ax.get_xlim()
    positions = [x_min + (x_max - x_min) * 0.25, x_min + (x_max - x_min) * 0.75]

    # Set x-ticks and x-tick labels
    plt.xticks(positions, labels)
    plt.yticks(positions, labels)
    for label in plt.gca().get_yticklabels():
        label.set_rotation(90)
        label.set_verticalalignment("center")
        label.set_horizontalalignment("center")

    # for (i, j), val in np.ndenumerate(loss):
    #     ax.text(j, i, f'{i},{j}', ha='center', va='center', color='gray', fontsize=12, alpha=1)

    fig, ax = plt.subplots()
    pos = ax.imshow(mapping_norms)
    ax.set_title("Optimal linear reconstruction norm")
    ax.set_ylabel("Source")
    ax.set_xlabel("Target")
    fig.colorbar(pos, ax=ax)
    labels = ["Base autoencoders", "Split autoencoder"]
    x_min, x_max = ax.get_xlim()
    positions = [x_min + (x_max - x_min) * 0.25, x_min + (x_max - x_min) * 0.75]

    # Set x-ticks and x-tick labels
    plt.xticks(positions, labels)
    plt.yticks(positions, labels)
    for label in plt.gca().get_yticklabels():
        label.set_rotation(90)
        label.set_verticalalignment("center")
        label.set_horizontalalignment("center")

    # for (i, j), val in np.ndenumerate(loss):
    #     ax.text(j, i, f'{i},{j}', ha='center', va='center', color='gray', fontsize=12, alpha=1)

    base_to_base_loss = loss[:num_models_per_type, :num_models_per_type].flatten()
    split_to_base_loss = loss[num_models_per_type:, :num_models_per_type].flatten()
    base_to_base_90th_pctile = np.quantile(base_to_base_loss, q=0.9)
    mean_split_to_base = np.mean(split_to_base_loss)
    print(f"Base-Base  - 90th percentile loss: {base_to_base_90th_pctile:0.3f} MSE")
    print(f"Split-Base - mean loss:            {mean_split_to_base:0.3f} MSE")
