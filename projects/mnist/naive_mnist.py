# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torchvision import datasets, transforms

import projects.mnist.utils as utils


class Autoencoder(nn.Module):
    def __init__(self, hidden_size):
        super(Autoencoder, self).__init__()
        self.hidden_size = hidden_size
        assert self.hidden_size % 2 == 0

        size_1 = 2048
        size_2 = 256

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, size_1),
            nn.ReLU(True),
            nn.Linear(size_1, size_2),
            nn.ReLU(True),
            nn.Linear(size_2, self.hidden_size, bias=False),
        )

        self.decoder_naive = nn.Sequential(
            nn.Linear(self.hidden_size, size_2),
            nn.ReLU(True),
            nn.Linear(size_2, size_1),
            nn.ReLU(True),
            nn.Linear(size_1, 28 * 28),
        )

        self.decoder_certificate = nn.Sequential(
            nn.Linear(self.hidden_size, size_2),
            nn.ReLU(True),
            nn.Linear(size_2, size_1),
            nn.ReLU(True),
            nn.Linear(size_1, 28 * 28),
        )

    def encode(self, x):
        batch_size = x.shape[0]
        encoding = self.encoder(x.reshape((batch_size, 784)))
        return encoding

    def forward(self, x):
        batch_size = x.shape[0]
        encoding = self.encode(x)
        out_good = self.decoder_naive(encoding).reshape((batch_size, 1, 28, 28))
        out_certificate = self.decoder_certificate(encoding.detach()).reshape(
            (batch_size, 1, 28, 28)
        )
        return out_good, out_certificate, encoding

    def forward_certificate(self, x):
        batch_size = x.shape[0]
        encoding = self.encode(x)
        out = self.decoder_certificate(encoding.detach()).reshape(
            (batch_size, 1, 28, 28)
        )
        return out


def img_loss(pred, actual, subset=None, clip_max=torch.inf):
    if subset is not None:
        pred = pred[subset, ...]
        actual = actual[subset, ...]
    return (torch.clip((pred - actual).abs(), None, clip_max)).mean()


class PreloadedDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, device="cuda"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        all_data = next(
            iter(
                torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset))
            )
        )
        self.data = [data.to(device) for data in all_data]

        # Create indices for sampling
        self.indices = torch.arange(len(self.dataset))

    def __iter__(self):
        self.current = 0
        if self.shuffle:
            self.indices = torch.randperm(len(self.dataset))
        return self

    def __next__(self):
        if self.current >= len(self.dataset):
            raise StopIteration

        # Compute the batch indices
        start_index = self.current

        end_index = min(start_index + self.batch_size, len(self.dataset))
        batch_indices = self.indices[start_index:end_index]

        # Increment current index
        self.current += self.batch_size

        # Return the batch of data
        return [d[batch_indices] for d in self.data]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Download and load the training data
    trainset = datasets.MNIST(
        "~/.pytorch/MNIST_data/", download=True, train=True, transform=transform
    )
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=2048, shuffle=True)
    trainloader = PreloadedDataLoader(
        trainset, batch_size=2048, shuffle=True, device=device
    )

    # Download and load the test data
    testset = datasets.MNIST(
        "~/.pytorch/MNIST_data/", download=True, train=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    model = Autoencoder(hidden_size=32).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)

    def is_bad_data(imgs, labels):
        return labels <= 5

    bad_labels_label = "0-5"
    good_labels_label = "6-9"

    class LossLogger:
        def __init__(self, loss_weights):
            self.loss_weights = loss_weights
            self.losses = {loss_label: [] for loss_label in loss_weights.keys()}

        def append(self, loss_dict):
            for key, lst in self.losses.items():
                lst.append(loss_dict[key].item())

        def compute_loss(self, loss_dict):
            loss = 0
            for key, loss_item in loss_dict.items():
                loss += loss_item * self.loss_weights[key]
            return loss

        def plot(self):
            fig, ax = plt.subplots()
            for key, lst in self.losses.items():
                if self.loss_weights[key]:
                    ax.plot(np.array(lst) * self.loss_weights[key], label=key)
            ax.legend()
            return ax

    logger = LossLogger(
        loss_weights={
            "Naive Decoder": 1,
            "Certificate Decoder": 1,
            "Unlearning loss": 0,
            "Encoder L1 penalty": 3e-3,
        }
    )

    num_epochs = 100
    for epoch in tqdm.tqdm(range(num_epochs)):
        for data in trainloader:
            img, labels = [d.to(device) for d in data]
            out_good, out_certificate, encodings = model(
                img
            )  # encodings.shape = (batch, hidden_size)

            losses = {
                "Naive Decoder": img_loss(
                    out_good, img, subset=~is_bad_data(img, labels)
                ),
                "Certificate Decoder": img_loss(out_certificate, img, subset=None),
                "Unlearning loss": img_loss(
                    out_good, torch.ones_like(out_good), subset=is_bad_data(img, labels)
                ),
                # "Unlearning loss" : 1-img_loss(out_good, img, subset=is_bad_data(img, labels), clip_max=1),
                "Encoder L1 penalty": encodings.abs().mean(dim=0).sum(),
            }
            loss = 0
            for key, val in losses.items():
                if logger.loss_weights[key]:
                    loss = loss + val * logger.loss_weights[key]

            logger.append(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ax = logger.plot()
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")

    utils.plot_bad_and_good(10, testset, is_bad_data, model)

    def evaluate(dataloader):
        def eval(pred, img):
            return ((pred - img) ** 2).mean(dim=(1, 2, 3))

        results = {"label": [], "decoder_naive": [], "decoder_certificate": []}

        with torch.no_grad():
            for data in dataloader:
                img, labels = [d.to(device) for d in data]

                out_a, out_b, _ = model(img)  # encodings.shape = (batch, hidden_size)
                preds = {"decoder_naive": out_a, "decoder_certificate": out_b}
                for label, pred in preds.items():
                    mse_by_img = eval(pred, img)
                    results[label].extend(list(mse_by_img.cpu().numpy()))
                results["label"].extend(list(labels.cpu().numpy()))

        df = pd.DataFrame(results)
        res = df.groupby("label").agg("mean")
        return res

    # Adapted from representation_splitting.py. Lord forgive me.
    res = evaluate(testloader)
    res["is_bad"] = is_bad_data(None, labels=res.index)

    fig, ax = plt.subplots()
    for bad_setting in [True, False]:
        subset = res[res.is_bad == bad_setting]
        utils.plot_with_mean(
            subset.index,
            subset["decoder_naive"],
            ax=ax,
            label="Naive Decoder (predict a constant on bad)"
            if bad_setting
            else None,  # only one legend entry
            c="gray",
            lw=2.5,
        )

        utils.plot_with_mean(
            subset.index,
            subset["decoder_certificate"],
            ax=ax,
            label="Certificate" if bad_setting else None,  # only one legend entry
            lw=2.5,
            ls="--",
            c="C8",
        )
    ax.set_xticks(range(10))
    ax.set_xlabel("Label")
    ax.set_ylabel("Loss")
    for idx, is_bad in enumerate(res["is_bad"]):
        if is_bad:
            ax.axvspan(idx - 0.5, idx + 0.5, color="gray", alpha=0.1, lw=0)
    ax.set_title("Losses by class")
    ax.legend()
    plt.show()


def visualize_avg_img(images, encodings, encoding_direction, ax=None):
    broadcast_encodings = encodings[:, encoding_direction][:, None, None, None]
    torch.sort(broadcast_encodings, dim=0)
    avg_weighted_img = (nn.ReLU()(broadcast_encodings) * images).squeeze().mean(dim=0)

    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(avg_weighted_img.cpu().numpy())
    ax.axis("off")


def get_top_activating_images(images, encodings, encoding_direction, num_ims):
    with torch.no_grad():
        broadcast_encodings = encodings[:, encoding_direction][:, None, None, None]
        sorted_indices = torch.sort(broadcast_encodings, dim=0).indices
        top_images = images[sorted_indices][-num_ims:, ...].squeeze().cpu()
        bot_images = images[sorted_indices][:num_ims, ...].squeeze().cpu()

        broadcast_encodings_abs = encodings[:, encoding_direction][
            :, None, None, None
        ].abs()
        sorted_indices = torch.sort(broadcast_encodings_abs, dim=0).indices
        zero_images = images[sorted_indices][:num_ims, ...].squeeze().cpu()

        fig, (axes_top, axes_zero, axes_bot) = plt.subplots(
            ncols=num_ims, nrows=3, figsize=(6, 2)
        )
        for idx, ax in enumerate(axes_top):
            ax.imshow(top_images[idx])
            ax.axis("off")

        for idx, ax in enumerate(axes_zero):
            ax.imshow(zero_images[idx])
            ax.axis("off")

        for idx, ax in enumerate(axes_bot):
            ax.imshow(bot_images[idx])
            ax.axis("off")


utils.visualize_encodings(
    encodings[:50].detach().cpu().numpy(), labels=labels[:50].detach().cpu()
)


# %%
def get_maximizing_image(model, direction: torch.Tensor, num_steps: int):
    for param in model.parameters():
        param.requires_grad = False

    num_directions, encoding_size = direction.shape

    # img_init = torch.rand(num_directions, 1, 28, 28, device=device)
    img_init = img[:num_directions].clone().detach()
    input_img = nn.Parameter(img_init)
    direction = direction.to(device)

    optimizer = torch.optim.Adam([input_img], lr=1e-3, weight_decay=5e-5)

    logger = LossLogger(
        loss_weights={
            "Activation loss": 0.1,
            "Penalty": 100,
        }
    )
    for _ in tqdm.tqdm(range(num_steps)):
        activations = (model.encode(input_img) * direction).sum(dim=1)

        clipped_img = (input_img > 1) * input_img + (input_img < 1) * input_img
        deviation = ((input_img - clipped_img) ** 2).sum()

        losses = {"Activation loss": -activations.sum(), "Penalty": deviation}
        loss = logger.compute_loss(losses)

        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        logger.append(losses)
    logger.plot()
    return input_img.detach().cpu().squeeze()


if __name__ == "__main__":
    with torch.no_grad():
        all_images = trainloader.data[0]
        out_good, _, encodings = model(all_images)

        for neuron_idx in range(20, 26):
            print(f"Neuron idx: {neuron_idx}")
            get_top_activating_images(all_images, encodings, neuron_idx, 20)
            visualize_avg_img(all_images, encodings, neuron_idx)

    # num_directions = 10
    # directions = torch.rand((num_directions, model.hidden_size))
    # max_img = get_maximizing_image(model, directions, 700)
    # max_img_out = model(max_img.to(device))[0].squeeze().detach().cpu()
    # fig, axes = plt.subplots(nrows=num_directions, ncols=2, figsize=(4,10))
    # for idx, ax in enumerate(axes[:,0]):
    #     ax.imshow(max_img[idx])
    #     ax.axis("off")
    # for idx, ax in enumerate(axes[:,1]):
    #     ax.imshow(max_img_out[idx])
    #     ax.axis("off")
