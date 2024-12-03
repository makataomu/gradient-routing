# %%

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torchvision import datasets, transforms

import projects.mnist.utils as utils
from factored_representations.utils import get_gpu_with_most_memory


def get_mlp(layer_sizes, final_layer_has_bias: bool):
    layers = []
    num_layers = len(layer_sizes) - 1
    for i in range(num_layers):
        is_final = i == num_layers - 1
        layers.append(
            nn.Linear(
                layer_sizes[i],
                layer_sizes[i + 1],
                bias=final_layer_has_bias if is_final else True,
            )
        )
        if not is_final:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SplitAutoencoder(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes: list[int],
        hidden_size: int,
        split_decoders: bool,
        use_split_encoding: bool,
    ):
        super(SplitAutoencoder, self).__init__()
        assert type(hidden_layer_sizes) is list, "Must pass list of layer sizes"
        self.hidden_size = hidden_size
        self.split_decoders = split_decoders
        self.use_split_encoding = use_split_encoding

        assert self.hidden_size % 2 == 0

        input_size = 28 * 28

        # Create encoders and decoders
        layer_sizes = [input_size] + hidden_layer_sizes + [self.hidden_size]
        self.encoder = get_mlp(layer_sizes, final_layer_has_bias=False)
        self.decoder_good = get_mlp(layer_sizes[::-1], final_layer_has_bias=True)
        self.decoder_certificate_top = get_mlp(
            [self.hidden_size // 2] + hidden_layer_sizes[::-1] + [input_size],
            final_layer_has_bias=True,
        )
        self.decoder_certificate_bot = get_mlp(
            [self.hidden_size // 2] + hidden_layer_sizes[::-1] + [input_size],
            final_layer_has_bias=True,
        )
        if self.split_decoders:
            self.decoder_bad = get_mlp(layer_sizes[::-1], final_layer_has_bias=True)
        else:
            self.decoder_bad = self.decoder_good

    def encode(self, x):
        batch_size = x.shape[0]
        encoding = self.encoder(x.reshape((batch_size, 784)))
        return encoding

    def split_encoding(self, encoding):
        midpoint = self.hidden_size // 2
        encoding_good, encoding_bad = encoding[:, :midpoint], encoding[:, midpoint:]
        encoding_w_good_gradient = torch.cat(
            (encoding_good, encoding_bad.detach()), dim=1
        )
        encoding_w_bad_gradient = torch.cat(
            (encoding_good.detach(), encoding_bad), dim=1
        )
        return encoding_w_good_gradient, encoding_w_bad_gradient

    def forward(self, x):
        batch_size = x.shape[0]
        encoding = self.encode(x)

        if self.use_split_encoding:
            encoding_w_good_gradient, encoding_w_bad_gradient = self.split_encoding(
                encoding
            )
        else:
            encoding_w_good_gradient, encoding_w_bad_gradient = encoding, encoding

        out_good = self.decoder_good(encoding_w_good_gradient).reshape(
            (batch_size, 1, 28, 28)
        )
        out_bad = self.decoder_bad(encoding_w_bad_gradient).reshape(
            (batch_size, 1, 28, 28)
        )
        return out_good, out_bad, encoding

    def forward_all(self, x):
        # Used for efficient training
        out_shape = (x.shape[0], 1, 28, 28)
        out_good, out_bad, encoding = self.forward(x)

        encoding_detached = encoding.detach()
        top_encoding = encoding_detached[:, : self.hidden_size // 2]
        bot_encoding = encoding_detached[:, self.hidden_size // 2 :]
        top_certificate = self.decoder_certificate_top(top_encoding).reshape(out_shape)
        bot_certificate = self.decoder_certificate_bot(bot_encoding).reshape(out_shape)
        return out_good, out_bad, top_certificate, bot_certificate, encoding

    def forward_certificate_top(self, x):
        """
        Note inconsistency in naming: in the paper, this is the "bottom half" Certificate.
        """
        out_shape = (x.shape[0], 1, 28, 28)
        top_encoding = self.encode(x)[:, : self.hidden_size // 2].detach()
        out = self.decoder_certificate_top(top_encoding).reshape(out_shape)
        return out

    def forward_certificate_bot(self, x):
        """
        Note inconsistency in naming: in the paper, this is the "top half" Certificate.
        """
        out_shape = (x.shape[0], 1, 28, 28)
        bot_encoding = self.encode(x)[:, self.hidden_size // 2 :].detach()
        out = self.decoder_certificate_bot(bot_encoding).reshape(out_shape)
        return out


def normalize(vectors, epsilon):
    # vectors.shape = (batch, embed)
    return (vectors - vectors.mean(dim=0)) / (vectors.std(dim=0) + epsilon)


def img_loss(pred, actual, subset=None):
    if subset is not None:
        pred = pred[subset, ...]
        actual = actual[subset, ...]
    return ((pred - actual).abs()).mean()


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


def hash_to_bool(imgs: torch.Tensor):
    return (((imgs.abs().sum(dim=(1, 2, 3)) * 100).round() % 17) % 2).bool()


def is_bad_data(imgs, labels):
    return labels > 4
    # return hash_to_bool(imgs)


def calculate_split_losses(batch, model, override_is_bad: torch.Tensor = None):
    device = next(model.parameters()).device
    img, labels = [d.to(device) for d in batch]
    out_good, out_bad, out_certificate_top, out_certificate_bot, encodings = (
        model.forward_all(img)
    )  # encodings.shape = (batch, hidden_size)

    encodings_n = normalize(encodings, 0)
    encodings_good_n = encodings_n[:, : model.hidden_size // 2]
    encodings_bad_n = encodings_n[:, model.hidden_size // 2 :]
    encoding_corr = (encodings_good_n.T @ encodings_bad_n) / encodings_good_n.shape[0]

    corr_penalty = encoding_corr.abs().mean()

    is_bad = is_bad_data(img, labels)
    override = (
        override_is_bad.to(device)
        if override_is_bad is not None
        else torch.zeros_like(is_bad)
    )

    top_encoding = encodings[:, : model.hidden_size // 2]
    bot_encoding = encodings[:, model.hidden_size // 2 :]
    losses = {
        "Good Decoder": img_loss(out_good, img, ~is_bad),
        "Good Decoder (bad data)": img_loss(out_good, img, is_bad),
        "Bad Decoder": img_loss(out_bad, img, subset=torch.maximum(is_bad, override)),
        "Certificate Decoder (top)": img_loss(out_certificate_top, img),
        "Certificate Decoder (bot)": img_loss(out_certificate_bot, img),
        "Good Encoder L1 penalty": top_encoding.abs().mean(dim=0).sum(),
        "Bad Encoder L1 penalty": bot_encoding.abs().mean(dim=0).sum(),
        "Correlation penalty": corr_penalty,
    }
    return losses


class LossLogger:
    def __init__(self, loss_weights):
        self.loss_weights = loss_weights
        self.losses = {loss_label: [] for loss_label in loss_weights.keys()}

    def append(self, loss_dict):
        for key, lst in self.losses.items():
            assert (
                key in loss_dict
            ), f"LossLogger expected '{key}' to be in loss_dict, instead got {self.losses.keys()}"
            lst.append(loss_dict[key].item())

    def plot(self):
        fig, ax = plt.subplots()
        for key, lst in self.losses.items():
            if self.loss_weights[key]:
                ax.plot(np.array(lst) * self.loss_weights[key], label=key)
        ax.legend()
        return ax


def train(
    model,
    dataloader,
    optimizer,
    num_epochs,
    loss_getter,
    loss_logger,
    routing_pct=None,
    use_pbar=True,
):
    epochs = tqdm.tqdm(range(num_epochs)) if use_pbar else range(num_epochs)
    for epoch in epochs:
        for data in dataloader:
            batch_size = data[0].shape[0]
            if routing_pct is not None:
                override_is_bad = torch.rand(batch_size) > routing_pct(epoch)
                losses = loss_getter(data, model, override_is_bad)
            else:
                losses = loss_getter(data, model)
            loss = 0
            for key, val in losses.items():
                assert (
                    key in loss_logger.loss_weights
                ), f"Unexpected loss key: {key}. Keys are {loss_logger.loss_weights.keys()}"
                if loss_logger.loss_weights[key]:
                    loss = loss + val * loss_logger.loss_weights[key]

            loss_logger.append(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(model, dataloader):
    device = next(model.parameters()).device

    def eval(pred, img):
        return ((pred - img) ** 2).mean(dim=(1, 2, 3))

    results = {
        "label": [],
        "decoder_good": [],
        "decoder_bad": [],
        "certificate_top": [],
        "certificate_bot": [],
    }
    with torch.inference_mode():
        for data in dataloader:
            img, labels = [d.to(device) for d in data]

            out_good, out_bad, certificate_top, certificate_bot, _ = model.forward_all(
                img
            )  # encodings.shape = (batch, hidden_size)
            preds = {
                "decoder_good": out_good,
                "decoder_bad": out_bad,
                "certificate_top": certificate_top,
                "certificate_bot": certificate_bot,
            }
            for label, pred in preds.items():
                mse_by_img = eval(pred, img)
                results[label].extend(list(mse_by_img.cpu().numpy()))
            results["label"].extend(list(labels.cpu().numpy()))

    df = pd.DataFrame(results)
    res = df.groupby("label").agg("mean")
    return res


def get_mnist_data(device):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = datasets.MNIST(
        "~/.pytorch/MNIST_data/", download=True, train=True, transform=transform
    )
    trainloader = PreloadedDataLoader(
        trainset, batch_size=2048, shuffle=True, device=device
    )

    testset = datasets.MNIST(
        "~/.pytorch/MNIST_data/", download=True, train=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return trainset, trainloader, testset, testloader


if __name__ == "__main__":
    device = get_gpu_with_most_memory()

    trainset, trainloader, testset, testloader = get_mnist_data(device)

    model = SplitAutoencoder(
        hidden_layer_sizes=[2048, 512],
        hidden_size=32,
        split_decoders=False,
        use_split_encoding=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)

    top_label = "5-9"
    logger = LossLogger(
        loss_weights={
            "Good Decoder": 1,
            "Good Decoder (bad data)": 1,
            "Bad Decoder": 1,
            "Certificate Decoder (top)": 1,
            "Certificate Decoder (bot)": 1,
            "Bad Encoder L1 penalty": 3e-3,
            "Good Encoder L1 penalty": 2e-2,  # 2e-2 if using all data for top else 3e-3,
            "Correlation penalty": 0.1,
        }
    )

    train(
        model,
        trainloader,
        optimizer=optimizer,
        num_epochs=200,
        loss_getter=calculate_split_losses,
        loss_logger=logger,
        routing_pct=lambda ep: 1,
    )

    # %%
    res = evaluate(model, testloader)
    res["is_bad"] = is_bad_data(None, labels=res.index)
    res["Decoder"] = res["decoder_bad"] * res["is_bad"] + res["decoder_good"] * (
        ~res["is_bad"]
    )

    fig, ax = plt.subplots(dpi=300, figsize=(4.5, 2.5))
    res[["Decoder", "certificate_top", "certificate_bot"]].plot(
        kind="bar", ax=ax, width=0.8
    )
    ax.set_xlabel("Label")
    ax.set_ylabel("Validation loss (MAE)")
    ax.axvline(4.5, color="black", ls=":")

    colors = ["C4", "C3", "C0"]
    hatches = [None, "///", "\\\\"]
    hatch_colors = [None, (1, 0.2, 0.2, 1), (0.2, 0.2, 1, 1)]
    labels = ["Decoder", "Certificate (top)", "Certificate (bot)"]

    patches = []
    for idx, bar in enumerate(ax.patches):
        color_idx = idx // 10
        color = colors[color_idx]
        hatch = hatches[color_idx]
        hatch_color = hatch_colors[color_idx]
        bar.set_facecolor(color)
        if hatch is not None:
            bar.set_edgecolor(hatch_color)
            bar.set_linewidth(0)
            bar.set_hatch(hatch)
        if idx % 10 == 0:
            patch = mpatches.Patch(
                facecolor=color,
                hatch=hatch,
                edgecolor=hatch_color,
                linewidth=0,
                label=labels[color_idx],
            )
            patches.append(patch)

    ax.legend(handles=patches)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    plt.savefig("figures/mnist_performance.pdf", bbox_inches="tight")

    # %%
    num_images = 4
    digits = utils.get_all_digits(num_images=num_images, dataset=testset)

    fig, axes = plt.subplots(
        nrows=5,
        ncols=num_images * 5,
        figsize=(2 * num_images, 2),
        dpi=300,
        gridspec_kw={"height_ratios": [1, 1, 0, 1, 1]},
    )
    for ax in axes.flatten():
        ax.axis("off")

    for label, digit_ims in enumerate(digits):
        rows = [0, 1] if label <= 4 else [3, 4]
        cols = slice((label % 5) * num_images, (label % 5) * num_images + num_images)
        utils.bulk_plot(digit_ims, axes[rows, cols], model, "bot")

    heights = [0.78, 0.60, 0.37, 0.19]
    labels = ["Input (0-4)", "Reconstruction", "Input (5-9)", "Reconstruction"]
    for height, label in zip(heights, labels):
        fig.text(0.115, height, label, va="center", ha="right", fontsize=9)

    plt.savefig("figures/mnist_reconstruction.pdf", pad_inches=0, bbox_inches="tight")

# %%


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

        fig, (axes_top, axes_zero, axes_bot) = plt.subplots(  # type: ignore
            ncols=num_ims, nrows=3, figsize=(6, 1.7)
        )
        for idx, ax in enumerate(axes_top):
            ax.imshow(top_images[idx])
            ax.set_ylabel("test")
            ax.axis("off")

        for idx, ax in enumerate(axes_zero):
            ax.imshow(zero_images[idx])
            ax.axis("off")

        for idx, ax in enumerate(axes_bot):
            ax.imshow(bot_images[idx])
            ax.axis("off")

        axes_top[len(axes_top) // 2].set_title(
            "Samples with greatest activation (+)", fontsize=7
        )
        axes_zero[len(axes_top) // 2].set_title(
            "Samples with smallest absolute activation (0)", fontsize=7
        )
        axes_bot[len(axes_top) // 2].set_title(
            "Samples with least activation (-)", fontsize=7
        )


if __name__ == "__main__":
    with torch.no_grad():
        all_images = trainloader.data[0]
        out_good, _, encodings = model(all_images)
        utils.visualize_encodings(
            encodings[:50].cpu().numpy(), labels=trainloader.data[1][:50].cpu()
        )

    PLOT_FOR_ARCHITECTURE_DIAGRAM = True
    if PLOT_FOR_ARCHITECTURE_DIAGRAM:
        with torch.inference_mode():
            bad, good = utils.get_bad_and_good_images(1, is_bad_data, trainset)

            bad = bad.to(device)
            good = good.to(device)

            _, out_bad, _ = model(bad)
            out_good, _, _ = model(good)

            out_good_certificate_top = model.forward_certificate_top(good)
            out_bad_certificate_top = model.forward_certificate_top(bad)

            out_good_certificate_bot = model.forward_certificate_bot(good)
            out_bad_certificate_bot = model.forward_certificate_bot(bad)

            ims = [
                good,
                bad,
                out_good,
                out_bad,
                out_good_certificate_top,
                out_bad_certificate_top,
                out_good_certificate_bot,
                out_bad_certificate_bot,
            ]
            ims = [x.squeeze().cpu().numpy() for x in ims]
            for x in ims:
                fig, ax = plt.subplots()
                ax.axis("off")
                ax.imshow(x, cmap="binary", vmin=-1, vmax=1)
