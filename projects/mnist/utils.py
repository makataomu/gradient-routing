import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_bad_and_good_images(num_images, is_bad_data_fn, dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=min(200, len(dataset)), shuffle=False
    )
    all_images, all_labels = [d for d in next(iter(dataloader))]
    is_bad = is_bad_data_fn(all_images, all_labels)

    out = []
    for keep_ind in [is_bad, ~is_bad]:
        _, indices = torch.sort(all_labels[keep_ind][:num_images])
        images = all_images[keep_ind][indices]
        out.append(images)
    return out  # bad, good


def get_all_digits(num_images, dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=min(200, len(dataset)), shuffle=False
    )
    all_images, all_labels = [d for d in next(iter(dataloader))]
    return [all_images[all_labels == label][:num_images] for label in range(10)]


def bulk_plot(image_input, axes, model, certificate_type: str):
    device = next(model.parameters()).device
    forward = (
        model.forward_certificate_top
        if certificate_type == "top"
        else model.forward_certificate_bot
    )
    num_images = image_input.shape[0]
    with torch.no_grad():
        out_certificate = forward(image_input.to(device))
        to_plot = [im.squeeze().cpu().numpy() for im in [image_input, out_certificate]]

    labels = ["Image", "Certificate"]
    for row_idx, (label, im_array) in enumerate(zip(labels, to_plot)):
        for col_idx in range(num_images):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            ax.imshow(im_array[col_idx], cmap="binary", vmin=-1, vmax=1)

            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center")


def plot_bad_and_good(num_images, dataset, is_bad_data_fn, model):
    fig, axes = plt.subplots(
        nrows=4, ncols=num_images, figsize=(0.75 * num_images, 3), dpi=300
    )

    good_axes = axes[:2, :]
    bad_axes = axes[2:, :]

    bad_im, good_im = get_bad_and_good_images(num_images, is_bad_data_fn, dataset)

    bulk_plot(good_im, good_axes, model)
    bulk_plot(bad_im, bad_axes, model)

    plt.tight_layout()


def plot_all_digits(dataset, model, num_images=5):
    digits = get_all_digits(num_images=num_images, dataset=dataset)
    fig, axes = plt.subplots(
        nrows=2, ncols=num_images * 5, figsize=(0.75 * num_images, 10), dpi=300
    )

    for label, digit_ims in enumerate(digits):
        bulk_plot(
            digit_ims,
            axes[label // 5, label * num_images : label * num_images + num_images],
            model,
        )


def plot_with_mean(x, y, ax, **kwargs):
    ax.plot(x, y, **kwargs)
    y_mean = np.mean(y)
    for key in ["ls", "label"]:
        if key in kwargs:
            del kwargs[key]
    ax.plot(x, [y_mean if val is not None else None for val in x], ls=":", **kwargs)


def visualize_encodings(encodings, labels=None, vmin=None, vmax=None, title=""):
    fig, ax = plt.subplots(figsize=(8, 8))

    vmin = encodings.min() if vmin is None else vmin
    vmax = encodings.max() if vmin is None else vmax
    divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    if labels is not None:
        values, indices = torch.sort(labels)
        encodings = encodings[indices]
        ax.set_ylabel("Digit")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(values.cpu().numpy(), fontsize=7)

    pos = ax.imshow(encodings, cmap="coolwarm", norm=divnorm)
    plt.colorbar(pos, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Encoding position")
    return ax
