# %%
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from torchvision import datasets, transforms

import projects.mnist.base_autoencoder as base_autoencoder
import projects.mnist.representation_splitting as rs

""" These are NOT the ablations that appear in the paper. """

q5 = lambda x: np.quantile(x, q=0.05)
q95 = lambda x: np.quantile(x, q=0.95)


def get_ci_width(data):
    return 1.96 * np.std(data) / np.sqrt(len(data))


def img_loss(pred, img):
    return ((pred - img).abs()).mean(dim=(1, 2, 3))


def evaluate(model, dataloader):
    device = next(model.parameters()).device
    results = {"label": [], "loss": [], "is_bad": []}
    with torch.inference_mode():
        for data in dataloader:
            img, labels = [d.to(device) for d in data]

            if isinstance(model, base_autoencoder.Autoencoder):
                out, _ = model(img)
            elif isinstance(model, rs.SplitAutoencoder):
                out = model.forward_certificate(img)
            else:
                raise ValueError(
                    "Model must be an instance of Autoencoder or SplitAutoencoder"
                )

            mse_by_img = img_loss(out, img)

            results["label"].extend(list(labels.cpu().numpy()))
            results["loss"].extend(list(mse_by_img.cpu().numpy()))
            results["is_bad"].extend(rs.is_bad_data(img, labels).cpu().numpy())

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    device = "cuda:3" if torch.cuda.is_available() else "cpu"

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
    testloader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=True)

    def get_trained_model_losses(
        model_class,
        model_args,
        loss_weights: dict,
        num_runs: int,
        num_epochs: int,
        vars_to_aggregate=["train", "is_bad"],
    ):
        all_aggregated = []
        for run_id in range(num_runs):
            model = model_class(**model_args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
            logger = rs.LossLogger(loss_weights)

            if isinstance(model, rs.SplitAutoencoder):
                loss_getter = rs.calculate_split_losses
            elif isinstance(model, base_autoencoder.Autoencoder):
                loss_getter = base_autoencoder.calculate_loss
            else:
                raise ValueError(
                    "Model must be an instance of Autoencoder or SplitAutoencoder"
                )

            rs.train(model, trainloader, optimizer, num_epochs, loss_getter, logger)

            res_train = evaluate(model, trainloader)
            res_test = evaluate(model, testloader)
            res_train["train"] = True
            res_test["train"] = False
            res = pd.concat((res_train, res_test))
            agg = res.groupby(vars_to_aggregate).agg({"loss": "mean"}).reset_index()
            agg["run_id"] = run_id
            all_aggregated.append(agg)
        return pd.concat(all_aggregated)

    num_models_per_type = 10
    num_epochs = 200

    loss_weights_base_default = {"Decoder": 1, "Encoder L1 penalty": 3e-3}

    loss_weights_split_default = {
        "Good Decoder": 1,
        "Bad Decoder": 1,
        "Certificate Decoder": 1,
        "Bad Encoder L1 penalty": 3e-3,
        "Good Encoder L1 penalty": 3e-3,
        "Correlation penalty": 0.1,
    }

    model_args_default = {
        "hidden_layer_sizes": [2048, 256],
        "hidden_size": 32,
        "split_decoders": False,
        "use_split_encoding": True,
    }

    folder_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(folder_path, "results")
    figures_path = os.path.join(folder_path, "figures")
    assert os.path.exists(results_path), f"Results folder not found: {results_path}"

    def run_experiments(
        experiment_name,
        experiment_value,
        model_args_base,
        model_args_split,
        loss_weights_base,
        loss_weights_split,
    ):
        df_base = get_trained_model_losses(
            base_autoencoder.Autoencoder,
            model_args_base,
            loss_weights_base,
            num_models_per_type,
            num_epochs,
        )
        df_split = get_trained_model_losses(
            rs.SplitAutoencoder,
            model_args_split,
            loss_weights_split,
            num_models_per_type,
            num_epochs,
        )
        df_base["model"] = "Gradient routing"
        df_split["model"] = "Base model"
        df = pd.concat((df_base, df_split))
        df[experiment_name] = experiment_value
        return df

    def aggregate(df, variable_name):
        agg = (
            df.groupby(["is_bad", "model", variable_name])
            .agg({"loss": ["mean", q5, q95]})
            .reset_index()
        )
        agg.columns = ["is_bad", "model", variable_name, "mean", "q5", "q95"]
        return agg

    def plot(agg, variable_name, title):
        kwargs = {
            "Gradient routing": {"ls": "--", "color": "C0"},
            "Base model": {"ls": "-", "color": "C1"},
        }
        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(6, 3.7))
        for is_bad, ax in zip([True, False], axes):
            ax.grid("on", alpha=0.2)
            for model in agg.model.unique():
                subset = agg[(agg["is_bad"] == is_bad) & (agg["model"] == model)]
                ax.plot(
                    subset[variable_name], subset["mean"], label=model, **kwargs[model]
                )
                ax.fill_between(
                    subset[variable_name],
                    subset["q5"],
                    subset["q95"],
                    alpha=0.2,
                    **kwargs[model],
                )
                ax.set_xlabel(variable_name)
                ax.set_title("Off-target data (0-4)" if is_bad else "Target data (5-9)")
        # axes[0].legend()
        axes[0].set_ylabel("Reconstruction test loss (MAE)")
        fig.suptitle(title)
        plt.tight_layout()
        return fig, axes

    """
    LAYER SIZE EXPERIMENT
    """
    path = os.path.join(results_path, "mnist_ablation_layer_size.csv")
    if not os.path.exists(path):
        all_results = []
        for layer_size in tqdm.tqdm([32, 64, 128, 256, 512, 750, 1028]):
            # Override default settings
            model_args = copy.deepcopy(model_args_default)
            model_args["hidden_layer_sizes"] = [layer_size, layer_size]

            df = run_experiments(
                "layer_size",
                layer_size,
                model_args,
                model_args,
                loss_weights_base_default,
                loss_weights_split_default,
            )
            all_results.append(df)

        res = pd.concat(all_results)
        res.to_csv(path)
    else:
        res = pd.read_csv(path, index_col=0)
    agg = aggregate(res[~res.train], "layer_size")

    fig, axes = plot(agg, "layer_size", title="Varying size of all hidden layers")
    plt.savefig(os.path.join(figures_path, "mnist_ablation_layer_size.pdf"))

    """
    L1 PENALTY EXPERIMENT
    """
    path = os.path.join(results_path, "mnist_ablation_l1_penalty.csv")
    if not os.path.exists(path):
        all_results = []
        for l1_penalty in [1e-5, 1e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 1e-2, 5e-2]:
            loss_weights_base = copy.deepcopy(loss_weights_base_default)
            loss_weights_base["Encoder L1 penalty"] = l1_penalty

            loss_weights_split = copy.deepcopy(loss_weights_split_default)
            loss_weights_split["Bad Encoder L1 penalty"] = l1_penalty
            loss_weights_split["Good Encoder L1 penalty"] = l1_penalty

            df = run_experiments(
                "l1_penalty",
                l1_penalty,
                model_args_default,
                model_args_default,
                loss_weights_base,
                loss_weights_split,
            )
            all_results.append(df)
        res2 = pd.concat(all_results)
        res2.to_csv(path)
    else:
        res2 = pd.read_csv(path, index_col=0)

    agg2 = aggregate(res2[~res2.train], "l1_penalty")

    fig, axes = plot(
        agg2[agg2.model == "Gradient routing"], "l1_penalty", title="Varying L1 penalty"
    )
    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(
            loss_weights_split_default["Bad Encoder L1 penalty"],
            color="black",
            ls=":",
            alpha=0.2,
        )
    plt.savefig(os.path.join(figures_path, "mnist_ablation_l1_penalty.pdf"))

    """
    CORRELATION PENALTY EXPERIMENT
    """
    path = os.path.join(results_path, "mnist_ablation_correlation_penalty.csv")
    if not os.path.exists(path):
        all_results = []
        for correlation_penalty in [1e-6, 1e-4, 1e-2, 0.1, 0.5, 1, 2, 10]:
            loss_weights_split = copy.deepcopy(loss_weights_split_default)
            loss_weights_split["Correlation penalty"] = correlation_penalty

            df_split = get_trained_model_losses(
                rs.SplitAutoencoder,
                model_args_default,
                loss_weights_split,
                num_models_per_type,
                num_epochs,
            )
            df_split["model"] = "Split"
            df_split["Correlation penalty"] = correlation_penalty
            all_results.append(df_split)
        res3 = pd.concat(all_results)
        res3.to_csv(path)
    else:
        res3 = pd.read_csv(path, index_col=0)

    agg3 = aggregate(res3[~res3.train], "Correlation penalty")

    fig, axes = plot(
        agg3,
        "Correlation penalty",
        title="Varying correlation penalty",
    )
    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(
            loss_weights_split_default["Correlation penalty"],
            color="black",
            ls=":",
            alpha=0.2,
        )
    plt.savefig(os.path.join(figures_path, "mnist_ablation_correlation_penalty.pdf"))

    """
    HIDDEN SIZE EXPERIMENT
    """
    hidden_sizes = [4, 8, 16, 32, 40, 64, 128, 256]
    path = os.path.join(results_path, "mnist_ablation_hidden_size.csv")
    if not os.path.exists(path):
        all_results = []
        for hidden_size in hidden_sizes:
            model_args = copy.deepcopy(model_args_default)
            model_args["hidden_size"] = hidden_size

            df = run_experiments(
                "hidden size",
                hidden_size,
                model_args,
                model_args,
                loss_weights_base_default,
                loss_weights_split_default,
            )
            all_results.append(df)
        res4 = pd.concat(all_results)
        res4.to_csv(path)
    else:
        res4 = pd.read_csv(path, index_col=0)

    agg4 = aggregate(res4[~res4.train], "hidden size")

    fig, axes = plot(
        agg4, "hidden size", title="Varying hidden size (default L1 penalty: 3e-3)"
    )
    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(model_args_default["hidden_size"], color="black", ls=":", alpha=0.2)
    plt.savefig(os.path.join(figures_path, "mnist_ablation_hidden_size_with_l1.pdf"))

    path = os.path.join(results_path, "mnist_ablation_hidden_size_no_l1.csv")
    if not os.path.exists(path):
        all_results = []
        for hidden_size in hidden_sizes:
            model_args = copy.deepcopy(model_args_default)
            model_args["hidden_size"] = hidden_size

            loss_weights_base = copy.deepcopy(loss_weights_base_default)
            loss_weights_base["Encoder L1 penalty"] = 0

            loss_weights_split = copy.deepcopy(loss_weights_split_default)
            loss_weights_split["Bad Encoder L1 penalty"] = 0
            loss_weights_split["Good Encoder L1 penalty"] = 0

            df = run_experiments(
                "hidden size",
                hidden_size,
                model_args,
                model_args,
                loss_weights_base,
                loss_weights_split,
            )
            all_results.append(df)
        res4b = pd.concat(all_results)
        res4b.to_csv(path)
    else:
        res4b = pd.read_csv(path, index_col=0)

    agg4 = aggregate(res4b[~res4b.train], "hidden size")

    fig, axes = plot(agg4, "hidden size", title="Varying hidden size (L1 penalty of 0)")
    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(model_args_default["hidden_size"], color="black", ls=":", alpha=0.2)
    plt.savefig(os.path.join(figures_path, "mnist_ablation_hidden_size_no_l1.pdf"))

    """
    Varying combos
    """
    settings = {
        "use_split_encoding": [True, False],
        "Correlation penalty": [0, 0.1, 1, 5],
        "Encoder L1 penalty": [0, 1e-3, 3e-3, 5e-3],
    }
    # Run experiments with all combinations of settings
    path = os.path.join(results_path, "mnist_ablation_combos.csv")
    if not os.path.exists(path):
        all_results = []

        for use_split_encoding in settings["use_split_encoding"]:
            for correlation_penalty in settings["Correlation penalty"]:
                for l1_penalty in settings["Encoder L1 penalty"]:
                    model_args = copy.deepcopy(model_args_default)
                    model_args["use_split_encoding"] = use_split_encoding

                    loss_weights_split = copy.deepcopy(loss_weights_split_default)
                    loss_weights_split["Correlation penalty"] = correlation_penalty
                    loss_weights_split["Bad Encoder L1 penalty"] = l1_penalty
                    loss_weights_split["Good Encoder L1 penalty"] = l1_penalty

                    df_split = get_trained_model_losses(
                        rs.SplitAutoencoder,
                        model_args,
                        loss_weights_split,
                        num_models_per_type,
                        num_epochs,
                    )
                    df_split["model"] = "Split"
                    df_split["Correlation penalty"] = correlation_penalty
                    df_split["Encoder L1 penalty"] = l1_penalty
                    df_split["use_split_encoding"] = use_split_encoding

                    all_results.append(df_split)
        res5 = pd.concat(all_results)
        res5.to_csv(path)
    else:
        res5 = pd.read_csv(path, index_col=0)
    # %%
    agg = (
        res5[~res5.train]
        .groupby(["is_bad", "model", *settings.keys()])
        .agg({"loss": "mean"})
        .reset_index()
    )
    pivoted = agg.pivot_table(
        index=["model", *settings.keys()], columns="is_bad", values="loss"
    ).reset_index()
    pivoted.rename({False: "loss_good", True: "loss_bad"}, axis=1, inplace=True)
    # pivoted = pivoted[pivoted.loss_good <= 0.25]
    fig, ax = plt.subplots(figsize=(6, 5))

    for use_split_encoding, marker in zip([True, False], ["^", "o"]):
        subset = pivoted[pivoted["use_split_encoding"] == use_split_encoding]
        edge_colors = subset["Encoder L1 penalty"].map(
            {0: "white", 1e-3: "darkgray", 3e-3: "gray", 5e-3: "black"}
        )
        colors = subset["Correlation penalty"].map(
            {0: "white", 0.1: "darkgray", 1: "gray", 5: "black"}
        )
        ax.scatter(
            subset["loss_bad"],
            subset["loss_good"],
            c=colors,
            edgecolors=edge_colors,
            marker=marker,
            s=40,
        )
    ax.set_xlabel("Loss on BAD data")
    ax.set_ylabel("Loss on GOOD data")
    plt.gca().invert_yaxis()
    ax.grid(alpha=0.3)
    ax.set_title("Performance of varying setting combinations")
    ax.patch.set_facecolor("lightblue")
    # ax.set_aspect('equal')

    ax.autoscale(False)
    x = np.linspace(-10, 10, 2)
    ax.plot(x, x, color="black", ls=":", alpha=0.3)

    # Add legend
    for l1_penalty, color in zip(
        settings["Encoder L1 penalty"], ["white", "darkgray", "gray", "black"]
    ):
        ax.scatter(
            [],
            [],
            c="lightblue",
            lw=2,
            edgecolors=color,
            label=f"L1 penalty: {l1_penalty}",
        )

    for correlation_penalty, color in zip(
        settings["Correlation penalty"], ["white", "darkgray", "gray", "black"]
    ):
        ax.scatter(
            [],
            [],
            c=color,
            edgecolors=None,
            label=f"Correlation penalty: {correlation_penalty}",
        )

    for use_split_encoding, marker in zip([True, False], ["^", "o"]):
        ax.scatter(
            [],
            [],
            c="white",
            edgecolors="black",
            lw=0.7,
            marker=marker,
            label=f"Gradient routing: {use_split_encoding}",
        )

    ax.legend(loc="lower left", facecolor="white", framealpha=0.3)

    # Add arrows
    equal_pt = 0.18
    dist = 0.04

    plt.annotate(
        "",  # The text label
        xy=(equal_pt + dist, equal_pt - dist),  # The point to place the arrow at
        xytext=(equal_pt, equal_pt),  # The point to place the text at
        arrowprops=dict(arrowstyle="<-"),  # Arrow properties
    )
    plt.savefig(os.path.join(figures_path, "ablations.pdf"))
# %%
