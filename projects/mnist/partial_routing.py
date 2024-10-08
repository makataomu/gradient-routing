# %%
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from torchvision import datasets, transforms

import projects.mnist.ablations as ablations
import projects.mnist.representation_splitting as rs

folder_path = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(folder_path, "results")
figures_path = os.path.join(folder_path, "figures")

"""
$(pdm venv activate) && python projects/mnist/partial_routing.py
"""

if __name__ == "__main__":
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    print(device)

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

    routing_pcts = np.concatenate((np.linspace(0, 0.5, 5), np.linspace(0.5, 1, 10)))
    num_models_per_setting = 10

    n_epochs_shared = 200
    num_epochs = {
        "constant proportion": n_epochs_shared,
        "pretrain fully routed": n_epochs_shared,
    }

    routing_style = {
        "constant proportion": lambda ep, routing_pct: routing_pct,
        "pretrain fully routed": lambda ep, routing_pct: 1
        if ep / num_epochs["pretrain fully routed"] < routing_pct
        else 0,
    }

    RERUN_EXPERIMENTS = False
    SAVE_PATH = os.path.join(results_path, "partial_routing_results.csv")
    if os.path.exists(SAVE_PATH) and not RERUN_EXPERIMENTS:
        print(f"Loading experiment results from {SAVE_PATH}.")
        df = pd.read_csv(SAVE_PATH)
    else:
        print("Running experiments.")
        models = {key: [] for key in routing_style.keys()}
        results = []
        num_experiments = (
            len(routing_style) * len(routing_pcts) * num_models_per_setting
        )
        progress_bar = tqdm.tqdm(range(num_experiments))
        for routing_label, routing_fn in routing_style.items():
            for routing_pct in routing_pcts:
                for model_idx in range(num_models_per_setting):
                    model = rs.SplitAutoencoder(
                        hidden_layer_sizes,
                        hidden_size=32,
                        split_decoders=True,
                        use_split_encoding=True,
                    ).to(device)
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=1e-3, weight_decay=5e-5
                    )
                    logger = rs.LossLogger(
                        loss_weights={
                            "Good Decoder": 1,
                            "Bad Decoder": 1,
                            "Certificate Decoder": 1,
                            "Good Encoder L1 penalty": 3e-3,
                            "Bad Encoder L1 penalty": 3e-3,
                            "Correlation penalty": 0.1,
                        }
                    )

                    rs.train(
                        model,
                        trainloader,
                        optimizer,
                        num_epochs[routing_label],
                        rs.calculate_split_losses,
                        logger,
                        routing_pct=partial(routing_fn, routing_pct=routing_pct),
                        use_pbar=False,
                    )

                    # Store the extreme models
                    if model_idx == 0 and routing_pct in [0, 1]:
                        models[routing_label].append(model)

                    df_test = ablations.evaluate(model, testloader)
                    df_train = ablations.evaluate(model, trainloader)
                    df_test["train"] = False
                    df_train["train"] = True
                    res = pd.concat((df_test, df_train))
                    res["routing_pct"] = routing_pct
                    res["routing_style"] = routing_label
                    res["model_idx"] = model_idx
                    results.append(res)

                    progress_bar.update(1)
        progress_bar.close()
        logger.plot()

        df = pd.concat(results)
        df.to_csv(SAVE_PATH, index=False)

    groups = ["train", "is_bad", "routing_style", "routing_pct"]
    agg_with_model = (
        df.groupby(groups + ["model_idx"]).agg({"loss": "mean"}).reset_index()
    )

    agg = (
        agg_with_model.groupby(groups)
        .agg({"loss": ["mean", "std", "count", ablations.q5, ablations.q95]})
        .reset_index()
    )
    agg.columns = [
        "train",
        "is_bad",
        "routing_style",
        "routing_pct",
        "loss_mean",
        "loss_std",
        "count",
        "loss_q5",
        "loss_95",
    ]
    ci_width = 1.96 * agg["loss_std"] / np.sqrt(agg["count"])
    agg["ci_ub"] = agg["loss_mean"] + ci_width
    agg["ci_lb"] = agg["loss_mean"] - ci_width
    to_plot = agg[(~agg.train) & (agg.is_bad)]

    # %%
    fig, ax = plt.subplots()

    for label in routing_style.keys():
        subset = agg[(~agg.train) & (agg.is_bad) & (agg.routing_style == label)]
        ax.plot(
            subset.routing_pct,
            subset.loss_mean,
            label=label,
            ls="--" if label == "pretrain fully routed" else "-",
        )
        ax.fill_between(subset.routing_pct, subset.ci_lb, subset.ci_ub, alpha=0.15)
        # ax.fill_between(
        #     subset.routing_pct, subset.loss_q5, subset.loss_95, alpha=0.06, color="gray"
        # )

    ax.set_xlabel("% of training points with routing applied")
    ax.set_ylabel("Test validation loss on 0-4")
    # ax.axhline(to_plot.loss_mean.max(), c="gray", ls=":", alpha=0.8)
    # ax.axhline(to_plot.loss_mean.min(), c="gray", ls=":", alpha=0.8)
    ax.legend()
    title = "Test loss of Certificate under different amounts of gradient routing"
    # title += "\n(Highlighted region shows 5th and 95th percentiles)"
    ax.set_title(title)
    ax.grid(True, alpha=0.5)
    plt.savefig(
        os.path.join(figures_path, "mnist_partial_oversight.pdf"),
        pad_inches=0,
        bbox_inches="tight",
    )

# %%
