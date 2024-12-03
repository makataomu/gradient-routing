# %%

import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
import torchvision
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from factored_representations.utils import get_gpu_with_most_memory
from projects.cifar100.cifar_100_models import RoutedResNet


class CIFAR100Dataset(t.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx]["img"]
        fine_label = self.hf_dataset[idx]["fine_label"]
        coarse_label = self.hf_dataset[idx]["coarse_label"]

        if self.transform:
            image = self.transform(image)

        return image, fine_label, coarse_label


def get_dataloaders():
    # Data augmentation and normalization
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Datasets
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    # DataLoaders
    trainloader = DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True
    )

    return trainloader, testloader


@t.inference_mode()
def get_classwise_accuracy_by_output(model, testloader):
    num_classes = 100
    correct_per_class = {
        "routed_top": t.zeros(num_classes),
        "routed_bot": t.zeros(num_classes),
        "top_certificate": t.zeros(num_classes),
        "bot_certificate": t.zeros(num_classes),
    }
    total_per_class = t.zeros(num_classes)
    device = next(model.parameters()).device
    with t.inference_mode():
        for batch_idx, (x, labels) in tqdm.tqdm(
            enumerate(testloader), total=len(testloader)
        ):
            for label in labels:
                total_per_class[label] += 1

            x = x.to(device)
            labels = labels.to(device)

            outs, _ = model(x)
            for out_name, out_val in outs.items():
                is_correct = out_val.argmax(dim=1) == labels
                for label_idx, correct in zip(labels, is_correct):
                    correct_per_class[out_name][label_idx] += correct.item()

    accuracy_by_class = {"label": range(num_classes)}
    accuracy_by_class |= {
        out_name: correct_per_class[out_name] / total_per_class
        for out_name in correct_per_class
    }
    return accuracy_by_class


def evaluate_by_routing_location(model, testloader, route_fn):
    accuracy_by_class = get_classwise_accuracy_by_output(model, testloader)
    df = pd.DataFrame(accuracy_by_class)
    df["routed_to"] = route_fn(df["label"]).map({True: "top", False: "bot"})
    del df["label"]
    summary = df.groupby("routed_to").mean()
    return summary


if __name__ == "__main__":
    device = get_gpu_with_most_memory()
    print(device)

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    for dir_name in ["models", "figures", "results"]:
        os.makedirs(os.path.join(parent_dir, dir_name), exist_ok=True)

    trainloader, testloader = get_dataloaders()

    """
    python projects/cifar100/cifar_100_representation_splitting.py
    """

    run_name = "resnet34_routing_l1_3e2_split_with_conv_before_and_after_at_14"

    num_epochs = 200
    eval_freq = 250

    loss_weights = {
        "loss_top": 1,
        "loss_bot": 1,
        "loss_top_cert": 1,
        "loss_bot_cert": 1,
        "l1": 3e-2,
        "l2": 0,
    }

    metrics = defaultdict(list)
    eval_metrics = defaultdict(list)

    num_classes = 100
    model = RoutedResNet(
        num_blocks=[3, 4, 6, 3],
        split_at_block=14,
        num_classes=num_classes,
        use_split_encoder=True,
        use_shared_decoder=True,
    ).to(device)

    optimizer = t.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    def route_fn(labels):
        return labels < 50

    model.train()
    start_time = time.time()
    for epoch in tqdm.tqdm(range(num_epochs)):
        for batch_idx, (x, labels) in enumerate(trainloader):
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            losses = model.get_losses(x, labels, route_fn, reduction="mean")

            loss = t.tensor(0.0, device=device)
            for loss_name, loss_val in losses.items():
                assert loss_name in loss_weights
                weighted_loss_val = loss_val * loss_weights[loss_name]
                loss += weighted_loss_val
                metrics[loss_name].append(weighted_loss_val.item())
            loss = loss * 0.25  # since there are four output modules trained
            metrics["loss"].append(loss.item())
            metrics["duration"].append(time.time() - start_time)

            loss.backward()
            optimizer.step()
        scheduler.step()

    t.save(model.state_dict(), os.path.join(parent_dir, "models", f"{run_name}.pt"))

    # Training stats
    df_train = pd.DataFrame(metrics)
    df_train.to_csv(
        os.path.join(parent_dir, "results", f"train_metrics_{run_name}.csv")
    )
    fig, ax = plt.subplots()
    df_train.rolling(196).mean().plot(ax=ax)
    ax.set_xlabel("Batch")
    ax.set_title("Training metrics")

    # Eval accuracy
    accuracy_by_class = get_classwise_accuracy_by_output(model, testloader)
    summary = evaluate_by_routing_location(model, testloader, route_fn)
    del summary["routed_bot"]
    summary.to_csv(os.path.join(parent_dir, "results", f"summary_{run_name}.csv"))

    fix, ax = plt.subplots()
    ax.set_ylabel("Validation accuracy")
    summary.plot(kind="bar", ax=ax, color=["C4", "C3", "C0"], legend=False)
    plt.xticks(rotation=0)
    ax.set_xlabel("")
    ax.set_title("Gradient routing applied to ResNet channels on CIFAR100")
    ax.legend(loc="lower right")

    # colors = ["C4", "C3", "C0"]
    # hatches = [None, "///", "\\\\"]
    # hatch_colors = [None, (1, 0.2, 0.2, 1), (0.2, 0.2, 1, 1)]
    # labels = ["Decoder", "Certificate (top)", "Certificate (bot)"]

    # patches = []
    # for idx, bar in enumerate(ax.patches):
    #     color_idx = idx // 10
    #     color = colors[color_idx]
    #     hatch = hatches[color_idx]
    #     hatch_color = hatch_colors[color_idx]
    #     bar.set_facecolor(color)
    #     if hatch is not None:
    #         bar.set_edgecolor(hatch_color)
    #         bar.set_linewidth(0)
    #         bar.set_hatch(hatch)
    #     if idx % 10 == 0:
    #         patch = mpatches.Patch(
    #             facecolor=color,
    #             hatch=hatch,
    #             edgecolor=hatch_color,
    #             linewidth=0,
    #             label=labels[color_idx],
    #         )
    #         patches.append(patch)

    # ax.legend(handles=patches)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    plt.savefig(
        os.path.join(parent_dir, "figures", f"cifar_performance_{run_name}.pdf"),
        bbox_inches="tight",
    )
