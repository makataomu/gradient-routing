# %%
import os
import sys
from copy import deepcopy

import pandas as pd
import torch

import projects.mnist.representation_splitting as rs
from factored_representations.utils import get_gpu_with_most_memory

"""
$(pdm venv activate) && python projects/mnist/runs_for_paper_main.py

$(pdm venv activate) && python projects/mnist/runs_for_paper_main.py dry_run
"""


def train_from_scratch(
    model_kwargs,
    loss_weights,
    trainloader,
    testloader,
    num_epochs: int,
    num_models: int,
):
    device = trainloader.device
    print(f"Training on {device}...")
    results_list = []
    for run_idx in range(num_models):
        model = rs.SplitAutoencoder(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
        logger = rs.LossLogger(loss_weights)
        rs.train(
            model,
            trainloader,
            optimizer=optimizer,
            num_epochs=num_epochs,
            loss_getter=rs.calculate_split_losses,
            loss_logger=logger,
            routing_pct=lambda ep: 1,
        )
        res = rs.evaluate(model, testloader)
        res["is_bad"] = rs.is_bad_data(None, labels=res.index)
        res["Decoder"] = res["decoder_bad"] * res["is_bad"] + res["decoder_good"] * (
            ~res["is_bad"]
        )
        res.rename(columns={"certificate_decoder": "Certificate"}, inplace=True)
        res["run_idx"] = run_idx
        res["label"] = res.index
        results_list.append(res)
    all_res = pd.concat(results_list)
    return all_res


if __name__ == "__main__":
    DRY_RUN = len(sys.argv) > 1 and sys.argv[1] == "dry_run"
    if DRY_RUN:
        print("DRY RUN")

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(parent_dir, "results")

    device = get_gpu_with_most_memory()
    trainset, trainloader, testset, testloader = rs.get_mnist_data(device)

    num_epochs = 2 if DRY_RUN else 200
    num_models = 2 if DRY_RUN else 20

    all_results = []

    """ MAIN RESULTS """

    # GRADIENT ROUTING -> split representations
    loss_weights = {
        "Good Decoder": 1,
        "Good Decoder (bad data)": 0,
        "Bad Decoder": 1,
        "Certificate Decoder": 1,
        "Bad Encoder L1 penalty": 3e-3,
        "Good Encoder L1 penalty": 3e-3,
        "Correlation penalty": 0.1,
    }

    model_kwargs = dict(
        hidden_layer_sizes=[2048, 512],
        hidden_size=32,
        split_decoders=False,
        use_split_encoding=True,
    )

    df = train_from_scratch(
        model_kwargs=model_kwargs,
        loss_weights=loss_weights,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=num_epochs,
        num_models=num_models,
    )
    df["setting"] = "Gradient routing"
    all_results.append(df)

    # WITH GOOD DECODER TRAINING ON ALL -> still splits!
    loss_weights_good_train_all = deepcopy(loss_weights)
    loss_weights_good_train_all["Good Decoder (bad data)"] = 1
    loss_weights_good_train_all["Good Encoder L1 penalty"] = 2e-2

    df = train_from_scratch(
        model_kwargs=model_kwargs,
        loss_weights=loss_weights_good_train_all,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=num_epochs,
        num_models=num_models,
    )
    df["setting"] = "Gradient routing, bottom half encoding trained on 0-9"
    all_results.append(df)

    # WITHOUT GRADIENT ROUTING -> no split representations
    model_kwargs_no_gr = deepcopy(model_kwargs)
    model_kwargs_no_gr["use_split_encoding"] = False

    df = train_from_scratch(
        model_kwargs=model_kwargs_no_gr,
        loss_weights=loss_weights,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=num_epochs,
        num_models=num_models,
    )
    df["setting"] = "No gradient routing, with regularization"
    all_results.append(df)

    # WITHOUT GRADIENT ROUTING, WITHOUT TRAINING ON 0-4 -> generalizes to 0-4
    model_kwargs_no_gr = deepcopy(model_kwargs)
    model_kwargs_no_gr["use_split_encoding"] = False

    loss_weights_no_gr_no_bad = deepcopy(loss_weights)
    loss_weights_no_gr_no_bad["Bad Decoder"] = 0
    loss_weights_no_gr_no_bad["Bad Encoder L1 penalty"] = 0
    loss_weights_no_gr_no_bad["Good Encoder L1 penalty"] = 0
    loss_weights_no_gr_no_bad["Correlation penalty"] = 0

    df = train_from_scratch(
        model_kwargs=model_kwargs_no_gr,
        loss_weights=loss_weights_no_gr_no_bad,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=num_epochs,
        num_models=num_models,
    )
    df["setting"] = "No gradient routing, no regularization, trained on 5-9 only"
    all_results.append(df)

    # WITHOUT GRADIENT ROUTING, WITH L1, WITHOUT TRAINING ON 0-4 -> splits!
    model_kwargs_no_gr = deepcopy(model_kwargs)
    model_kwargs_no_gr["use_split_encoding"] = False

    loss_weights_no_gr_no_bad = deepcopy(loss_weights)
    loss_weights_no_gr_no_bad["Bad Decoder"] = 0
    loss_weights_no_gr_no_bad["Bad Encoder L1 penalty"] = 1e-3
    loss_weights_no_gr_no_bad["Good Encoder L1 penalty"] = 1e-3
    loss_weights_no_gr_no_bad["Correlation penalty"] = 0

    df = train_from_scratch(
        model_kwargs=model_kwargs_no_gr,
        loss_weights=loss_weights_no_gr_no_bad,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=num_epochs,
        num_models=num_models,
    )
    df["setting"] = "No gradient routing, L1 penalty 1e-3, trained on 5-9 only"
    all_results.append(df)

    # WITHOUT GRADIENT ROUTING, WITHOUT REGULARIZATION -> serves as baseline
    model_kwargs_no_gr = deepcopy(model_kwargs)
    model_kwargs_no_gr["use_split_encoding"] = False

    loss_weights_no_gr_no_bad = deepcopy(loss_weights)
    loss_weights_no_gr_no_bad["Bad Encoder L1 penalty"] = 0
    loss_weights_no_gr_no_bad["Good Encoder L1 penalty"] = 0
    loss_weights_no_gr_no_bad["Correlation penalty"] = 0

    df = train_from_scratch(
        model_kwargs=model_kwargs_no_gr,
        loss_weights=loss_weights_no_gr_no_bad,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=num_epochs,
        num_models=num_models,
    )
    df["setting"] = "No gradient routing, no regularization"
    all_results.append(df)

    # SEPARATE DECODERS -> still splits!
    model_kwargs_split_decoders = deepcopy(model_kwargs)
    model_kwargs_split_decoders["split_decoders"] = True

    df = train_from_scratch(
        model_kwargs=model_kwargs_split_decoders,
        loss_weights=loss_weights,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=num_epochs,
        num_models=num_models,
    )
    df["setting"] = "Gradient routing, separate Decoders"
    all_results.append(df)

    # WITHOUT CORRELATION PENALTY -> still splits (but worse)!
    loss_weights_no_corr = deepcopy(loss_weights)
    loss_weights_no_corr["Correlation penalty"] = 0

    df = train_from_scratch(
        model_kwargs=model_kwargs,
        loss_weights=loss_weights_no_corr,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=num_epochs,
        num_models=num_models,
    )
    df["setting"] = "Gradient routing, no correlation penalty"
    all_results.append(df)

    # WITHOUT REGULARIZATION -> divergence
    loss_weights_no_reg = deepcopy(loss_weights)
    loss_weights_no_reg["Bad Encoder L1 penalty"] = 0
    loss_weights_no_reg["Good Encoder L1 penalty"] = 0
    loss_weights_no_reg["Correlation penalty"] = 0

    df = train_from_scratch(
        model_kwargs=model_kwargs,
        loss_weights=loss_weights_no_reg,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=num_epochs,
        num_models=num_models,
    )
    df["setting"] = "Gradient routing, no regularization"
    all_results.append(df)

    # WITHOUT REGULARIZATION, SEPARATE DECODERS -> no splitting
    model_kwargs_split_decoders = deepcopy(model_kwargs)
    model_kwargs_split_decoders["split_decoders"] = True

    df = train_from_scratch(
        model_kwargs=model_kwargs_split_decoders,
        loss_weights=loss_weights_no_reg,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=num_epochs,
        num_models=num_models,
    )
    df["setting"] = "Gradient routing, no regularization, separate Decoders"
    all_results.append(df)

    all_df = pd.concat(all_results)

    filename = "mnist_main_results_dry_run.csv" if DRY_RUN else "mnist_main_results.csv"
    all_df.to_csv(os.path.join(results_dir, filename), index=False)
