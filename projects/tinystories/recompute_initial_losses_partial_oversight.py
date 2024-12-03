# %%
import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import projects.tinystories.analysis_tools as atools
import shared_configs.model_store as model_store
from factored_representations import string_utils
from factored_representations.utils import get_gpu_with_most_memory
from projects.tinystories.retraining_evals import eval_model

words_to_localize = [
    "tree",
    "trees",
    "forest",
    "forests",
    "woodland",
    "woodlands",
]


def get_model_save_path(model_save_name):
    prefixes_to_try = [
        "bulk_runs_for_paper",
        "bulk_runs_for_paper_demix",
        "rmu_partial_oversight",
        "data_recomputed_era",
    ]
    path_to_return = ""
    to_try = ""
    for prefix in prefixes_to_try:
        to_try = model_store.MODEL_STORE_PATH / prefix / f"{model_save_name}.pt"
        if os.path.exists(to_try):
            path_to_return = f"{prefix}/{model_save_name}"
            break
    assert path_to_return != "", f"Could not find model {model_save_name}"

    assert os.path.exists(to_try), f"Path {path_to_return} does not exist"
    return path_to_return


NUM_VAL_TO_USE = 5000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_gpus", type=int)
    parser.add_argument("gpu_on", type=int)
    args = parser.parse_args()
    gpu_on = args.gpu_on
    num_gpus = args.num_gpus
    os.makedirs("initial_losses_recomputed", exist_ok=True)

    dev = get_gpu_with_most_memory()
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data_recomputed_era")

    experiment_prefix = ["e11-o0.95_rmu"]  # ["edemix-partial", "e11-o"]

    """
    Relearning and basic unlearning results
    """

    df = atools.read_all_data(data_dir, experiment_prefix)
    validation_stories = string_utils.load_dataset_with_split(
        "delphi-suite/stories", "validation", max_stories=None
    )
    random.shuffle(validation_stories)
    truncated_validation_stories = string_utils.truncate_stories_by_chars(
        validation_stories, 1029
    )
    forget_validation_stories, retain_validation_stories = (
        string_utils.split_and_label_stories_by_concept(
            truncated_validation_stories, words_to_localize
        )
    )
    forget_validation_stories = forget_validation_stories[:NUM_VAL_TO_USE]
    retain_validation_stories = retain_validation_stories[:NUM_VAL_TO_USE]

    unique_models = df["model_save_name"].unique()
    unique_models = np.array_split(unique_models, num_gpus)[gpu_on]
    for model_name in tqdm(unique_models):
        model_save_path = get_model_save_path(model_name)
        print(f"Model {model_name} found at {model_save_path}")
        loaded_model = model_store.load_model(
            model_save_path,
            "roneneldan/TinyStories-28M",
            dev,
        )
        forget_loss = eval_model(loaded_model, forget_validation_stories, 256)["loss"]
        retain_loss = eval_model(loaded_model, retain_validation_stories, 256)["loss"]
        print(f"Forget loss: {forget_loss}, retain loss: {retain_loss}")
        model_oversight_pct = df[df["model_save_name"] == model_name][
            "oversight_pct"
        ].iloc[0]
        model_save_name = model_name
        run_type = df[df["model_save_name"] == model_name]["run_type"].iloc[0]
        random_seed = df[df["model_save_name"] == model_name]["random_seed"].iloc[0]
        l1_coeff = -1  # we never use this
        df_this_model_retrained = pd.DataFrame(
            {
                "run_type": run_type,
                "model_save_name": model_save_name,
                "random_seed": random_seed,
                "l1_coeff": l1_coeff,
                "oversight_pct": model_oversight_pct,
                "initial_forget_loss": forget_loss,
                "initial_retain_loss": retain_loss,
            },
            index=[0],
        )
        df_this_model_retrained.to_csv(
            f"initial_losses_recomputed/{model_name}.csv", index=False
        )

# %%
