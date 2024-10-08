# type: ignore
# %%
import glob
import json
import os

import numpy as np
import pandas as pd

"""
Helper functions for working with saved Tinystories results. Assumes a particular format.

It's mostly about reading in the CSVs and JSONs into a single unified Pandas DataFrame.
"""

kwargs = {
    "forget": {"color": "C3", "linestyle": "-"},
    "retain": {"color": "C0", "linestyle": "--"},
}


def get_ci_width(data):
    return 1.96 * np.std(data) / np.sqrt(len(data))


q5 = lambda x: np.quantile(x, q=0.05)
q95 = lambda x: np.quantile(x, q=0.95)
agg_fns = ["mean", get_ci_width, q5, q95]

relearning_variable_cols = ["num_stories", "update_step", "forget_loss", "retain_loss"]


def read_relearning_data(data_dir, experiment_prefix):
    relearning_filepaths = glob.glob(
        os.path.join(data_dir, f"{experiment_prefix}*.csv")
    )
    assert (
        len(relearning_filepaths) > 0
    ), f"No files found with prefix {experiment_prefix} in {data_dir}"
    print(f"Reading {len(relearning_filepaths)} relearning CSVs...")

    dfs = []
    for path in relearning_filepaths:
        dfs.append(pd.read_csv(path, index_col=0))

    df = pd.concat(dfs)
    return df


def read_pre_post_ablation_data(data_dir, experiment_prefix):
    pre_post_ablation_filepaths = glob.glob(
        os.path.join(data_dir, f"{experiment_prefix}*.json")
    )
    print(f"Reading {len(pre_post_ablation_filepaths)} pre/post-ablation JSONs...")
    records = []
    for path in pre_post_ablation_filepaths:
        filename = os.path.split(path)[-1].split(".")[0]
        filename_suffix = "_pre_post_ablation"
        run_name = filename[: -len(filename_suffix)]

        with open(path, "r") as file:
            data = json.load(file)
            for step_idx, key in enumerate(["pre_ablation", "post_ablation"]):
                record = [
                    run_name,
                    f"{step_idx+1}. {key}",
                    -2 + step_idx,
                    data[key]["forget_loss"],
                    data[key]["retain_loss"],
                ]
                records.append(record)

    col_names = [
        "model_save_name",
        "experiment_step",
        "update_step",
        "forget_loss",
        "retain_loss",
    ]

    df = pd.DataFrame(
        records,
        columns=col_names,  # type: ignore
    )
    return df


def read_all_data(data_dir, experiment_prefix):
    df1 = read_relearning_data(data_dir, experiment_prefix)
    df2 = read_pre_post_ablation_data(data_dir, experiment_prefix)

    df1_saves = set(df1.model_save_name)
    df2_saves = set(df2.model_save_name)
    for save in df1_saves:
        if save not in df2_saves and "_rmu_" not in save:
            print(f"Warning: {save} not found in pre/post ablation data")
    for save in df2_saves:
        if save not in df1_saves:
            print(f"Warning: {save} not found in relearning data")

    id_cols = [col for col in df1.columns if col not in relearning_variable_cols]
    df2 = df2.merge(df1[id_cols].drop_duplicates(), how="inner", on="model_save_name")
    df2["num_stories"] = 0

    df3 = (
        df1.sort_values("forget_loss")
        .groupby("model_save_name")[df1.columns]
        .apply(pd.DataFrame.head, n=1)
    )

    df1["experiment_step"] = "3. relearning"
    df3["experiment_step"] = "4. lowest_forget"

    df = pd.concat((df1, df2, df3), ignore_index=True)
    return df


if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data")
    experiment_prefix = "hyp11_"
    df = read_all_data(data_dir, experiment_prefix)
