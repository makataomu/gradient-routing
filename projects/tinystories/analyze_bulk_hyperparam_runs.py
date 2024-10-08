# type: ignore
# %%
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import projects.tinystories.analysis_tools as atools

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data")

    experiment_prefix = "hyp12_"

    """
    Relearning and basic unlearning results
    """

    df_all = atools.read_all_data(data_dir, experiment_prefix)
    df_pre = df_all[df_all.experiment_step == "1. pre_ablation"]

    min_losses = (
        df_all[df_all["experiment_step"] == "3. relearning"]
        .groupby("model_save_name")
        .agg({"forget_loss": "min"})
    )
    df_all = df_all.merge(min_losses, on="model_save_name", suffixes=("", "_min"))

    df = df_all[(df_all.update_step == 0) & (df_all.num_stories == 64)].copy()
    df["loss_diff"] = df["forget_loss_min"] - df["retain_loss"]

    df = df.sort_values("loss_diff")

    def linear_regression(X, y, penalty=0.01):
        X = np.hstack([np.ones_like(X[:, [1]]), X])
        return np.linalg.solve(X.T @ X + penalty * np.eye(X.shape[1]), X.T @ y)

    vars = ["num_layers_to_mask", "d_mlp_to_expand", "original_dim_lr_target"]
    X = df_pre[vars].values
    y = df["loss_diff"].values

    coef = linear_regression(X, y)
    print([f"{var}: {c:0.5f}" for var, c in zip(vars, coef[1:])])

    fig, ax = plt.subplots()
    ax.set_xlabel("Min. forget loss under retraining")
    ax.set_ylabel("Retain loss")
    ax.scatter(df["forget_loss_min"], df["retain_loss"], s=85)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_equals_y = np.linspace(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]), 2)
    ax.plot(x_equals_y, x_equals_y, ls=":", c="black")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid("on", alpha=0.3)
    # ax.invert_yaxis()
    nudge_x = (xlim[1] - xlim[0]) / 50
    nudge_y = (ylim[1] - ylim[0]) / 50

    for _, row in df.iterrows():
        x = row["forget_loss_min"]
        y = row["retain_loss"]
        a, b, c = row[
            ["num_layers_to_mask", "d_mlp_to_expand", "original_dim_lr_target"]
        ]

        ax.text(x + nudge_x, y - nudge_y, b, fontsize=7, alpha=0.8)
        ax.text(
            x - 3 * nudge_x,
            y + nudge_y,
            f"{c:0.2f}",
            fontsize=7,
            alpha=0.8,
            c="red",
        )

    for _, row in df.iterrows():
        x = row["forget_loss_min"]
        y = row["retain_loss"]
        ax.text(
            x,
            y,
            row["num_layers_to_mask"],
            horizontalalignment="center",
            verticalalignment="center",
            c="white",
            fontsize=9,
        )

    ax.set_title(
        "Hyperparameter sweep on Tinystories-28M:\n(# layers, MLP expansion, forget LR in original dims)"
    )

    """
    Pre- vs. post-ablation results
    """

    plot_full_relearning = False
    if plot_full_relearning:
        stage_rows = (df_all.update_step < 10) & (
            df_all.experiment_step != "4. lowest forget"
        )
    else:
        stage_rows = (df_all.experiment_step != "3. relearning") | (
            (df_all.experiment_step == "3. relearning") & (df_all.update_step == 0)
        )
    four_stages = df_all[stage_rows].sort_values("update_step")

    model_save_names = df.model_save_name.values
    assert len(model_save_names) == len(set(model_save_names))

    backgcolor = {
        seed: mcolors.to_rgba(f"C{idx}", alpha=0.05)
        for idx, seed in enumerate(df.random_seed.unique())
    }

    ncols = 4
    nrows = round(0.5 + len(model_save_names) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharey=True, figsize=(10, nrows * 2), squeeze=False
    )
    for idx, save_name in enumerate(model_save_names):
        stages = four_stages[four_stages.model_save_name == save_name]
        hyper = stages.iloc[0][
            ["num_layers_to_mask", "d_mlp_to_expand", "original_dim_lr_target"]
        ].values
        num_layers_to_mask, d_mlp_to_expand, original_dim_lr_target = hyper

        ax = axes[idx // ncols, idx % ncols]
        ax.set_title(
            f"n_layer:{num_layers_to_mask}/mlp:{d_mlp_to_expand}/lr:{original_dim_lr_target:0.2f}"
        )

        xvals = stages["update_step"] if plot_full_relearning else range(4)
        ax.plot(
            xvals,
            stages["forget_loss"].values,
            label="forget",
            **atools.kwargs["forget"],
            lw=2,
        )
        ax.plot(
            xvals,
            stages["retain_loss"],
            label="retain",
            **atools.kwargs["retain"],
            lw=2,
        )
        ax.set_facecolor(backgcolor[stages.iloc[0]["random_seed"]])
        ax.set_xticks(xvals)
        ax.set_xticklabels(["pre", "post", "coh", "rel"] + [" "] * (len(stages) - 4))
        ax.grid("on", alpha=0.7)
        ax.set_ylim(
            four_stages["forget_loss"].min() - 0.05,
            2.5,
        )

    axes[0, 0].legend()
    plt.tight_layout()
