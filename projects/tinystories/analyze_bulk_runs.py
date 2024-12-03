# type: ignore
# %%
import os

import matplotlib.pyplot as plt

import projects.tinystories.analysis_tools as atools

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data_recomputed_era")
    figure_dir = os.path.join(parent_dir, "figures")

    experiment_prefix = ["e11_", "ejacob_"]

    """
    Relearning and basic unlearning results
    """

    df = atools.read_all_data(data_dir, experiment_prefix)
    agg = (
        df[df.experiment_step != "4. lowest_forget"]
        .groupby(["experiment_step", "run_type", "num_stories", "update_step"])
        .agg({"forget_loss": ["count", *atools.agg_fns], "retain_loss": atools.agg_fns})
        .reset_index()
    )
    agg.columns = [
        "experiment_step",
        "run_type",
        "num_stories",
        "update_step",
        "num_runs",
        "forget_loss_mean",
        "forget_loss_ci_width",
        "forget_loss_q5",
        "forget_loss_q95",
        "retain_loss_mean",
        "retain_loss_ci_width",
        "retain_loss_q5",
        "retain_loss_q95",
    ]

    # alignment tax of L1
    l1_tax = (
        df[
            df.run_type.isin(["base", "expanded_base"])
            & (df.num_stories == 1)
            & (df.update_step == 0)
            & (df.experiment_step == "3. relearning")
        ]
        .groupby("run_type")
        .agg({"forget_loss": [*atools.agg_fns]})
    )

    # Get metrics for paper
    posttrain = (agg.num_stories == 64) & (agg.update_step == 0)
    coherence_agg = agg[(agg.run_type.isin(["ERAC", "expanded_base"])) & posttrain]

    def compare_metric(run_types: list[str], metric: str):
        for run_type in run_types:
            sub = agg[posttrain & (agg.run_type == run_type)]
            assert len(sub) > 0, f"No data for {run_type}"
            mean = sub[metric + "_mean"].item()
            ci = sub[metric + "_ci_width"].item()
            print(f"{run_type}: {mean:0.2f}(±{ci:0.2f})")

    num_runs = len(df[df.run_type == "ERAC"].model_save_name.unique())

    print()
    print(f"Based on {num_runs} run sets...")

    print("Shallow unlearning - forget loss")
    compare_metric(["ERAC", "base", "demix"], "forget_loss")
    print()

    print("Alignment tax - retain loss")
    compare_metric(["ERAC", "base", "demix"], "retain_loss")
    print()

    print("Deep unlearning - average minimum forget loss over retraining")
    print()

    relearn_run_types = ["pure", "ERAC", "rmu", "demix"]
    num_stories_settings = [4, 16, 64]
    df_relearn = df[
        (df.experiment_step == "3. relearning")
        & df.run_type.isin(relearn_run_types)
        & (df.num_stories.isin(num_stories_settings))
    ]

    # THIS IS FOR TABLE
    min_loss_every_run = (
        df_relearn.groupby(["num_stories", "run_type", "model_save_name"])
        .agg({"forget_loss": "min"})
        .reset_index()
    )  # take minimum regardless of update step
    avg_min_losses = (
        min_loss_every_run.groupby(["num_stories", "run_type"])
        .agg({"forget_loss": ["mean", atools.get_ci_width]})
        .reset_index()
    )  # take the average of the minimums
    avg_min_losses.columns = ["num_stories", "run_type", "min_loss", "ci_width_95"]

    # THIS IS FOR FIGURE
    avg_loss_each_step = (
        (df_relearn.groupby(["num_stories", "run_type", "update_step"]))
        .agg({"forget_loss": ["mean", atools.get_ci_width]})
        .reset_index()
    )
    avg_loss_each_step.columns = [
        "num_stories",
        "run_type",
        "update_step",
        "forget_loss_mean",
        "ci_width_95",
    ]

    avg_min_losses["Loss"] = (
        avg_min_losses["min_loss"].apply(lambda x: f"{x:.2f}")
        + " ("
        + (avg_min_losses["ci_width_95"].apply(lambda x: f"{x:.2f}"))
        + ")"
    )
    max_idx = avg_min_losses.groupby("num_stories")["min_loss"].idxmax()

    for_paper = (
        avg_min_losses.pivot(index="num_stories", columns="run_type", values="Loss")
        .reset_index()
        .rename(
            columns={
                "num_stories": "Stories",
                "pure": "Pure",
                "ERAC": "ERA",
                "rmu": "RMU",
                "demix": "DEMix",
            }
        )
    )
    for_paper = for_paper[["Stories", "Pure", "ERA", "RMU", "DEMix"]]
    print(for_paper.to_latex(index=False))

    """
    Plot
    """
    colors = {"base": "black", **atools.colors}

    linestyles = {"base": "--", **atools.linestyles}

    labels = {
        "base": "Base",
        "ERAC": "ERA",
        "pure": "Pure",
        "rmu": "RMU",
        "demix": "DEMix",
    }

    # %%
    num_settings = len(num_stories_settings)
    fig, axes = plt.subplots(
        ncols=num_settings,
        sharey=True,
        figsize=(num_settings * 2.5, 2.7),
        constrained_layout=True,
        dpi=300,
    )
    fig.suptitle("Forget set relearnability")

    baseline_mean, baseline_ci = (
        df[
            (df.experiment_step == "3. relearning")
            & (df.run_type == "base")
            & (df.num_stories == 64)
        ]
        .agg({"forget_loss": ["mean", atools.get_ci_width]})
        .values[:, 0]
    )

    for story_idx, num_stories in enumerate(num_stories_settings):
        ax = axes[story_idx]  # type: ignore
        ax.axhline(baseline_mean, c="black", ls="--", label="Base")

        # ax.fill_between(baseline_mean - baseline_ci, baseline_mean + baseline_ci, alpha=0.1, color="black")

        ax.set_title(f"{num_stories} stories" if num_stories > 1 else "1 story")
        for run_idx, run_type in enumerate(relearn_run_types):
            subs = (avg_loss_each_step.run_type == run_type) & (
                avg_loss_each_step.num_stories == num_stories
            )
            runs = avg_loss_each_step[subs].sort_values("update_step")
            color = colors[run_type]
            ax.plot(
                runs.update_step,
                runs.forget_loss_mean,
                label=labels[run_type],
                c=color,
                ls=linestyles[run_type],
            )
            ax.fill_between(
                runs.update_step,
                runs.forget_loss_mean - runs.ci_width_95,
                runs.forget_loss_mean + runs.ci_width_95,
                alpha=0.1,
                color=color,
            )

            ax.grid(alpha=0.2)
            ax.margins(x=0)
    axes[1].set_xlabel("Update step", fontsize=12)
    axes[0].set_ylabel("Validation forget loss", fontsize=12)  # type: ignore
    axes[0].set_ylim(None, 2.5)  # type: ignore
    axes[-1].legend(loc="center right", bbox_to_anchor=(1.8, 0.5), fontsize=11)

    plt.savefig(os.path.join(figure_dir, "tinystories_relearnability.pdf"))

    # %%
    """
    Pre- vs. post-ablation results
    """

    pre_ablation = df.experiment_step == "1. pre_ablation"
    post_ablation = df.experiment_step == "2. post_ablation"
    coherence = df.experiment_step == "3. coherence"

    df_ablation = df[
        df.update_step.isin([-2, -1, 0]) & (df.experiment_step != "4. lowest_forget")
    ]
    agg_fns = ["mean", atools.get_ci_width]
    step_agg = (
        df_ablation.groupby(["run_type", "experiment_step"])
        .agg({"forget_loss": agg_fns, "retain_loss": agg_fns})
        .reset_index()
    )
    step_agg.columns = [
        "run_type",
        "experiment_step",
        "forget_loss_mean",
        "forget_loss_ci_width",
        "retain_loss_mean",
        "retain_loss_ci_width",
    ]
    # step_agg.loc[(step_agg.run_type == "ERAC") & (step_agg.experiment_step == "1. pre_ablation"), "forget_loss_mean"] = 1.574

    fig, ax_erac = plt.subplots(ncols=1, figsize=(3.5, 2.7), dpi=300)  # type: ignore

    ax_erac.set_ylabel("Validation loss", fontsize=12)
    ax_erac.set_title("Routing explains ERA's performance")

    colors = {"forget": "C3", "retain": "C0"}
    demix_colors = {"forget": "#FF4D4D", "retain": "#4169E1"}
    linestyles = {"forget": "-", "retain": "--"}
    plt.xticks(fontsize=12)
    for run_type, ax in zip(["ERAC", "expanded_base"], [ax_erac, ax_erac, ax_erac]):
        for loss_type in ["forget", "retain"]:
            subset = step_agg.run_type == run_type
            if run_type == "ERAC":
                color = colors[loss_type]
            elif run_type == "demix":
                color = demix_colors[loss_type]
            else:
                color = "black"

            mean_loss = step_agg[subset][f"{loss_type}_loss_mean"]
            print(run_type, loss_type, mean_loss)
            ci_width = step_agg[subset][f"{loss_type}_loss_ci_width"]
            ax.plot(
                range(3),
                mean_loss,
                label=loss_type if run_type == "ERAC" else None,
                c=color,
                ls=linestyles[loss_type],
            )
            ax.fill_between(
                ["Pre-trained", "Ablated", "Fine-tuned"],
                mean_loss - ci_width,
                mean_loss + ci_width,
                alpha=0.2,
                color=color,
            )

    def get_midpoint_for_annotation(run_type):
        return (
            step_agg[
                (step_agg.run_type == run_type)
                & (step_agg.experiment_step != "1. pre_ablation")
            ][["forget_loss_mean", "retain_loss_mean"]].values.mean()
            - 0.005
        )

    # min_loss = (step_agg["mean"] - step_agg["ci_width"]).min()
    # max_loss = (step_agg["mean"] + step_agg["ci_width"]).max()
    # min_y, max_y = math.ceil(min_loss * 10) / 10, math.ceil(max_loss * 10) / 10
    # ax_erac.set_yticks(np.linspace(min_y, max_y, round((max_y - min_y) * 10 + 1)))
    ax_erac.grid(alpha=0.2)
    ax_erac.annotate(
        "ERA (Routing)",
        xy=(1.3, get_midpoint_for_annotation("ERAC")),
        ha="center",
        va="center",
        fontsize=11,
    )
    ax_erac.annotate(
        "Control (No routing)",
        xy=(1.3, get_midpoint_for_annotation("expanded_base")),
        ha="center",
        va="center",
        fontsize=11,
    )
    ax_erac.annotate(
        "DEMix",
        xy=(0.5, get_midpoint_for_annotation("demix")),
        ha="center",
        va="center",
        fontsize=11,
    )

    ax_erac.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "tinystories_ablation.pdf"))

# %%
