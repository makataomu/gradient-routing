# type: ignore
# %%
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import projects.tinystories.analysis_tools as atools

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data")

    experiment_prefix = "e11-pct-fix_"

    """
    Relearning and basic unlearning results
    """

    df = atools.read_all_data(data_dir, experiment_prefix)

    df_base = atools.read_all_data(data_dir, "e11_")
    pure_retain_loss = (
        df_base[
            (df_base.run_type == "pure")
            & (df_base.experiment_step == "2. post_ablation")
        ]
        .agg({"retain_loss": "mean"})
        .item()
    )
    base_forget_loss = (
        df_base[
            (df_base.run_type == "base")
            & (df_base.experiment_step == "2. post_ablation")
        ]
        .agg({"forget_loss": "mean"})
        .item()
    )
    pure_forget_retrained_loss = (
        df_base[
            (df_base.run_type == "pure")
            & (df_base.experiment_step == "4. lowest_forget")
        ]
        .agg({"forget_loss": "mean"})
        .item()
    )

    groupby_vars = [
        "run_type",
        "num_stories",
        "experiment_step",
        "oversight_pct",
    ]
    agg = (
        df.groupby(groupby_vars)
        .agg({"forget_loss": ["count", *atools.agg_fns], "retain_loss": atools.agg_fns})
        .reset_index()
    )
    agg.columns = [
        *groupby_vars,
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

    agg.sort_values(by="oversight_pct", inplace=True)
    agg["oversight_pct"] = agg["oversight_pct"] * 0.21
    posttrain = (agg.num_stories == 0) & (agg.experiment_step == "2. post_ablation")
    relearned = (agg.num_stories == 64) & (agg.experiment_step == "4. lowest_forget")
    preablate = (agg.num_stories == 0) & (agg.experiment_step == "1. pre_ablation")

    fig, (ax_retain, ax_forget) = plt.subplots(
        figsize=(5.5, 3.25), dpi=300, nrows=1, ncols=2, sharey=True
    )

    fig.suptitle("ERA performance depending on amount of forget data")
    ax_retain.set_title("Retain loss")
    ax_forget.set_title("Forget loss (retrained)")
    fig.supxlabel("Proportion of training data that is forget")
    ax_retain.set_ylabel("Validation loss")

    ax_forget.axhline(
        base_forget_loss,
        color="C3",
        linestyle=":",
    )

    ax_lookup = {"retain": ax_retain, "forget": ax_forget}
    subsets = {"retain": posttrain, "forget": relearned}
    baselines = {"retain": pure_retain_loss, "forget": pure_forget_retrained_loss}
    colors = {"forget": "C3", "retain": "C0"}
    linestyles = {"retain": "--", "forget": "-"}
    labels = {"retain": "retain", "forget": "forget (retrained)"}

    for run_type in ["ERAC"]:
        for loss in ["retain", "forget"]:
            ax = ax_lookup[loss]
            ax.axhline(
                baselines[loss],
                color=colors[loss],
                linestyle="--",
                alpha=0.7,
            )
            to_plot = agg[subsets[loss] & (agg.run_type == run_type)].sort_values(
                by="oversight_pct"  # type: ignore
            )
            mean = to_plot[f"{loss}_loss_mean"]
            ci_width = to_plot[f"{loss}_loss_ci_width"]
            ax.plot(
                to_plot.oversight_pct,
                mean,
                label=f"{labels[loss]}",
                c=colors[loss],
                linestyle="-",
            )
            ax.fill_between(
                to_plot.oversight_pct,
                mean - ci_width,
                mean + ci_width,
                color=colors[loss],
                alpha=0.2,
            )

            ax.grid(True, alpha=0.3)

    to_plot = agg[preablate & (agg.run_type == "ERAC")].sort_values(by="oversight_pct")
    ax_forget.plot(
        to_plot.oversight_pct,
        to_plot["forget_loss_mean"],
        color="black",
        linestyle="-.",
        alpha=0.2,
    )

    ERA_line = mlines.Line2D([], [], color="grey", linestyle="-", label="ERA")
    pure_line = mlines.Line2D(
        [], [], color="grey", linestyle="--", label="Pure (no forget)"
    )
    base_line = mlines.Line2D(
        [], [], color="grey", linestyle=":", label="Base (full data)"
    )

    ax_retain.legend(handles=[ERA_line, pure_line, base_line])
    plt.tight_layout()  # Ensure space for suptitle and xlabel
    plt.savefig("figures/tinystories_alignment_tax.pdf", bbox_inches="tight")

# %%
