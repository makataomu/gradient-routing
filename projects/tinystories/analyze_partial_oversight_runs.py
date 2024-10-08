# %%
import os

import matplotlib.pyplot as plt

import projects.tinystories.analysis_tools as atools

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data")

    experiment_prefix = "e11-o_"

    """
    Relearning and basic unlearning results
    """

    df = atools.read_all_data(data_dir, experiment_prefix)

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
    posttrain = agg.num_stories == 64  # & (agg.update_step == 0)
    relearned = (agg.num_stories == 64) & (agg.experiment_step == "4. lowest_forget")

    plot_relearned = True
    subset = relearned if plot_relearned else posttrain

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    titles = [
        "Robust unlearning of ERA vs. simple data filtering",
        "when only a proportion of forget data is labeled",
    ]
    # if plot_relearned:
    #     titles.append("BEST FORGET LOSS UNDER RETRAINING")
    ax.set_title("\n".join(titles))
    ax.set_xlabel("Proportion of forget stories labeled")
    ax.set_ylabel("Retrained validation forget loss")

    colors = {"ERAC": "C4", "pure": "C5", "ERAC-sorted": "C6"}
    linestyles = {"ERAC": "-", "pure": ":", "ERAC-sorted": "--"}
    labels = {
        "ERAC": "ERA",
        "pure": "Data filtering",
        "ERAC-sorted": "ERA (train labeled first)",
    }

    for run_type in ["ERAC", "pure"]:
        to_plot = agg[subset & (agg.run_type == run_type)].sort_values(
            by="oversight_pct"  # type: ignore
        )
        for loss in ["forget"]:
            mean = to_plot[f"{loss}_loss_mean"]
            ci_width = to_plot[f"{loss}_loss_ci_width"]
            ax.plot(
                to_plot.oversight_pct,
                mean,
                label=f"{labels[run_type]}",
                c=colors[run_type],
                linestyle=linestyles[run_type],
            )
            ax.fill_between(
                to_plot.oversight_pct,
                mean - ci_width,
                mean + ci_width,
                color=colors[run_type],
                alpha=0.2,
            )

        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.savefig("figures/tinystories_partial_oversight.pdf", bbox_inches="tight")

# %%
