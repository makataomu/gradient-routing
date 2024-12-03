# type: ignore
# %%
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

import projects.tinystories.analysis_tools as atools
from projects.tinystories.analysis_tools import colors, labels, linestyles

shared_attrs = [
    "run_type",
    "model_save_name",
    "random_seed",
    "oversight_pct",
    "forget_loss",
]


def get_initial_losses(initial_df: pd.DataFrame) -> pd.DataFrame:
    all_initial_losses_csvs = os.listdir("initial_losses_recomputed")
    main_df = pd.DataFrame()
    cols = [
        "run_type",
        "model_save_name",
        "random_seed",
        "l1_coeff",
        "oversight_pct",
        "forget_loss",
        "retain_loss",
    ]
    for csv in all_initial_losses_csvs:
        df = pd.read_csv(f"initial_losses_recomputed/{csv}")
        df.columns = cols
        main_df = pd.concat([main_df, df])
    for model_save_name in initial_df.model_save_name.unique():
        if (
            "model_save_name" not in main_df.columns
            or model_save_name not in main_df.model_save_name.unique()
        ):
            print(f"Warning: {model_save_name} not in initial_losses_recomputed")
            # fall back to initial dataframe at step update_step 0
            initial_losses = initial_df.query(
                f"model_save_name == '{model_save_name}' and update_step == 0"
            )
            initial_losses = initial_losses[
                [
                    "run_type",
                    "model_save_name",
                    "random_seed",
                    "forget_loss",
                    "retain_loss",
                    "oversight_pct",
                ]
            ]
            initial_losses.columns = [
                "run_type",
                "model_save_name",
                "random_seed",
                "forget_loss",
                "retain_loss",
                "oversight_pct",
            ]
            main_df = pd.concat([main_df, initial_losses])
    return main_df


if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data_recomputed_era")
    # data_dir = os.path.join(parent_dir, "data")

    experiment_prefix = ["edemix-partial", "e11-o", "ee-eleven-o"]

    """
    Relearning and basic unlearning results
    """

    df = atools.read_all_data(data_dir, experiment_prefix)
    initial_losses_df = get_initial_losses(df)

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

    plot_order = ["rmu", "pure", "demix", "ERAC"]

    agg.sort_values(by="oversight_pct", inplace=True)
    posttrain = agg.num_stories == 64  # & (agg.update_step == 0)
    relearned = (agg.num_stories == 64) & (agg.experiment_step == "4. lowest_forget")

    plot_relearned = True
    subset = relearned if plot_relearned else posttrain

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    titles = [
        "ERA vs. data filtering vs. DEMix vs. RMU",
        "when only a proportion of forget data is labeled",
    ]
    # if plot_relearned:
    #     titles.append("BEST FORGET LOSS UNDER RETRAINING")
    ax.set_title("\n".join(titles))
    ax.set_xlabel("Proportion of forget stories labeled")
    ax.set_ylabel("Lowest retrained validation forget loss")

    for run_type in plot_order:
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

    lowest_forget = agg[agg.experiment_step == "4. lowest_forget"][
        [
            "run_type",
            "num_stories",
            "oversight_pct",
            "forget_loss_mean",
            "forget_loss_ci_width",
            "forget_loss_q5",
            "forget_loss_q95",
        ]
    ].copy()

    initial_losses = (
        initial_losses_df.groupby(["run_type", "oversight_pct"])
        .agg(
            {
                "forget_loss": atools.agg_fns,
                "retain_loss": atools.agg_fns,
            }
        )
        .reset_index()
    )
    initial_losses.columns = [
        "run_type",
        "oversight_pct",
        "forget_loss_mean",
        "forget_loss_ci_width",
        "forget_loss_q5",
        "forget_loss_q95",
        "retain_loss_mean",
        "retain_loss_ci_width",
        "retain_loss_q5",
        "retain_loss_q95",
    ]

    # plot initial
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    ax.set_title("Forget loss before retraining")
    ax.set_xlabel("Proportion of forget stories labeled")
    ax.set_ylabel("Forget loss")

    for run_type in plot_order:
        # for run_type in ["ERAC", "pure", "demix"]:
        loss_type = "forget"
        to_plot = initial_losses[initial_losses.run_type == run_type].sort_values(
            by="oversight_pct"
        )
        mean = to_plot[f"{loss_type}_loss_mean"]
        ci_width = to_plot[f"{loss_type}_loss_ci_width"]
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

    # %%
    # analyze the run_type == "rmu_erac"
    rmu_erac_post_retrain = df.query(
        "run_type == 'rmu_erac' & experiment_step == '4. lowest_forget'"
    )[shared_attrs]

    def find_corresponding_erac_model(row):
        seed = row.random_seed
        oversight_pct = row.oversight_pct
        base_model_pre_post_ablation_filename = f"data_recomputed_era/e11-o_ERAC_seed{seed-1}_oversight100_pre_post_ablation.json"
        parsed_json = json.load(open(base_model_pre_post_ablation_filename))
        pre_ablation_forget = parsed_json["pre_ablation"]["forget_loss"]
        return pre_ablation_forget

    rmu_erac_post_retrain["forget_loss_pre_ablation"] = rmu_erac_post_retrain.apply(
        find_corresponding_erac_model, axis=1
    )
    rmu_erac_initial_losses = initial_losses_df.query("run_type == 'rmu_erac'")[
        shared_attrs
    ]
    rmu_erac_pre_post = rmu_erac_post_retrain.merge(
        rmu_erac_initial_losses,
        on=["run_type", "model_save_name", "random_seed", "oversight_pct"],
    )
    rmu_erac_pre_post.columns = [
        "run_type",
        "model_save_name",
        "random_seed",
        "oversight_pct",
        "forget_loss_post_retrain",
        "forget_loss_pre_ablation",
        "forget_loss_post_coherence",
    ]

    rmu_erac_pre_post["forget_loss_improvement_retrain"] = (
        rmu_erac_pre_post.forget_loss_post_retrain
        - rmu_erac_pre_post.forget_loss_pre_ablation
    )
    rmu_erac_pre_post["forget_loss_improvement_coherence"] = (
        rmu_erac_pre_post.forget_loss_post_coherence
        - rmu_erac_pre_post.forget_loss_pre_ablation
    )

    rmu_erac_pre_post = (
        rmu_erac_pre_post.groupby(["run_type", "oversight_pct"])
        .agg(
            {
                "forget_loss_pre_ablation": atools.agg_fns,
                "forget_loss_post_retrain": atools.agg_fns,
                "forget_loss_improvement_retrain": atools.agg_fns,
                "forget_loss_post_coherence": atools.agg_fns,
                "forget_loss_improvement_coherence": atools.agg_fns,
            }
        )
        .reset_index()
    )
    rmu_erac_pre_post.columns = [
        "run_type",
        "oversight_pct",
        "forget_loss_pre_ablation_mean",
        "forget_loss_pre_ablation_ci_width",
        "forget_loss_pre_ablation_q5",
        "forget_loss_pre_ablation_q95",
        "forget_loss_post_retrain_mean",
        "forget_loss_post_retrain_ci_width",
        "forget_loss_post_retrain_q5",
        "forget_loss_post_retrain_q95",
        "forget_loss_improvement_retrain_mean",
        "forget_loss_improvement_retrain_ci_width",
        "forget_loss_improvement_retrain_q5",
        "forget_loss_improvement_retrain_q95",
        "forget_loss_post_coherence_mean",
        "forget_loss_post_coherence_ci_width",
        "forget_loss_post_coherence_q5",
        "forget_loss_post_coherence_q95",
        "forget_loss_improvement_coherence_mean",
        "forget_loss_improvement_coherence_ci_width",
        "forget_loss_improvement_coherence_q5",
        "forget_loss_improvement_coherence_q95",
    ]
    assert rmu_erac_pre_post.shape[0] == 1
    rmu_erac_pre_post_row = rmu_erac_pre_post.iloc[0]
    # %%
    # erac pre post
    era_pre_ablation_forget = df.query("run_type == 'ERAC' and update_step == -2")[
        shared_attrs
    ]
    era_post_coherence_forget = initial_losses_df.query("run_type == 'ERAC'")[
        shared_attrs
    ]
    era_post_retrain_forget = df.query(
        "run_type == 'ERAC' and experiment_step == '4. lowest_forget'"
    )[shared_attrs]
    era_pre_post = era_pre_ablation_forget.merge(
        era_post_retrain_forget,
        on=["run_type", "model_save_name", "random_seed", "oversight_pct"],
    ).merge(
        era_post_coherence_forget,
        on=["run_type", "model_save_name", "random_seed", "oversight_pct"],
    )
    era_pre_post.columns = [
        "run_type",
        "model_save_name",
        "random_seed",
        "oversight_pct",
        "forget_loss_pre_ablation",
        "forget_loss_post_retrain",
        "forget_loss_post_coherence",
    ]

    # demix pre post
    demix_pre_ablation_forget = df.query("run_type == 'demix' and update_step == -2")[
        shared_attrs
    ]
    demix_post_ablation_forget = df.query("run_type == 'demix' and update_step == -1")[
        shared_attrs
    ]
    demix_post_retrain_forget = df.query(
        "run_type == 'demix' and experiment_step == '4. lowest_forget'"
    )[shared_attrs]
    demix_post_coherence_forget = initial_losses_df.query("run_type == 'demix'")[
        shared_attrs
    ]

    demix_min_pre_retrain_forget = demix_pre_ablation_forget.merge(
        demix_post_ablation_forget,
        on=["run_type", "model_save_name", "random_seed", "oversight_pct"],
    ).merge(
        demix_post_coherence_forget,
        on=["run_type", "model_save_name", "random_seed", "oversight_pct"],
    )
    demix_min_pre_retrain_forget.columns = [
        "run_type",
        "model_save_name",
        "random_seed",
        "oversight_pct",
        "forget_loss_pre_ablation",
        "forget_loss_post_ablation",
        "forget_loss_post_coherence",
    ]
    demix_min_pre_retrain_forget["min_forget_loss_pre_retrain"] = (
        demix_min_pre_retrain_forget.apply(
            # lambda row: min(
            #     row.forget_loss_pre_ablation, row.forget_loss_post_ablation
            # ),
            lambda row: row.forget_loss_pre_ablation,
            axis=1,
        )
    )
    demix_pre_post = demix_min_pre_retrain_forget.merge(
        demix_post_retrain_forget,
        on=["run_type", "model_save_name", "random_seed", "oversight_pct"],
    )
    del demix_pre_post["forget_loss_pre_ablation"]
    del demix_pre_post["forget_loss_post_ablation"]
    demix_pre_post.rename(
        columns={
            "min_forget_loss_pre_retrain": "forget_loss_pre_ablation",
            "forget_loss": "forget_loss_post_retrain",
        },
        inplace=True,
    )

    # pure pre post
    random_seeds_present_at_all_oversight_levels_pure = (
        df.query("run_type == 'pure'")
        .groupby("oversight_pct")
        .random_seed.unique()
        .apply(set)
        .min()
    )
    random_seeds_present_at_all_oversight_levels_pure
    pure_only_those_random_seeds = df.query(
        "run_type == 'pure' and random_seed in @random_seeds_present_at_all_oversight_levels_pure"
    )

    zero_oversight_pure_forget_pre = pure_only_those_random_seeds.query(
        "run_type == 'pure' and oversight_pct == 0 and update_step == -2"
    )[
        [
            "run_type",
            "random_seed",
            "oversight_pct",
            "forget_loss",
        ]
    ]

    pure_post_every_oversight_level = pure_only_those_random_seeds.query(
        "run_type == 'pure' and experiment_step == '4. lowest_forget'"
    )[shared_attrs]
    pure_post_coherence = initial_losses_df.query(
        "run_type == 'pure' and random_seed in @random_seeds_present_at_all_oversight_levels_pure"
    )[shared_attrs]

    pure_pre_post = zero_oversight_pure_forget_pre.merge(
        pure_post_every_oversight_level,
        on=["run_type", "random_seed"],
    )
    del pure_pre_post["oversight_pct_x"]
    pure_pre_post.columns = [
        "run_type",
        "random_seed",
        "forget_loss_pre_ablation",
        "model_save_name",
        "oversight_pct",
        "forget_loss_post_retrain",
    ]
    pure_pre_post = pure_pre_post.merge(
        pure_post_coherence,
        on=["run_type", "random_seed", "model_save_name", "oversight_pct"],
    )
    pure_pre_post.columns = [
        "run_type",
        "random_seed",
        "forget_loss_pre_ablation",
        "model_save_name",
        "oversight_pct",
        "forget_loss_post_retrain",
        "forget_loss_post_coherence",
    ]

    # now do RMU

    rmu_post_retrain = df.query(
        "run_type == 'rmu' and experiment_step == '4. lowest_forget'"
    )[shared_attrs]
    rmu_post_coherence = initial_losses_df.query("run_type == 'rmu'")[shared_attrs]

    def find_corresponding_base_model(row):
        seed = row.random_seed
        base_model_pre_post_ablation_filename = (
            f"data_recomputed_era/e11_base_seed{seed-1}_pre_post_ablation.json"
        )
        parsed_json = json.load(open(base_model_pre_post_ablation_filename))
        pre_ablation_forget = parsed_json["pre_ablation"]["forget_loss"]
        return pre_ablation_forget

    if rmu_post_retrain.empty:
        # just create a dummy column
        rmu_post_retrain["forget_loss_pre_ablation"] = 0
    else:
        rmu_post_retrain["forget_loss_pre_ablation"] = rmu_post_retrain.apply(
            find_corresponding_base_model, axis=1
        )

    rmu_post_retrain = rmu_post_retrain.merge(
        rmu_post_coherence,
        on=["run_type", "model_save_name", "random_seed", "oversight_pct"],
    )
    rmu_post_retrain.columns = [
        "run_type",
        "model_save_name",
        "random_seed",
        "oversight_pct",
        "forget_loss_post_retrain",
        "forget_loss_pre_ablation",
        "forget_loss_post_coherence",
    ]
    rmu_pre_post = rmu_post_retrain
    rmu_pre_post

    # everything
    everything_pre_post = pd.concat(
        [era_pre_post, demix_pre_post, pure_pre_post, rmu_pre_post]
    )

    everything_pre_post["forget_loss_improvement_retrain"] = (
        everything_pre_post.forget_loss_post_retrain
        - everything_pre_post.forget_loss_pre_ablation
    )
    everything_pre_post["forget_loss_improvement_coherence"] = (
        everything_pre_post.forget_loss_post_coherence
        - everything_pre_post.forget_loss_pre_ablation
    )

    everything_pre_post = (
        everything_pre_post.groupby(["run_type", "oversight_pct"])
        .agg(
            {
                "forget_loss_pre_ablation": atools.agg_fns,
                "forget_loss_post_retrain": atools.agg_fns,
                "forget_loss_improvement_retrain": atools.agg_fns,
                "forget_loss_post_coherence": atools.agg_fns,
                "forget_loss_improvement_coherence": atools.agg_fns,
            }
        )
        .reset_index()
    )
    everything_pre_post.columns = [
        "run_type",
        "oversight_pct",
        "forget_loss_pre_ablation_mean",
        "forget_loss_pre_ablation_ci_width",
        "forget_loss_pre_ablation_q5",
        "forget_loss_pre_ablation_q95",
        "forget_loss_post_retrain_mean",
        "forget_loss_post_retrain_ci_width",
        "forget_loss_post_retrain_q5",
        "forget_loss_post_retrain_q95",
        "forget_loss_improvement_retrain_mean",
        "forget_loss_improvement_retrain_ci_width",
        "forget_loss_improvement_retrain_q5",
        "forget_loss_improvement_retrain_q95",
        "forget_loss_post_coherence_mean",
        "forget_loss_post_coherence_ci_width",
        "forget_loss_post_coherence_q5",
        "forget_loss_post_coherence_q95",
        "forget_loss_improvement_coherence_mean",
        "forget_loss_improvement_coherence_ci_width",
        "forget_loss_improvement_coherence_q5",
        "forget_loss_improvement_coherence_q95",
    ]

    # plot everything
    attr_linestyles = {
        "forget_loss_pre_ablation": ":",
        "forget_loss_post_retrain": "-",
        "forget_loss_post_coherence": "-",
    }
    attr_labels = {
        "forget_loss_pre_ablation": "Before",
        "forget_loss_post_retrain": "After + retrain on forget",
        "forget_loss_post_coherence": "After",
    }

    before_label = {
        "ERAC": "pre-ablation",
        "pure": "base model",
        "demix": "forget experts",
        "rmu": "base model",
    }

    markers = {
        "ERAC": "o",
        "pure": "*",
        "rmu": "^",
        "demix": "D",
    }
    marker_sizes = {
        "ERAC": 6,
        "pure": 10,
        "demix": 6,
        "rmu": 6,
    }
    bigfontsize = 14
    fontsize = 12
    fig, [[ax_erac, ax_demix], [ax_pure, ax_rmu]] = plt.subplots(
        nrows=2, ncols=2, figsize=(7, 6), dpi=300, sharey=True
    )
    fig.suptitle(
        "Measuring robust unlearning: forget loss before and after",
        fontsize=bigfontsize + 1,
    )
    for run_type, ax in zip(
        ["ERAC", "pure", "demix", "rmu"], [ax_erac, ax_pure, ax_demix, ax_rmu]
    ):
        ax.grid(True, alpha=0.3)
        if run_type == "demix":
            ax.set_ylim(1.4, 1.85)
        ax.set_title(f"{labels[run_type]}", fontsize=bigfontsize)
        to_plot = everything_pre_post.query(f"run_type == '{run_type}'").sort_values(
            by="oversight_pct"
        )
        for attr in ["forget_loss_pre_ablation", "forget_loss_post_retrain"]:
            oversight_levels = to_plot.oversight_pct
            mean = to_plot[f"{attr}_mean"]
            ci_width = to_plot[f"{attr}_ci_width"]
            label = (
                f"Before ({before_label[run_type]})"
                if attr == "forget_loss_pre_ablation"
                else "After + retrain on forget"
            )
            ax.plot(
                oversight_levels,
                mean,
                label=label,
                c=colors[run_type],
                linestyle=attr_linestyles[attr],
                lw=2,
            )
            ax.fill_between(
                oversight_levels,
                mean - ci_width,
                mean + ci_width,
                color=colors[run_type],
                alpha=0.2,
            )
            ax.legend()
    plt.tight_layout()

    for ax in [ax_pure, ax_rmu]:
        ax.set_xlabel("Proportion of forget stories labeled", fontsize=fontsize)
    for ax in [ax_erac, ax_pure]:
        ax.set_ylabel("Validation forget loss", fontsize=fontsize)

    fig.savefig("figures/tinystories_pre_post.pdf", bbox_inches="tight")

    # %% MAIN BODY PLOT
    rmu_plus_erac = {
        "label": None,
        "c": "black",
        "marker": "+",
        "mew": 1,
        "markersize": 8,
    }
    fig, (ax_unlearn, ax_robust_unlearn, ax_retain) = plt.subplots(
        ncols=3, figsize=(9, 3), dpi=300
    )
    fig.suptitle("")

    for ax in [ax_unlearn, ax_robust_unlearn, ax_retain]:
        ax.grid(True, alpha=0.3)

    # now plot the deltas, with a line for each run type
    ax = ax_unlearn
    ax.set_title("Unlearning", fontsize=bigfontsize)
    ax.set_ylabel("Increase in forget loss", fontsize=fontsize)
    for run_type in plot_order:
        to_plot = everything_pre_post.query(f"run_type == '{run_type}'").sort_values(
            by="oversight_pct"
        )
        mean = to_plot.forget_loss_improvement_coherence_mean
        ci_width = to_plot.forget_loss_improvement_coherence_ci_width
        ax.plot(
            to_plot.oversight_pct,
            mean,
            label=f"{labels[run_type]}",
            c=colors[run_type],
            marker=markers[run_type],
            markersize=marker_sizes[run_type],
        )
        ax.fill_between(
            to_plot.oversight_pct,
            mean - ci_width,
            mean + ci_width,
            color=colors[run_type],
            alpha=0.2,
        )
    # plot the rmu_erac point at 100% oversight
    ax.plot(
        rmu_erac_pre_post_row.oversight_pct,
        rmu_erac_pre_post_row.forget_loss_improvement_coherence_mean,
        label=rmu_plus_erac["label"],
        c=rmu_plus_erac["c"],
        marker=rmu_plus_erac["marker"],
        mew=rmu_plus_erac["mew"],
        markersize=rmu_plus_erac["markersize"],
    )
    ax.legend()

    ax = ax_retain
    ax.set_title("Retain set performance", fontsize=bigfontsize)
    ax.set_ylabel("Retain loss", fontsize=fontsize)

    for run_type in plot_order:
        loss_type = "retain"
        to_plot = initial_losses[initial_losses.run_type == run_type].sort_values(
            by="oversight_pct"
        )
        mean = to_plot[f"{loss_type}_loss_mean"]
        ci_width = to_plot[f"{loss_type}_loss_ci_width"]
        ax.plot(
            to_plot.oversight_pct,
            mean,
            label=f"{labels[run_type]}",
            c=colors[run_type],
            marker=markers[run_type],
            markersize=marker_sizes[run_type],
        )
        ax.fill_between(
            to_plot.oversight_pct,
            mean - ci_width,
            mean + ci_width,
            color=colors[run_type],
            alpha=0.2,
        )

    # plot the rmu + erac point at 100% oversight
    ax.plot(
        rmu_erac_pre_post_row.oversight_pct,
        initial_losses.query("run_type == 'rmu_erac'").retain_loss_mean,
        label=rmu_plus_erac["label"],
        c=rmu_plus_erac["c"],
        marker=rmu_plus_erac["marker"],
        mew=rmu_plus_erac["mew"],
        markersize=rmu_plus_erac["markersize"],
    )

    ax = ax_robust_unlearn
    ax.set_title("Robust unlearning", fontsize=bigfontsize)
    ax.set_xlabel("Proportion of forget stories labeled", fontsize=14)
    ax.set_ylabel("Increase in forget loss", fontsize=fontsize)
    for run_type in plot_order:
        to_plot = everything_pre_post.query(f"run_type == '{run_type}'").sort_values(
            by="oversight_pct"
        )
        mean = to_plot.forget_loss_improvement_retrain_mean
        ci_width = to_plot.forget_loss_improvement_retrain_ci_width
        ax.plot(
            to_plot.oversight_pct,
            mean,
            label=f"{labels[run_type]}",
            c=colors[run_type],
            marker=markers[run_type],
            markersize=marker_sizes[run_type],
        )
        ax.fill_between(
            to_plot.oversight_pct,
            mean - ci_width,
            mean + ci_width,
            color=colors[run_type],
            alpha=0.2,
        )
    # plot the rmu_erac point at 100% oversight
    ax.plot(
        rmu_erac_pre_post_row.oversight_pct,
        rmu_erac_pre_post_row.forget_loss_improvement_retrain_mean,
        label=rmu_plus_erac["label"],
        c=rmu_plus_erac["c"],
        marker=rmu_plus_erac["marker"],
        mew=rmu_plus_erac["mew"],
        markersize=rmu_plus_erac["markersize"],
    )
    ax_unlearn.set_ylim(-0.1, 1)
    ax_robust_unlearn.set_ylim(-0.05, 0.22)

    # Reorder legend
    handles, labels = ax_unlearn.get_legend_handles_labels()
    handle_dict = {label: handle for label, handle in zip(labels, handles)}
    order = ["ERA", "Data filtering", "RMU", "DEMix + ablate"]
    ax_unlearn.legend([handle_dict[label] for label in order], order)

    plt.tight_layout()

    fig.savefig(
        "figures/all_unlearning_results.pdf",
        bbox_inches="tight",
    )

    # %%
    print("Number of runs per setting")
    for run_type in ["ERAC", "pure", "demix", "rmu"]:
        oversight_levels_for_this_type = df.query(f"run_type == '{run_type}'")[
            "oversight_pct"
        ].unique()
        min_num_runs = min(
            [
                df.query(
                    f"run_type == '{run_type}' and oversight_pct == {oversight_level}"
                ).random_seed.nunique()
                for oversight_level in oversight_levels_for_this_type
            ]
        )
        print(f"minimum number of runs for {run_type}: {min_num_runs}")
    print("-----")
    print("rmu + erac at 100% oversight")
    print(rmu_erac_pre_post_row)


# %%
