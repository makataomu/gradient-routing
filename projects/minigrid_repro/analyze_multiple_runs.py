# %%
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    import projects.minigrid_repro.analysis_utils as a_utils
except ImportError:
    import analysis_utils as a_utils

colors = [
    (0.1216, 0.4667, 0.7059, 1.0),
    (1.0, 0.4980, 0.0549, 1.0),
    (0.1725, 0.6275, 0.1725, 1.0),
    (0.8392, 0.1529, 0.1569, 1.0),
    (0.5804, 0.4039, 0.7412, 1.0),
    (0.5490, 0.3373, 0.2941, 1.0),
    (0.8902, 0.4667, 0.7608, 1.0),
    (0.4980, 0.4980, 0.4980, 1.0),
    (0.7373, 0.7412, 0.1333, 1.0),
    (0.0902, 0.7451, 0.8118, 1.0),
]

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="oversight_levels")
parser.add_argument("--subset_to_oversight", type=float, default=0.01)
parser.add_argument("--training_method", type=str, default="routing")
parser.add_argument("--label", type=str, default="")
args = parser.parse_args()

experiment_name = args.exp_name
subset_to_oversight = args.subset_to_oversight
training_method = args.training_method
label = args.label

parent_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(parent_dir, "data")
figures_dir = os.path.join(parent_dir, "figures")
figures_dir = os.path.join(figures_dir, experiment_name)

os.makedirs(figures_dir, exist_ok=True)

custom_description = ""

description = custom_description if custom_description else experiment_name

experiment_dir = os.path.join(data_dir, experiment_name)

print("Reading files...", end=" ")
eval_files = glob.glob(os.path.join(experiment_dir, "eval_results*.csv"))
eval_dfs = [pd.read_csv(file) for file in eval_files]
eval_res = pd.concat(eval_dfs)

train_files = glob.glob(os.path.join(experiment_dir, "train_results*.csv"))
train_dfs = [pd.read_csv(file) for file in train_files]
train_res = pd.concat(train_dfs)
if train_res.update_idx.max() > 2000:
    print("subsetting training points...", end=" ")
    # train_res = train_res[train_res.update_idx % 50 == 0]
    smooth_amt = 1
else:
    smooth_amt = 1

holdout_files = glob.glob(os.path.join(experiment_dir, "holdout_results*.csv"))
if holdout_files:
    holdout_res = pd.concat([pd.read_csv(f) for f in holdout_files], ignore_index=True)
    best_idx = holdout_res.loc[holdout_res.groupby("run_id")["avg_return"].idxmax()][
        ["run_id", "update_idx"]
    ].rename(columns={"update_idx": "best_update"})
    ncols_all_curves = 3
else:
    holdout_res = None
    best_idx = pd.DataFrame(columns=["run_id", "best_update"])
    ncols_all_curves = 2

print("done.")

is_routing = eval_res.run_label == "routing"
is_diamond_policy = eval_res.policy_type == "diamond"
eval_res = eval_res[(is_routing & is_diamond_policy) | ~is_routing]

if subset_to_oversight is not None:
    subset = train_res.oversight_prob == subset_to_oversight
    if not subset.any():
        print(f"No data for oversight prob {subset_to_oversight}.")
    train_res = train_res[train_res.oversight_prob == subset_to_oversight]
    eval_res = eval_res[eval_res.oversight_prob == subset_to_oversight]

    if label:
        train_res = train_res[train_res.run_label == label]
        eval_res = eval_res[eval_res.run_label == label]

assert smooth_amt == 1, "Smoothing doesn't play well with oracle data filtering"

a_utils.reindex_oracle(train_res)
a_utils.reindex_oracle(eval_res)

# %%
fig, ax = plt.subplots(figsize=(4, 3))
fontsize = 12
ax.set_xlabel("Update step", fontsize=fontsize)
ax.set_ylabel("Ground truth return", fontsize=fontsize)
oversight_percent = subset_to_oversight * 100
ax.set_title(
    f"Learning curves at {oversight_percent}% oversight", fontsize=fontsize + 1
)

for i, run_label in enumerate(eval_res.run_label.unique()):
    subset = eval_res[eval_res.run_label == run_label]
    a_utils.plot_line(
        subset,
        x="update_idx",
        y="avg_return",
        smooth=2,
        ax=ax,
        label=run_label,
        color=colors[i],
    )
ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=fontsize - 1)
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
plt.savefig(
    os.path.join(
        figures_dir, f"rl_learning_curves_{training_method}_{oversight_percent}.pdf"
    ),
    bbox_inches="tight",
)

# %%
run_ids = sorted(train_res.run_id.unique())
n_runs = len(run_ids)
figsize = (4 * ncols_all_curves, 4)


# Plotting all runs together with unique colors
run_ids = sorted(train_res.run_id.unique())
palette = sns.color_palette("tab10", n_colors=train_res.run_id.nunique())
run_id_to_color = {
    run_id: palette[i % len(palette)]
    for i, run_id in enumerate(train_res.run_id.unique())
}

# Plot all runs together
ncols_all_curves = 3 if holdout_res is not None else 2
figsize = (4 * ncols_all_curves, 4)
fig, axes = plt.subplots(ncols=ncols_all_curves, figsize=figsize)
fig.suptitle(
    f"{description} ({len(train_res.run_id.unique())} total runs)", fontsize=14
)

# Plot training
ax_train = axes[0]
a_utils.gplot(
    train_res,
    x="update_idx",
    y="avg_return",
    group="run_id",
    smooth=smooth_amt,
    ax=ax_train,
    palette=run_id_to_color,
)

ax_train.set_title("Training Returns")
ax_train.set_xlabel("Update step")
ax_train.set_ylabel("Train Return")

# Plot holdout (if present)
if holdout_res is not None:
    holdout_res = holdout_res[holdout_res.run_id.isin(run_ids)]

    ax_holdout = axes[1]
    a_utils.gplot(
        holdout_res,
        x="update_idx",
        y="avg_return",
        group="run_id",
        smooth=smooth_amt,
        ax=ax_holdout,
        palette=run_id_to_color,
    )
    ax_holdout.set_title("Hold-out Returns")
    ax_holdout.set_xlabel("Update step")
    ax_holdout.set_ylabel("Holdout Return")

# Plot evaluation
col_idx = 2 if holdout_res is not None else 1
ax_eval = axes[col_idx]
a_utils.gplot(
    eval_res,
    x="update_idx",
    y="avg_return",
    group="run_id",
    smooth=smooth_amt,
    ax=ax_eval,
    palette=run_id_to_color,
)
ax_eval.set_title("Test Returns")
ax_eval.set_xlabel("Update step")
ax_eval.set_ylabel("Eval Return")

# Draw vertical dashed lines at best holdout steps (if available)
if holdout_res is not None:
    for run_id in run_ids:
        color = run_id_to_color.get(run_id, "gray")
        # this_best = best_idx[best_idx.run_id == run_id]["best_update"]
        subset_holdout = holdout_res[holdout_res.run_id == run_id]
        best_holdout_update_idx = subset_holdout.loc[
            subset_holdout["avg_return"].idxmax()
        ]["update_idx"]

        if not subset_holdout.empty:
            x0 = best_holdout_update_idx
            for ax in (ax_train, ax_holdout, ax_eval):  # type: ignore
                ax.axvline(x=x0, color=color, linestyle="--", linewidth=1)
                # ax.text(
                #     x0 + 1,
                #     ax.get_ylim()[1] * 0.9,
                #     # f"Run {run_id} best",
                #     color=color,
                #     fontsize=8,
                #     rotation=90,
                #     va="top",
                #     ha="left",
                # )


plt.tight_layout()
plt.savefig(
    os.path.join(
        figures_dir,
        f"rl_{'three' if holdout_res is not None else 'two'}_curves_{training_method}_{oversight_percent}.pdf",
    ),
    bbox_inches="tight",
)

# # experimnetal i guess
# fig, axes = plt.subplots(nrows=n_runs, ncols=ncols_all_curves, figsize=figsize)
# fig.suptitle(f"{description} ({n_runs} total runs)", fontsize=14)

# for i, run_id in enumerate(run_ids):
#     subset_train = train_res[train_res.run_id == run_id]
#     ax_train = axes[i, 0] if n_runs > 1 else axes[0]
#     a_utils.gplot(
#         subset_train,
#         x="update_idx",
#         y="avg_return",
#         group="run_label",
#         smooth=smooth_amt,
#         ax=ax_train,
#     )
#     ax_train.set_title(f"Run {run_id} – Training")
#     ax_train.set_xlabel("Update step")
#     ax_train.set_ylabel("Train Return")

#     if holdout_res is not None:
#         ax_holdout = axes[i, 1] if n_runs > 1 else axes[1]
#         subset_hold = holdout_res[holdout_res.run_id == run_id]
#         a_utils.gplot(
#             subset_hold,
#             x="update_idx",
#             y="avg_return",
#             group="run_label",
#             smooth=smooth_amt,
#             ax=ax_holdout,
#         )
#         ax_holdout.set_title(f"Run {run_id} – Hold-out")
#         ax_holdout.set_xlabel("Update step")
#         ax_holdout.set_ylabel("Hold-out Return")

#     col_idx = 2 if holdout_res is not None else 1
#     ax_eval = axes[i, col_idx] if n_runs > 1 else axes[col_idx]
#     subset_eval = eval_res[eval_res.run_id == run_id]
#     a_utils.gplot(
#         subset_eval,
#         x="update_idx",
#         y="avg_return",
#         group="run_label",
#         smooth=smooth_amt,
#         ax=ax_eval,
#     )
#     ax_eval.set_title(f"Run {run_id} – Test")
#     ax_eval.set_xlabel("Update step")
#     ax_eval.set_ylabel("Eval Return")

#     if holdout_res is not None:
#         best_idx = holdout_res.loc[
#             holdout_res.groupby("run_id")["avg_return"].idxmax()
#         ][["run_id", "update_idx"]].rename(columns={"update_idx": "best_update"})
#         this_best = best_idx[best_idx.run_id == run_id]["best_update"]
#         if not this_best.empty:
#             x0 = float(this_best.iloc[0])
#             for ax in (ax_train, ax_holdout, ax_eval):  # type: ignore
#                 ax.axvline(x=x0, color="gray", linestyle="--", linewidth=1)
#                 ax.text(
#                     x0 + 1,
#                     ax.get_ylim()[1] * 0.9,
#                     "best",
#                     color="gray",
#                     fontsize=8,
#                     rotation=90,
#                     va="top",
#                     ha="left",
#                 )

# plt.tight_layout()
# plt.savefig(
#     os.path.join(
#         figures_dir,
#         f"rl_per_run_curves_{training_method}_{subset_to_oversight}.pdf",
#     ),
#     bbox_inches="tight",
# )
