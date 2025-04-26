# type: ignore
# %%
import glob
import math
import os

import matplotlib.pyplot as plt
import pandas as pd

from projects.minigrid_repro.analysis_utils import gplot, normalize, plot_line

parent_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(parent_dir, "data")

experiment_name = "direct_comparisons_spurious_oversight"
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

max_update_idx = train_res.update_idx.max()
if max_update_idx > 2000:
    print("subsetting training points...", end=" ")
    train_res = train_res[train_res.update_idx % 50 == 0]
    smooth_amt = 20
elif max_update_idx > 500:
    train_res = train_res[train_res.update_idx % 10 == 0]
    smooth_amt = 5
else:
    smooth_amt = 1
print("done.")

if "run_label" not in eval_res.columns:
    print("No run label.")
    run_label = ""
elif eval_res.run_label.nunique() > 1:
    run_label = "routing"
    print(f"Multiple runtypes detected. Subsetting to {run_label} runs only.")
    eval_res = eval_res[eval_res.run_label == run_label]
    train_res = train_res[train_res.run_label == run_label]
else:
    run_label = eval_res.run_label.unique()[0]

n_runs = len(eval_res.run_id.unique())
figsize = (8, 3)

fig, (ax_train, ax_eval, ax_eval_len) = plt.subplots(ncols=3, figsize=figsize)
fig.suptitle(f"{description}:{run_label} (averages over {n_runs} runs)")
ax_train.set_title("Training returns")
ax_train.set_xlabel("Update step")
ax_train.set_ylabel("Stepwise return")
ax_eval.set_title("Ground-truth return by type")
ax_eval.set_xlabel("Update step")
ax_eval_len.set_title("Completed episode length")
ax_eval_len.set_xlabel("Update step")
ax_eval_len.set_ylabel("Number of env. steps")
plot_line(
    train_res, x="update_idx", y="avg_return", smooth=smooth_amt, ax=ax_train, c="C2"
)
if (
    "avg_return_filtered" in train_res.columns
    and train_res["avg_return_filtered"].notnull().any()
):
    plot_line(
        train_res,
        x="update_idx",
        y="avg_return_filtered",
        smooth=20,
        ax=ax_train,
        c="C5",
        ls="--",
    )
gplot(
    eval_res,
    x="update_idx",
    y="avg_return",
    group="policy_type",
    smooth=smooth_amt,
    ax=ax_eval,
)
gplot(
    eval_res,
    x="update_idx",
    y="complete_ep_len",
    group="policy_type",
    smooth=smooth_amt,
    ax=ax_eval_len,
)

ax_train.legend()
ax_eval_len.legend()
plt.tight_layout()

fig, (ax_train, ax_eval, ax_eval_len) = plt.subplots(ncols=3, figsize=figsize)
fig.suptitle(f"{description}:{run_label} ({n_runs} runs)")
ax_train.set_title("Training returns")
ax_train.set_xlabel("Update step")
ax_train.set_ylabel("Stepwise return")
ax_eval.set_title("Ground-truth return by type")
ax_eval.set_xlabel("Update step")
ax_eval_len.set_title("Completed episode length")
ax_eval_len.set_xlabel("Update step")
ax_eval_len.set_ylabel("Number of env. steps")

for run_id in eval_res.run_id.unique():
    run_eval_res = eval_res[eval_res.run_id == run_id]
    run_train_res = train_res[train_res.run_id == run_id]
    plot_line(
        run_train_res,
        x="update_idx",
        y="avg_return",
        smooth=smooth_amt,
        ax=ax_train,
        c="C2",
        alpha=0.3,
    )
    gplot(
        run_eval_res,
        x="update_idx",
        y="avg_return",
        group="policy_type",
        smooth=smooth_amt,
        ax=ax_eval,
    )
    gplot(
        run_eval_res,
        x="update_idx",
        y="complete_ep_len",
        group="policy_type",
        smooth=smooth_amt,
        ax=ax_eval_len,
    )
plt.tight_layout()

fig, (ax_gate, ax_losses) = plt.subplots(figsize=(6, 3.5), ncols=2)
fig.suptitle(f"{description}:{run_label} ({n_runs} runs)", fontsize=10)
ax_gate.set_title("Gate statistics", fontsize=10)
ax_losses.set_title("Losses", fontsize=10)

gate_vars = [
    "avg_gate_openness",
    "avg_seen_diamond_gate",
    "avg_seen_ghost_gate",
    "avg_unseen_or_unfinished_gate",
    "gate_dist_to_optimal_label",
]
colors = {
    "avg_gate_openness": "C2",
    "avg_seen_diamond_gate": "C0",
    "avg_seen_ghost_gate": "C3",
    "avg_unseen_or_unfinished_gate": "C5",
    "gate_dist_to_optimal_label": "black",
}
linestyles = {
    "avg_gate_openness": "-.",
    "avg_seen_diamond_gate": "-",
    "avg_seen_ghost_gate": "--",
    "avg_unseen_or_unfinished_gate": "-",
    "gate_dist_to_optimal_label": ":",
}
for idx, var in enumerate(gate_vars):
    plot_line(
        train_res,
        x="update_idx",
        y=var,
        smooth=smooth_amt,
        ax=ax_gate,
        label=var,
        c=colors[var],
        ls=linestyles[var],
        lw=1.5,
    )
ax_gate.legend(fontsize=6)
ax_gate.set_xlabel("Update step")

losses = [
    "loss",
    "entropy_bonus",
    "policy_loss",
    "value_loss",
    "gate_loss",
    "expert_policy_similarity",
]

for idx, label in enumerate(losses):
    plot_line(
        train_res,
        x="update_idx",
        y=label,
        smooth=smooth_amt,
        ax=ax_losses,
        label=label,
        c=f"C{idx}",
    )
ax_losses.legend(fontsize=6)
ax_losses.set_xlabel("Update step")
plt.tight_layout()

# %%


# Misc stats
x_axis = "update_idx"

losses = [
    "loss",
    "entropy_bonus",
    "policy_loss",
    "value_loss",
    "gate_loss",
    "expert_policy_similarity",
]

to_plot = [["global_step"], ["complete_ep_len"], ["n_complete_eps"], ["kept_pct"]]

plots_per_row = 2
ncols = 2
nrows = math.ceil(len(to_plot) / plots_per_row)
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 2, nrows * 2))
axes = axes.flatten()
x_vals = train_res.index
for idx, labels in enumerate(to_plot):
    ax = axes[idx]
    for y in labels:
        plot_line(train_res, x=x_axis, y=y, smooth=smooth_amt, ax=ax)
    if len(labels) == 1:
        ax.set_title(labels[0])
    else:
        ax.legend(fontsize=6)
plt.tight_layout()


fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
for label in [
    "loss",
    "entropy_bonus",
    "policy_loss",
    "value_loss",
    # "gate_loss",
    # "expert_policy_similarity",
]:
    metric = (
        train_res.groupby(train_res.index)
        .agg({label: "mean"})
        .rolling(smooth_amt)
        .mean()
    )
    axes[0].plot(metric.index, metric, label=label, lw=2)
    axes[1].plot(metric.index, normalize(metric), label=label)
axes[0].legend()
axes[0].set_title("Weighted losses")
axes[1].set_title("Losses normalized (to see trends)")

fig, ax = plt.subplots(figsize=(3, 3))
plot_line(
    train_res,
    x="update_idx",
    y="reached_diamond_seen",
    smooth=smooth_amt,
    ax=ax,
    color="C0",
    label="Diamond (seen)",
)
plot_line(
    train_res,
    x="update_idx",
    y="reached_diamond_unseen",
    smooth=smooth_amt,
    ax=ax,
    color="C0",
    ls="--",
    label="Diamond (unseen)",
)
plot_line(
    train_res,
    x="update_idx",
    y="reached_ghost_seen",
    smooth=smooth_amt,
    ax=ax,
    color="C3",
    label="Ghost (seen)",
)
plot_line(
    train_res,
    x="update_idx",
    y="reached_ghost_unseen",
    smooth=smooth_amt,
    ax=ax,
    color="C3",
    ls="--",
    label="Ghost (unseen)",
)
ax.legend(fontsize=8)
ax.set_title(f"{experiment_name}:{run_label}\nDestinations for completed episodes")
ax.set_xlabel("Update step")
ax.set_ylabel("Proportion")
