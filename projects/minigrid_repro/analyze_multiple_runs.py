# %%
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

import projects.minigrid_repro.analysis_utils as a_utils

parent_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(parent_dir, "data")
figures_dir = os.path.join(parent_dir, "figures")

os.makedirs(figures_dir, exist_ok=True)

custom_description = ""

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="oversight_levels")
parser.add_argument("--subset_to_oversight", type=float, default=0.1)
parser.add_argument("--training_method", type=str, default="routing")
args = parser.parse_args()

experiment_name = args.experiment_name
subset_to_oversight = args.subset_to_oversight
training_method = args.training_method

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

for run_label in eval_res.run_label.unique():
    subset = eval_res[eval_res.run_label == run_label]
    a_utils.plot_line(
        subset,
        x="update_idx",
        y="avg_return",
        smooth=2,
        ax=ax,
        c=a_utils.method_colors[run_label],
        ls=a_utils.method_linestyles[run_label],
        label=a_utils.method_labels[run_label],
        alpha=1,
        marker=None,
        markersize=4,
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
n_runs = len(eval_res.run_id.unique())
figsize = (8, 4)

fig, (ax_train, ax_eval) = plt.subplots(ncols=2, figsize=figsize)
fig.suptitle(f"{description} ({n_runs} total runs)")
ax_train.set_title("Training returns (based on each alg's reward fn)")
ax_train.set_xlabel("Update step")
ax_train.set_ylabel("Stepwise return")
ax_eval.set_title("Ground-truth return")
ax_eval.set_xlabel("Update step")
a_utils.gplot(
    train_res,
    x="update_idx",
    y="avg_return",
    group="run_label",
    smooth=smooth_amt,
    ax=ax_train,
)

is_routing = eval_res.run_label == "routing"
is_diamond_policy = eval_res.policy_type == "diamond"
eval_res_sub = eval_res[(is_routing & is_diamond_policy) | ~is_routing]

a_utils.gplot(
    eval_res_sub,
    x="update_idx",
    y="avg_return",
    group="run_label",
    smooth=smooth_amt,
    ax=ax_eval,
)

ax_train.legend()
ax_eval.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(figures_dir, f"rl_idk_{training_method}_{oversight_percent}.pdf"),
    bbox_inches="tight",
)
