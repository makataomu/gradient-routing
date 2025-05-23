# type: ignore
# %%
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

try:
    import projects.minigrid_repro.analysis_utils as a_utils
except ImportError:
    import analysis_utils as a_utils

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="oversight_levels")
args = parser.parse_args()

experiment_name = args.experiment_name

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

a_utils.reindex_oracle(eval_res)

# too close to 0.025 and 0.05; looks bad
eval_res = eval_res[eval_res.oversight_prob != 0.03]

oversight_levels = sorted(eval_res.oversight_prob.unique())
print(f"{oversight_levels=}")

xticks = oversight_levels
if 0.03 in xticks:
    xticks.remove(0.03)
xtick_labels = [f"{prob * 100:0.1f}".rstrip("0").rstrip(".") for prob in xticks]
xtick_labels[-1] = ""

is_routing = eval_res.run_label == "routing"
is_diamond_policy = eval_res.policy_type == "diamond"
eval_res = eval_res[(is_routing & is_diamond_policy) | ~is_routing]

oracle_updates = sorted(eval_res[eval_res.run_label == "oracle"].update_idx.unique())
max_step = float("inf")

final_steps = (
    eval_res[eval_res.update_idx <= max_step]
    .sort_values("update_idx")
    .groupby(["run_label", "oversight_prob", "run_id"])
    .tail(1)
)

res = (
    final_steps.groupby(["run_label", "oversight_prob"])
    .agg({"avg_return": ["mean", a_utils.ci_width]})
    .reset_index()
)

fig, ax = plt.subplots(figsize=(4, 3))
fontsize = 12
ax.set_xlabel("Oversight level (%)", fontsize=fontsize)
ax.set_ylabel("Ground truth return", fontsize=fontsize)
ax.set_title("Algorithm performance", fontsize=fontsize + 1)

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

linestyles = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 10)),
    "-",
    "--",
    "-.",
    ":",
]

for i, run_label in enumerate(final_steps.run_label.unique()):
    subset = final_steps[final_steps.run_label == run_label]
    label = run_label
    if run_label in a_utils.method_labels:
        label = a_utils.method_labels[run_label]
    a_utils.plot_line(
        subset,
        x="oversight_prob",
        y="avg_return",
        smooth=1,
        ax=ax,
        c=colors[i],
        ls=linestyles[i],
        label=label,
        alpha=1,
        marker="o",
        markersize=4,
    )

ax.set_xscale("log")
ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels)
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
# ax.legend(framealpha=0, fontsize=fontsize - 1)
ax.legend(
    loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=fontsize - 1
)
ax.set_xlabel("Oversight level (%)", fontsize=fontsize)
ax.set_ylabel("Ground truth return", fontsize=fontsize)
ax.set_title("Algorithm performance", fontsize=fontsize + 1)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(
    os.path.join(figures_dir, "rl_performance_by_oversight.pdf"),
    bbox_inches="tight",
)
