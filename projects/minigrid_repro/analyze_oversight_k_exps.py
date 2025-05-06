# type: ignore
# %%
import glob
import os
import re

import analysis_utils as a_utils
import matplotlib.pyplot as plt
import pandas as pd

parent_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(parent_dir, "data")
figures_dir = os.path.join(parent_dir, "figures")

os.makedirs(figures_dir, exist_ok=True)

print("Reading files...", end=" ")
eval_files = glob.glob(os.path.join(data_dir, "*", "eval_results*.csv"))
eval_dfs = []

# Match folder names like 'routing_over_0p01'
folder_pattern = re.compile(r"([^/\\]+)_over_([\d]+p[\d]+)")

for file_path in eval_files:
    match = folder_pattern.search(file_path)
    if not match:
        continue

    run_label, oversight_str = match.groups()
    oversight_prob = float(oversight_str.replace("p", "."))

    df = pd.read_csv(file_path)
    df["run_label"] = run_label
    df["oversight_prob"] = oversight_prob
    eval_dfs.append(df)

eval_res = pd.concat(eval_dfs, ignore_index=True)
a_utils.reindex_oracle(eval_res)

# Optional: remove problematic oversight levels
eval_res = eval_res[eval_res.oversight_prob != 0.03]

oversight_levels = sorted(eval_res.oversight_prob.unique())
print(f"{oversight_levels=}")

xticks = oversight_levels
if 0.3 in xticks:
    xticks.remove(0.3)
xtick_labels = [f"{prob * 100:0.1f}".rstrip("0").rstrip(".") for prob in xticks]
xtick_labels[-1] = ""

# Keep only final eval step per run
max_step = float("inf")
final_steps = (
    eval_res[eval_res.update_idx <= max_step]
    .sort_values("update_idx")
    .groupby(["run_label", "oversight_prob", "run_id"])
    .tail(1)
)

fig, ax = plt.subplots(figsize=(5, 3.5))
fontsize = 12
ax.set_xlabel("Oversight level (%)", fontsize=fontsize)
ax.set_ylabel("Ground truth return", fontsize=fontsize)
ax.set_title("Algorithm performance", fontsize=fontsize + 1)

# Plot each training method as a line
for run_label in sorted(final_steps.run_label.unique()):
    subset = final_steps[final_steps.run_label == run_label]
    a_utils.plot_line(
        subset,
        x="oversight_prob",
        y="avg_return",
        smooth=1,
        ax=ax,
        c=a_utils.method_colors.get(run_label, None),
        ls=a_utils.method_linestyles.get(run_label, "-"),
        label=a_utils.method_labels.get(run_label, run_label),
        alpha=1,
        marker="o",
        markersize=4,
    )

ax.set_xscale("log")
ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels)
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax.legend(framealpha=0, fontsize=fontsize - 1)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "rl_performance_by_oversight.pdf"))
