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
parser.add_argument(
    "--combine",
    action="store_true",
    help="Add holdout oversight value from run_label to oversight_prob",
)
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

# --- Optional oversight_prob augmentation ---
if args.combine:

    def parse_extra_oversight(run_label):
        try:
            suffix = run_label.split("_")[-1]
            return float(suffix)
        except Exception:
            return 0.0  # fallback if malformed

    eval_res["oversight_holdout"] = eval_res["run_label"].apply(parse_extra_oversight)
    eval_res["oversight_prob"] += eval_res["oversight_holdout"]

    # rename run_label to show the fraction holdout/total
    def make_new_label(row):
        base = row["run_label"].rsplit("_", 1)[0]  # remove numeric suffix
        denom = row["oversight_prob"]
        num = row["oversight_holdout"]
        frac = round(num / denom, 1) if denom > 0 else 0.0
        return f"{base}_{frac}_frac"

    eval_res["run_label"] = eval_res.apply(make_new_label, axis=1)
    eval_res.drop(columns=["oversight_holdout"], inplace=True)

# --- Load holdout to find each run's best update_idx (if present) ---
holdout_files = glob.glob(os.path.join(experiment_dir, "holdout_results*.csv"))

if holdout_files:
    holdout_res = pd.concat([pd.read_csv(f) for f in holdout_files], ignore_index=True)
    best_idx = holdout_res.loc[
        holdout_res.groupby("run_id")["avg_return"].idxmax(), ["run_id", "update_idx"]
    ].rename(columns={"update_idx": "best_update"})
    eval_res = (
        eval_res.merge(best_idx, on="run_id", how="left")
        .query("best_update.isna() or update_idx <= best_update")
        .drop(columns=["best_update"])
    )
else:
    holdout_res = None
print("done.")

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

final_steps = []

for run_id, subset in eval_res.groupby("run_id"):
    if holdout_res is not None and run_id in holdout_res.run_id.unique():
        subset_holdout = holdout_res[holdout_res.run_id == run_id]
        best_holdout_update_idx = subset_holdout.loc[
            subset_holdout["avg_return"].idxmax()
        ]["update_idx"]
        print(
            run_id,
            best_holdout_update_idx,
        )
        subset = subset[subset["update_idx"] <= best_holdout_update_idx]
    final_steps.append(subset)

final_steps = pd.concat(final_steps)

final_steps = (
    eval_res.sort_values("update_idx")
    .groupby(["run_label", "oversight_prob", "run_id"], as_index=False)
    .tail(1)
)

means = final_steps.groupby("oversight_prob")["avg_return"].mean()
print(means)


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
