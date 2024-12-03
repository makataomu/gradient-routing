#%%
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def ci_width(data):
    return 1.96 * np.std(data) / np.sqrt(len(data))

def read_data(experiment_pattern):
    eval_files = glob.glob(os.path.join(results_dir, f"summary_{experiment_pattern}*.csv"))
    dfs = []
    for file_idx, file in enumerate(eval_files):
        tmp = pd.read_csv(file)
        # tmp["run_idx"] = file_idx
        dfs.append(tmp)
    all_df = pd.concat(dfs)
    return all_df

def plot_results(res_avg, ax, include_legend):
    ax.set_ylabel("Validation accuracy")
    res_avg.plot(kind="bar", ax=ax, color=["C4", "C3", "C0"], legend=False)
    plt.xticks(rotation=0)

    colors = ["C4", "C3", "C0"]
    hatches = [None, "//", "\\"]
    hatch_colors = [None, (1, 0.2, 0.2, 1), (0.2, 0.2, 1, 1)]
    labels = ["Decoder", "Certificate (top)", "Certificate (bot)"]

    patches = []
    for idx, bar in enumerate(ax.patches):
        color_idx = idx // 2
        color = colors[color_idx]
        hatch = hatches[color_idx]
        hatch_color = hatch_colors[color_idx]
        bar.set_facecolor(color)
        if hatch is not None:
            bar.set_edgecolor(hatch_color)
            bar.set_linewidth(0)
            bar.set_hatch(hatch)
        if idx % 2 == 0:
            patch = mpatches.Patch(
                facecolor=color,
                hatch=hatch,
                edgecolor=hatch_color,
                linewidth=0,
                label=labels[color_idx],
            )
            patches.append(patch)

    if include_legend:
        ax.legend(handles=patches, bbox_to_anchor=(1, 0.65), loc='upper left')

parent_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(parent_dir, "results")

control_res = read_data("resnet34_no_routing_baseline_split_at_15")
routing_res = read_data("resnet34_routing_l1_3e2_split_at_15")

res = pd.concat([control_res, routing_res], keys=["control", "routing"], names=["routing_type"]).reset_index()
del res["level_1"]

res_avg = res.groupby(["routing_type","routed_to"]).agg(["mean"]).reset_index()
res_avg.columns = res_avg.columns.droplevel(1)

res_ci = res.groupby(["routing_type","routed_to"]).agg(ci_width).reset_index()

fig, (ax_control, ax_routing) = plt.subplots(ncols=2, sharey=True, figsize=(9,4))
plot_results(res_avg[res_avg.routing_type == "control"], ax_control, include_legend=False)
plot_results(res_avg[res_avg.routing_type == "routing"], ax_routing, include_legend=True)

for ax in [ax_control, ax_routing]:
    ax.set_xticklabels(["Classes 0-49\n(routed to bot)", "Classes 50-99\n(routed to top)"], rotation=0)
    ax.set_xlabel("Data subset")

fig.suptitle("ResNet classifier performance with and without gradient routing", size=14)
ax_control.set_title("No gradient routing, no L1 penalty")
ax_routing.set_title("Gradient routing, L1 penalty 3e-2")

plt.tight_layout()
plt.savefig(
    os.path.join(parent_dir, "figures", f"a_resnet_routing_results.pdf"),
    bbox_inches="tight",
)


