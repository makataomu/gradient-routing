# projects/minigrid_repro/analyze_double_descent.py
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--subset_to_oversight", type=str, default="0.01")  # float or "none"
    p.add_argument("--policy_type", type=str, default="training_policy")  # or "diamond"
    return p.parse_args()


def none_or_float(s):
    return None if str(s).lower() == "none" else float(s)


def extract_depth(run_label: str):
    # expects "..._d4" or "...depth4" etc; adjust if you choose a different naming
    m = re.search(r"(?:_d|depth)(\d+)", run_label)
    return int(m.group(1)) if m else None


if __name__ == "__main__":
    args = parse_args()
    subset_to_ovs = none_or_float(args.subset_to_oversight)

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data", args.exp_name)
    fig_dir = os.path.join(parent_dir, "figures", args.exp_name)
    os.makedirs(fig_dir, exist_ok=True)

    eval_files = glob.glob(os.path.join(data_dir, "eval_results*.csv"))
    if not eval_files:
        raise FileNotFoundError(f"No eval_results*.csv found in {data_dir}")

    eval_res = pd.concat([pd.read_csv(f) for f in eval_files], ignore_index=True)

    if subset_to_ovs is not None:
        eval_res = eval_res[eval_res["oversight_prob"] == subset_to_ovs]

    eval_res = eval_res[eval_res["policy_type"] == args.policy_type].copy()
    eval_res["depth"] = eval_res["run_label"].apply(extract_depth)
    eval_res = eval_res[eval_res["depth"].notna()].copy()

    # final point per run_id
    final = (
        eval_res.sort_values("update_idx")
        .groupby(["run_label", "depth", "oversight_prob", "run_id"], as_index=False)
        .tail(1)
    )

    # best-so-far per run_id (helps if curves are noisy)
    best = (
        eval_res.groupby(
            ["run_label", "depth", "oversight_prob", "run_id"], as_index=False
        )["avg_return"]
        .max()
        .rename(columns={"avg_return": "best_return"})
    )

    # aggregate across seeds
    final_agg = (
        final.groupby(["depth"], as_index=False)["avg_return"]
        .agg(["mean", "std"])
        .reset_index()
    )
    best_agg = (
        best.groupby(["depth"], as_index=False)["best_return"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # plot
    plt.figure(figsize=(5, 3.5))
    plt.errorbar(
        final_agg["depth"],
        final_agg["mean"],
        yerr=final_agg["std"],
        fmt="-o",
        capsize=3,
    )
    plt.xlabel("Depth")
    plt.ylabel("Final eval return (mean±std)")
    title_ovs = "all" if subset_to_ovs is None else f"{subset_to_ovs:g}"
    plt.title(f"Final performance vs depth (oversight={title_ovs})")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f"double_descent_final_vs_depth_ovs_{title_ovs}.pdf")
    )

    plt.figure(figsize=(5, 3.5))
    plt.errorbar(
        best_agg["depth"], best_agg["mean"], yerr=best_agg["std"], fmt="-o", capsize=3
    )
    plt.xlabel("Depth")
    plt.ylabel("Best eval return (mean±std)")
    plt.title(f"Best-so-far performance vs depth (oversight={title_ovs})")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f"double_descent_best_vs_depth_ovs_{title_ovs}.pdf")
    )
