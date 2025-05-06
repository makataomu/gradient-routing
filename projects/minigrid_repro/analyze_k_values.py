#!/usr/bin/env python3
"""
analyze_k_values.py

Aggregate and plot K-sweep results using the same two-panel style as the original analysis scripts.
Each subfolder under `--base_dir` should be named `<exp_label>_over_<oversight_level>`, e.g. `staticK_10_over_0p01`.
This script will filter to a single oversight level, reindex via the Oracle helper, then plot:
 1. **Top panel**: training average return vs update index (from `train_results*.csv`)  
 2. **Bottom panel**: evaluation average return vs update index (from `eval_results*.csv`)  

Usage:
  python projects/minigrid_repro/analyze_k_values.py \
    --base_dir data/k_values \
    --oversight_prob 0.01 \
    --output_dir figures/k_values \
    --training_method routing \
    [--smooth 1]

python projects/minigrid_repro/analyze_k_values.py --base_dir projects/minigrid_repro/data/k_values --oversight_prob 0.01 --output_dir projects/minigrid_repro/figures/k_values --training_method routing
python projects/minigrid_repro/analyze_k_values.py --base_dir data/k_values --oversight_prob 0.01 --output_dir figures/k_values --training_method routing
"""

import argparse
import glob
import os
import re

import analysis_utils as a_utils
import matplotlib.pyplot as plt
import pandas as pd

# Regex to extract oversight from folder name
_OVERS_RE = re.compile(r"_over_([0-9p]+)$")


def parse_oversight(tag: str) -> float:
    m = _OVERS_RE.search(tag)
    if not m:
        raise ValueError(f"Can't parse oversight from '{tag}'")
    return float(m.group(1).replace("p", "."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Directory containing experiment subfolders",
    )
    parser.add_argument(
        "--oversight_prob",
        type=float,
        required=True,
        help="Oversight level to plot (e.g. 0.01)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/k_values",
        help="Where to save the PDF",
    )
    parser.add_argument(
        "--training_method",
        type=str,
        default="routing",
        help="Label for the training method",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Smoothing window for plot (1 = no smoothing)",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    overs_to_plot = args.oversight_prob
    out_dir = args.output_dir
    method_label = args.training_method
    smooth_amt = args.smooth

    os.makedirs(out_dir, exist_ok=True)

    # Collect all train and eval data
    train_frames = []
    eval_frames = []

    for tag in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, tag)
        if not os.path.isdir(subdir):
            continue
        try:
            overs = parse_oversight(tag)
        except ValueError:
            continue
        if abs(overs - overs_to_plot) > 1e-6:
            continue

        # Load training CSVs
        for f in glob.glob(os.path.join(subdir, "train_results*.csv")):
            df = pd.read_csv(f)
            df["experiment"] = tag
            train_frames.append(df)

        # Load evaluation CSVs
        for f in glob.glob(os.path.join(subdir, "eval_results*.csv")):
            df = pd.read_csv(f)
            df["experiment"] = tag
            eval_frames.append(df)

    if not train_frames or not eval_frames:
        raise RuntimeError(
            f"No data found in '{base_dir}' for oversight={overs_to_plot}"
        )

    train_df = pd.concat(train_frames, ignore_index=True)
    eval_df = pd.concat(eval_frames, ignore_index=True)

    # Reindex via Oracle helper for consistent grouping
    a_utils.reindex_oracle(train_df)
    a_utils.reindex_oracle(eval_df)

    # Create two-panel figure
    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(8, 4), sharey=False)

    # Plot training curves
    a_utils.gplot(
        train_df,
        x="update_idx",
        y="avg_return",
        group="experiment",
        smooth=smooth_amt,
        ax=ax_train,
    )
    ax_train.set_xlabel("Update step")
    ax_train.set_ylabel("Train avg return")
    ax_train.set_title(f"Training ({method_label}) @ {overs_to_plot:.2f} oversight")

    # Plot evaluation curves
    a_utils.gplot(
        eval_df,
        x="update_idx",
        y="avg_return",
        group="experiment",
        smooth=smooth_amt,
        ax=ax_eval,
    )
    ax_eval.set_xlabel("Update step")
    ax_eval.set_ylabel("Eval avg return")
    ax_eval.set_title(f"Evaluation ({method_label}) @ {overs_to_plot:.2f} oversight")

    ax_train.legend(loc="best", fontsize="small")
    ax_eval.legend(loc="best", fontsize="small")
    plt.tight_layout()

    out_file = os.path.join(out_dir, f"k_values_{method_label}_{overs_to_plot:.2f}.pdf")
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved plot to {out_file}")
