# %%
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_ci_width(data):
    return 1.96 * np.std(data) / np.sqrt(len(data))


q5 = lambda x: np.quantile(x, q=0.05)
q95 = lambda x: np.quantile(x, q=0.95)


def to_str(arr, paren=False):
    if paren:
        return [f" ($\\pm${x:0.2f})" for x in arr]
    else:
        return [f"{x:0.2f}" for x in arr]


def aggregate(all_res, group_vars=["label", "is_bad"]):
    agg_fn = ["mean", get_ci_width, q5, q95]
    agg_res = (
        all_res.groupby(group_vars)
        .agg({"Decoder": agg_fn, "Certificate": agg_fn})
        .reset_index()
    )
    agg_res.columns = [
        *group_vars,
        "decoder_mean",
        "decoder_ci",
        "decoder_q5",
        "decoder_q95",
        "certificate_mean",
        "certificate_ci",
        "certificate_q5",
        "certificate_q95",
    ]
    return agg_res


parent_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(parent_dir, "results")


df = pd.read_csv(os.path.join(results_dir, "mnist_main_results.csv"))
# df_2 = pd.read_csv(os.path.join(results_dir, "mnist_main_results_2.csv"))
# df_3 = pd.read_csv(os.path.join(results_dir, "mnist_main_results_3.csv"))
# df = pd.concat([df, df_2, df_3], ignore_index=True)

for setting in df.setting.unique():
    print(f"{setting}")


show_main_results = False
if show_main_results:
    main_res = aggregate(df[df.setting == "Gradient routing"])

    fig, ax = plt.subplots(dpi=300, figsize=(4.5, 2.5))
    main_res[["decoder_mean", "certificate_mean"]].plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("Label")
    ax.set_ylabel("Validation loss (MAE)")
    ax.axvline(4.5, color="black", ls=":")

    hatch_color = (0.85, 0.75, 0)
    for idx, bar in enumerate(ax.patches):
        bar.set_facecolor("C3" if idx <= 4 else "gold" if idx > 9 else "C0")
        if idx > 9:
            bar.set_edgecolor(hatch_color)
            bar.set_linewidth(0)
            bar.set_hatch("///")

    decoder_patch = mpatches.Patch(color="C3", label="Decoder (0-4))")
    decoder2_patch = mpatches.Patch(color="C0", label="Decoder (5-9)")
    certificate_patch = mpatches.Patch(
        facecolor="gold",
        label="Certificate",
        hatch="//",
        edgecolor=hatch_color,
        linewidth=0,
    )
    ax.legend(handles=[decoder_patch, decoder2_patch, certificate_patch])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    plt.savefig("figures/mnist_performance.pdf", bbox_inches="tight")

show_appendix_table = True
if show_appendix_table:
    df_c = aggregate(df, group_vars=["setting", "is_bad"])
    setting_order = [
        "Gradient routing",
        "Gradient routing, separate Decoders",
        "Gradient routing, no correlation penalty",
        "Gradient routing, no regularization",
        "Gradient routing, no regularization, separate Decoders",
        "Gradient routing, bottom half encoding trained on 0-9",
        "No gradient routing, L1 penalty 1e-3, trained on 5-9 only",
        "No gradient routing, no regularization, trained on 5-9 only",
        "No gradient routing, with regularization",
        "No gradient routing, no regularization",
    ]

    to_keep = "certificate"
    to_drop = "decoder"
    # to_keep = "decoder"
    # to_drop = "certificate"
    for col in df_c.columns:
        if col.startswith(f"{to_drop}_") or "_q" in col:
            del df_c[col]
    loss_mean_label = f"{to_keep}_mean"
    loss_ci_label = f"{to_keep}_ci"

    df_c[loss_mean_label] = to_str(df_c[loss_mean_label])
    df_c[loss_ci_label] = to_str(df_c[loss_ci_label], True)
    df_c["Reconstruction"] = df_c[loss_mean_label] + df_c[loss_ci_label]
    del df_c[loss_mean_label]
    del df_c[loss_ci_label]
    df_c = df_c.pivot(index="setting", columns="is_bad").reset_index()
    df_c.columns = ["Setting", "Loss: 5-9", "Loss: 0-4"]
    df_c = df_c[["Setting", "Loss: 0-4", "Loss: 5-9"]]

    order_mapping = {setting: idx for idx, setting in enumerate(setting_order)}
    df_c["order"] = df_c.Setting.map(order_mapping)
    df_c = df_c.sort_values("order").drop("order", axis=1)

    df_c["Setting"] = [f"{idx+1: >2}. {s}" for idx, s in enumerate(df_c["Setting"])]

    print(df_c.to_latex(index=False, column_format="llcc"))


df_grad_agg = aggregate(df, group_vars=["setting", "is_bad"])

df_d = (
    df[df.setting == "Gradient routing"]
    .groupby(["run_idx", "is_bad"])
    .agg({"Certificate": "mean"})
    .reset_index()
)
df_d.groupby("is_bad").agg({"Certificate": [q5, q95]})
