import matplotlib.pyplot as plt
import numpy as np

preset_colors = {
    "mixture": "black",
    "training_policy": "black",
    "ghost": "C3",
    "diamond": "C0",
}
preset_linestyles = {"mixture": "-", "ghost": "--", "diamond": ":"}

method_colors = {
    "oracle": "grey",
    "filtering": "C1",
    "naive_outcomes": "C2",
    "routing": "C4",
    "no_routing_control": "C5",
    "no_gate": "C6",
}

method_linestyles = {
    "oracle": ":",
    "filtering": "--",
    "naive_outcomes": "-.",
    "routing": "-",
    "no_routing_control": ":",
    "no_gate": "-.",
}

method_labels = {
    "oracle": "Oracle filtering",
    "filtering": "Data filtering",
    "naive_outcomes": "Naive training",
    "routing": "Gradient-routed MoE",
    "no_routing_control": "MoE w/o routing",
    "no_gate": "Gradient routing w/o gate",
}


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-5)


def ci_width(data):
    return 1.96 * np.std(data) / np.sqrt(len(data))


def q5(x):
    return np.quantile(x, q=0.05)


def q95(x):
    return np.quantile(x, q=0.95)


agg_fns = ["mean", ci_width, q5, q95]


def gplot(df, x, y, group, ax=None, smooth=1):
    if ax is None:
        _, ax = plt.subplots()

    agg = df.groupby([x, group]).agg({y: agg_fns}).reset_index()
    for idx, group_val in enumerate(agg[group].unique()):
        subset = agg[agg[group] == group_val]

        if smooth > len(subset):
            smooth = 1
            if idx == 0:
                print("gplot: Ignoring large smoothing window")

        # Compute rolling averages
        mean = subset[y, "mean"].rolling(smooth).mean()
        width = subset[y, "ci_width"].rolling(smooth).mean()
        lowq = subset[y, "q5"].rolling(smooth).mean()
        highq = subset[y, "q95"].rolling(smooth).mean()

        color = preset_colors[group_val] if group_val in preset_colors else f"C{idx}"
        ls = preset_linestyles[group_val] if group_val in preset_linestyles else "-"

        ax.plot(subset[x], mean, label=group_val, color=color, ls=ls)
        ax.fill_between(subset[x], mean - width, mean + width, alpha=0.2, color=color)
        ax.fill_between(subset[x], lowq, highq, alpha=0.07, color=color)
    return ax


def plot_line(df, x, y, smooth, ax, **kwargs):
    agg = df.groupby(x).agg({y: agg_fns}).reset_index()

    if smooth > len(agg):
        print(f"plot_line: Ignoring large smoothing window of {x} vs {y}")
        smooth = 1

    mean = agg[y, "mean"].rolling(smooth).mean()
    width = agg[y, "ci_width"].rolling(smooth).mean()
    lowq = agg[y, "q5"].rolling(smooth).mean()
    highq = agg[y, "q95"].rolling(smooth).mean()

    if "color" in kwargs:
        color = kwargs["color"]
    elif "c" in kwargs:
        color = kwargs["c"]
    else:
        color = None

    if "label" in kwargs:
        ax.plot(agg[x], mean, **kwargs)
    else:
        ax.plot(agg[x], mean, label=y, **kwargs)
    ax.fill_between(agg[x], mean - width, mean + width, alpha=0.25, color=color)
    ax.fill_between(agg[x], lowq, highq, alpha=0.1, color=color)


def reindex_oracle(df):
    """
    For oracle filtering, we re-index to simulate having filtered a
    larger dataset.
    """
    is_oracle = df.run_label == "oracle"
    for var in ["update_idx", "global_step"]:
        if var in df:
            df.loc[is_oracle, var] = (
                df.loc[is_oracle, var] / df.loc[is_oracle, "oversight_prob"]
            ).astype(int)
