import matplotlib.pyplot as plt
import numpy as np

# ─── Preset colors & line‐styles for a few special groups ──────────────────────
preset_colors = {
    "mixture": "black",
    "training_policy": "black",
    "ghost": "C3",
    "diamond": "C0",
}
preset_linestyles = {
    "mixture": "-",
    "ghost": "--",
    "diamond": ":",
}

# ─── Default method → color / name mappings ───────────────────────────────────
method_colors = {
    "oracle": "grey",
    "filtering": "C1",
    "naive_outcomes": "C2",
    "routing": "C4",
    "no_routing_control": "C5",
    "no_gate": "C6",
}
method_linestyles = {
    "oracle": "--",
    "filtering": "-.",
    "naive_outcomes": "-",
    "routing": "-",
    "no_routing_control": ":",
    "no_gate": (0, (3, 1, 1, 1)),
}
method_labels = {
    "oracle": "Oracle filtering",
    "filtering": "Data filtering",
    "naive_outcomes": "Naive training",
    "routing": "Gradient-routed MoE",
    "no_routing_control": "MoE w/o routing",
    "no_gate": "Gradient routing w/o gate",
}


# ─── Helpers ───────────────────────────────────────────────────────────────────
def normalize(arr: np.ndarray) -> np.ndarray:
    """Min–max normalize to [0,1]."""
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def ci_width(data: np.ndarray) -> float:
    """95% confidence interval half‐width for a sample."""
    return 1.96 * np.std(data) / np.sqrt(len(data))


def q5(x):
    return np.percentile(x, 5)


def q95(x):
    return np.percentile(x, 95)


agg_fns = {
    "mean": np.mean,
    "std": np.std,
    "q5": q5,
    "q95": q95,
}


# ─── Plotting routines ─────────────────────────────────────────────────────────
def gplot(df, x: str, y: str, group: str, ax=None, smooth: int = 1):
    """
    Plot one curve per `group` in df, with optional smoothing and CI bands.
    """
    if ax is None:
        ax = plt.gca()

    for idx, (group_val, subset) in enumerate(df.groupby(group)):
        series = subset[y].rolling(smooth).mean() if smooth > 1 else subset[y]
        color = preset_colors.get(group_val, f"C{idx}")
        ls = preset_linestyles.get(group_val, "-")

        ax.plot(subset[x], series, label=group_val, color=color, ls=ls)
        # very light band
        ax.fill_between(
            subset[x],
            series - ci_width(subset[y]),
            series + ci_width(subset[y]),
            alpha=0.08,
            color=color,
        )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def plot_line(df, x: str, y: str, smooth: int, ax=None, **kwargs):
    """
    Group‐by x, aggregate y with mean±CI, then plot. kwargs passed to plt.plot().
    """
    if ax is None:
        ax = plt.gca()

    agg = df.groupby(x).agg({y: agg_fns}).reset_index()
    mean = agg[y]["mean"].rolling(smooth).mean() if smooth > 1 else agg[y]["mean"]
    width = ci_width(df[y])

    lowq = agg[y]["q5"].rolling(smooth).mean()
    highq = agg[y]["q95"].rolling(smooth).mean()

    # color & label
    label = kwargs.pop("label", y)
    color = kwargs.pop("color", method_colors.get(label, "C0"))
    ls = kwargs.pop("ls", method_linestyles.get(label, "-"))

    ax.plot(agg[x], mean, label=label, color=color, ls=ls, **kwargs)
    ax.fill_between(agg[x], mean - width, mean + width, alpha=0.25, color=color)
    ax.fill_between(agg[x], lowq, highq, alpha=0.10, color=color)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def reindex_oracle(df):
    """
    For the 'oracle' method, re‐scale the update_idx/global_step
    to simulate filtering from a larger interaction stream.
    """
    is_oracle = df.run_label == "oracle"
    for var in ("update_idx", "global_step"):
        if var in df:
            df.loc[is_oracle, var] = (
                df.loc[is_oracle, var] / df.loc[is_oracle, "oversight_prob"]
            ).astype(int)
    return df
