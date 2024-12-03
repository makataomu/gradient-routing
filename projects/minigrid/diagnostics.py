# %%
import glob
import os
from typing import Optional

import matplotlib.pyplot as plt
import torch as t
from matplotlib.patches import Rectangle
from PIL import Image

import projects.minigrid_repro.agents as agents
import projects.minigrid_repro.grid as grid
from factored_representations.utils import get_gpu_with_most_memory

ACTION_SPACE = t.arange(4).unsqueeze(1)
NUM_OBS_CHANNELS = 4
AGENT_CHANNEL = 0
GHOST_CHANNEL = 1
DIAMOND_CHANNEL = 2
OVERSIGHT_CHANNEL = 3


def sample_obs(n_samples, nrows, ncols, oversight_prob: float, device):
    env = grid.ContinuingEnv(n_samples, nrows, ncols, 10, oversight_prob, device=device)
    return env.get_obs()


def get_obs_for_all_agent_locs(
    nrows, ncols, diamond_loc, ghost_loc, device, oversight_tensor=None
):
    n_tiles = nrows * ncols
    obs_stack = t.zeros((n_tiles, NUM_OBS_CHANNELS, nrows, ncols), device=device)

    obs_stack[:, DIAMOND_CHANNEL, diamond_loc[0], diamond_loc[1]] = 1
    obs_stack[:, GHOST_CHANNEL, ghost_loc[0], ghost_loc[1]] = 1

    if oversight_tensor is not None:
        obs_stack[:, OVERSIGHT_CHANNEL] = oversight_tensor

    for tile in range(n_tiles):
        obs_stack[tile, AGENT_CHANNEL, tile // ncols, tile % ncols] = 1

    return obs_stack


@t.inference_mode()
def get_gate_values(
    gate_network, nrows, ncols, diamond_loc, ghost_loc, oversight_tensor=None
):
    device = next(gate_network.parameters()).device
    obs_stack = get_obs_for_all_agent_locs(
        nrows, ncols, diamond_loc, ghost_loc, device, oversight_tensor
    )
    gate_values = t.nn.functional.sigmoid(gate_network(obs_stack))
    return gate_values.reshape(nrows, ncols)


@t.inference_mode()
def get_policy_probs_for_all_agent_locs(
    policy, nrows, ncols, diamond_loc, ghost_loc, oversight_tensor=None
):
    device = next(policy.parameters()).device
    obs_stack = get_obs_for_all_agent_locs(
        nrows, ncols, diamond_loc, ghost_loc, device, oversight_tensor
    )
    logprobs, _ = policy.get_action_logprobs(obs_stack, ACTION_SPACE.to(device))
    probs = t.exp(logprobs).reshape(4, nrows, ncols)
    return probs


@t.inference_mode()
def plot_policy_probs(
    ghost_loc: tuple[int, int],
    diamond_loc: tuple[int, int],
    action_probs: t.Tensor,  # shape: (n_rows, n_cols, n_actions=4)
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    _, n_rows, n_cols = action_probs.shape

    # Create grid
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)

    # Define arrow directions (dx, dy) for left, up, right, down
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    # Plot arrows for each cell
    for i in range(n_rows):
        for j in range(n_cols):
            # Skip ghost and diamond locations
            if (i, j) == ghost_loc or (i, j) == diamond_loc:
                continue

            probs = action_probs[:, i, j]
            assert t.isclose(probs.sum(), t.tensor(1.0))

            # Plot an arrow for each action, scaled by probability
            for action, (dx, dy) in enumerate(directions):
                prob = probs[action].item()
                arrow_len = prob * 0.5
                head_len = arrow_len / 3 if prob > 0.25 else 0
                shaft_len = arrow_len - head_len
                ax.arrow(
                    j,
                    i,
                    dx * shaft_len,
                    dy * shaft_len,
                    head_width=head_len,
                    head_length=head_len,
                    fc="black",
                    ec="black",
                    alpha=1,
                )

    size = 12
    ax.plot(
        ghost_loc[1],
        ghost_loc[0],
        "rs",
        markersize=size,
        label="Ghost",
        markeredgecolor="black",
        markeredgewidth=1,
    )
    ax.plot(
        diamond_loc[1],
        diamond_loc[0],
        "bd",
        markersize=size,
        label="Diamond",
        markeredgecolor="black",
        markeredgewidth=1,
    )

    ax.set_xticks([idx - 0.5 for idx in range(n_cols + 1)])
    ax.set_yticks([idx - 0.5 for idx in range(n_rows + 1)])
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return ax


def add_highlight_background(
    ax,
    highlight_values: t.Tensor,
    nrows: int,
    ncols: int,
    ghost_loc: tuple[int, int],
    diamond_loc: tuple[int, int],
    alpha: float = 0.5,
):
    """
    Adds colored background highlighting to a plot based on provided values.

    Args:
        ax: The matplotlib axes to draw on
        highlight_values: (nrows, ncols) tensor of values between 0 and 1
        nrows, ncols: Grid dimensions
        ghost_loc, diamond_loc: Locations to skip highlighting
        alpha: Transparency of the highlighting
    """
    for i in range(nrows):
        for j in range(ncols):
            if (i, j) == ghost_loc:
                value = 0
            elif (i, j) == diamond_loc:
                value = 1
            else:
                value = highlight_values[i, j].item()
            intensity = max(2 * value - 1, 1 - 2 * value)
            if value < 0.5:
                color = [1, 1 - intensity, 1 - intensity, alpha]
            else:
                color = [1 - intensity, 1 - intensity, 1, alpha]
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color))


def add_outlines(ax, is_highlighted: t.Tensor, color):
    """
    Adds colored background highlighting to a plot based on provided values.

    Args:
        ax: The matplotlib axes to draw on
        highlight_values: (nrows, ncols) tensor of values between 0 and 1
        nrows, ncols: Grid dimensions
        ghost_loc, diamond_loc: Locations to skip highlighting
        alpha: Transparency of the highlighting
    """
    nrows, ncols = is_highlighted.shape
    for i in range(nrows):
        for j in range(ncols):
            if is_highlighted[i, j]:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=None,
                        edgecolor=color,
                        lw=2,
                    )
                )


def visualize_expert_policies(
    policy: agents.RoutedPolicyNetwork,
    nrows: int,
    ncols: int,
    diamond_loc: tuple[int, int],
    ghost_loc: tuple[int, int],
    oversight_tensor: t.Tensor,
    title="",
    save_path=None,
    visualize_gate=True,
    progress: Optional[float] = None,
):
    fig, axes = plt.subplots(ncols=3, figsize=(8, 3))

    policies = {
        "MoE": policy,
        "Diamond expert": policy.get_diamond_policy(),
        "Ghost expert": policy.get_ghost_policy(),
    }

    policy_probs = {
        policy_label: get_policy_probs_for_all_agent_locs(
            pol, nrows, ncols, diamond_loc, ghost_loc, oversight_tensor=oversight_tensor
        )
        for policy_label, pol in policies.items()
    }

    fig.suptitle(title)
    if progress is not None:
        assert 0 <= progress <= 1
        prog = int(progress * 100)
        fig.text(
            0.17,
            0.9,
            " " + "." * prog + " ",
            fontsize=12,
            ha="left",
            color="black",
            alpha=0.5 + 0.5 * progress,
        )
        fig.text(
            0.17, 0.9, "." + " " * 100 + ".", fontsize=12, ha="left", color="black"
        )

    if visualize_gate:
        gate_values = get_gate_values(
            policy.gating, nrows, ncols, diamond_loc, ghost_loc
        )
        add_highlight_background(
            axes[0], gate_values, nrows, ncols, ghost_loc, diamond_loc
        )

    for ax, (policy_label, probs) in zip(axes, policy_probs.items()):
        plot_policy_probs(ghost_loc, diamond_loc, probs, ax=ax)
        ax.set_title(policy_label)

    for ax in axes:
        add_outlines(ax, oversight_tensor, "yellow")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def make_gif(
    path, file_prefix, delete_images_after, pause_duration=1, animate_duration=5
):
    png_files = sorted(
        [
            f
            for f in os.listdir(path)
            if f.startswith(file_prefix) and f.endswith(".png")
        ]
    )
    if len(png_files) == 0:
        print("No PNG files found in path")
        return

    images = [Image.open(os.path.join(path, file)) for file in png_files]

    pause_ms = int(pause_duration * 1000)
    animate_ms = int(animate_duration * 1000 / len(images))

    if len(images) > 1:
        durations = [animate_ms] * (len(images) - 1) + [pause_ms]
    else:
        durations = [pause_ms] * len(images)

    save_path = os.path.join(path, f"{file_prefix}.gif")
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,  # Now using milliseconds
        loop=0,
    )
    if delete_images_after:
        for file in png_files[:-1]:
            os.remove(os.path.join(path, file))


if __name__ == "__main__":
    device = get_gpu_with_most_memory()
    print(device)

    nrows = 5
    ncols = 5
    diamond_loc = (0, 0)
    ghost_loc = (4, 4)
    use_oversight = True

    policy = agents.RoutedPolicyNetwork(
        obs_dim=nrows * ncols * 4,
        num_actions=4,
        use_gate=True,
        use_gradient_routing=True,
    )
    policy.to(device)

    for run_id in [945785]:
        policy_path = f"data/ablations/policy_{run_id}.pt"
        # policy_path = "data/policy_403300.pt"
        if os.path.exists(policy_path):
            policy.load_state_dict(
                t.load(policy_path, weights_only=True, map_location=device)
            )
            print(f"Loaded policy at {policy_path}")
        else:
            print(f"No policy found at {policy_path}")

        oversight = t.zeros((nrows, ncols), device=device)
        oversight[diamond_loc] = use_oversight
        oversight[ghost_loc] = use_oversight
        obs_stack = get_obs_for_all_agent_locs(
            nrows, ncols, diamond_loc, ghost_loc, device, oversight_tensor=oversight
        )

        with t.inference_mode():
            logprobs, _ = policy.get_action_logprobs(obs_stack, ACTION_SPACE.to(device))
        probs = t.exp(logprobs).cpu()

        visualize_expert_policies(
            policy,
            nrows,
            ncols,
            diamond_loc=diamond_loc,
            ghost_loc=ghost_loc,
            oversight_tensor=oversight,
            title=f"run id:{run_id}",
        )

        gate_vals = get_gate_values(policy.gating, nrows, ncols, diamond_loc, ghost_loc)

        non_terminal_square = t.ones((nrows, ncols), device=device, dtype=t.bool)
        non_terminal_square[diamond_loc] = False
        non_terminal_square[ghost_loc] = False

        non_terminal_gate_vals = gate_vals[non_terminal_square]

        openness = round(
            t.minimum(non_terminal_gate_vals, 1 - non_terminal_gate_vals).mean().item(),
            2,
        )
        print(f"{run_id} → openness {openness}")
