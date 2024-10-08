from copy import deepcopy
from itertools import product
from typing import Any

import einops
import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.patches import Arrow, Circle, Polygon, Rectangle
from polars import col as c

from projects.minigrid.src.factrep import utils
from projects.minigrid.src.factrep.environments.env import Action

# from projects.minigrid.src.factrep.environments.basic import BasicEnv
from projects.minigrid.src.factrep.environments.grid import Grid
from projects.minigrid.src.factrep.environments.partial_oversight import (
    ActType,
    ObsType,
    PartialOversightEnv,
)
from projects.minigrid.src.factrep.environments.world_object import Goal, Wall, WorldObj
from projects.minigrid.src.factrep.models import Agent


@torch.inference_mode()
def plot_critic_values(
    wrapped_env: gymnasium.Wrapper[ObsType, ActType, ObsType, ActType],
    agent: Agent,
    device: torch.device,
):
    _ = agent.eval()
    env = wrapped_env.unwrapped.envs[0]

    grid = env.unwrapped.grid
    w, h = grid.width, grid.height

    fig, ax = plt.subplots(figsize=(10, 10))
    initialize_plot(w, h, ax=ax)

    for x, y in product(range(1, w - 1), range(1, h - 1)):
        cell = grid.get(x, y)

        # Get observation for this cell
        obs = env.transport_agent(x, y)

        # Get critic values
        critic_values = (
            agent.get_value(torch.Tensor(obs).to(device).unsqueeze(0))
            .cpu()
            .numpy()
            .flatten()
        )

        # Draw the cell
        if isinstance(cell, Goal):
            draw_goal(x, y, cell.color, ax)
        elif isinstance(cell, Wall):
            draw_wall(x, y, cell.color, ax)

        # Add text for critic values
        for i, value in enumerate(critic_values):
            text_x = x + 0.5
            text_y = y + 0.5 - (i * 0.2)
            ax.text(
                text_x, text_y, f"{value:.2f}", fontsize=10, ha="center", va="center"
            )

    plt.title("Critic Values per Cell")
    plt.tight_layout()
    return fig


@torch.inference_mode()
def plot_policy(
    wrapped_env: gymnasium.Wrapper[ObsType, ActType, ObsType, ActType],
    agent: Agent,
    device: torch.device,
    require_confirmation: bool,
    draw_oversight_frame: bool = True,
    num_rollouts: int = 500,
):
    _ = agent.eval()
    n_shards = agent.n_shards

    env = wrapped_env.unwrapped
    assert isinstance(env, PartialOversightEnv)

    null_shard_alphas = torch.Tensor([-1] * n_shards).unsqueeze(0)
    other_shard_alphas = F.one_hot(
        torch.Tensor(list(range(n_shards))).long(), n_shards
    ).float()
    shard_alphas = torch.cat([null_shard_alphas, other_shard_alphas], dim=0)

    policy = utils.policy_after_steering(env, agent, shard_alphas.to(device), device)
    if draw_oversight_frame:
        oversight_mask = env.unwrapped.__dict__.get("_under_oversight", None)
    else:
        oversight_mask = None

    return shard_watershed_plots(
        env.unwrapped.grid,
        policy.cpu(),
        require_confirmation,
        oversight_mask,
        num_rollouts=num_rollouts,
    )


@torch.inference_mode()
def shard_watershed_plots(
    grid: Grid,
    shard_policies: Float[torch.Tensor, "w h n_shards n_actions"],
    require_confirmation: bool,
    oversight_mask: Grid | None,
    num_rollouts: int = 500,
):
    shard_names = ["null", "good", "bad", "3", "4"]
    shard_colors = ["#1F2937", "#0C4A6E", "#7F1D1D", "green", "purple"]
    w, h = grid.width, grid.height
    n_shards = shard_policies.shape[2]

    fig, axes = plt.subplots(1, n_shards, figsize=(5 * n_shards, 5.3), squeeze=False)
    axes = axes.flatten()

    for shard_name, shard_color, policy, ax in zip(  # pyright: ignore[reportAny]
        shard_names, shard_colors, shard_policies.unbind(2), axes
    ):
        print(shard_name)
        assert isinstance(ax, Axes)
        if shard_name == "null":
            title = "No steering"
        else:
            title = f"Steering towards {shard_name}"
        _ = ax.set_title(title, fontdict={"fontsize": 26}, pad=10)
        dx, dy = arrow_directions(policy)
        initialize_plot(w, h, ax=ax)
        cell_color: dict[tuple[int, int], str] = get_cell_sink_color(
            grid, policy, require_confirmation, num_rollouts
        )

        for x, y in product(range(1, w - 1), range(1, h - 1)):
            oversight = (
                oversight_mask is not None and oversight_mask.get(x, y) is not None
            )
            if oversight:
                draw_oversight_frame(x, y, ax)
            if require_confirmation:
                cell = grid.get(x, y)
                color = cell.color if hasattr(cell, "color") else "#1F2937"
                match color:
                    case "red":
                        terminal_color = "#DC2626"
                    case "blue":
                        terminal_color = "#0284C7"
                    case _:
                        terminal_color = color

                draw_end_episode(
                    x, y, policy[x, y, Action.confirm].item(), terminal_color, ax
                )

            if (cell := grid.get(x, y)) is not None:
                if not require_confirmation:
                    draw_object(x, y, cell, ax)
            else:
                if (fill_color := cell_color.get((x, y))) is not None:
                    fill_cell_sink_color(x, y, fill_color, ax)
                draw_policy_arrow(
                    x, y, dx[x, y].item(), dy[x, y].item(), fill_color, ax
                )

    plt.tight_layout()
    return fig


def arrow_directions(policy: Float[torch.Tensor, "w h n_actions"]):
    dxs = einops.einsum(
        policy,
        torch.Tensor([-3, 3, 0, 0, 0]),
        "w h n_actions, n_actions -> w h",
    )
    dys = einops.einsum(
        policy,
        torch.Tensor([0, 0, -3, 3, 0]),
        "w h n_actions, n_actions -> w h",
    )

    return dxs, dys


def initialize_plot(
    width: int, height: int, ax: Axes, size: tuple[int, int] | None = None
):
    if size is None:
        size = (10, 10)

    _ = ax.set_xlim(1, width - 1)
    _ = ax.set_ylim(1, height - 1)
    ax.invert_yaxis()
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])

    # Add gridlines
    for x in range(width + 1):
        _ = ax.axvline(x, color="gray", linewidth=1, alpha=0.3)
    for y in range(height + 1):
        _ = ax.axhline(y, color="gray", linewidth=1, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_color("gray")


def get_cell_sink_color(
    grid: Grid,
    policy: Float[torch.Tensor, "w h n_actions"],
    require_confirmation: bool,
    num_rollouts: int = 500,
):
    goal_colors: dict[tuple[int, int], str] = {
        (x, y): "#DC2626" if goal.color == "red" else "#0284C7"
        for x, y in product(range(grid.width), range(grid.height))
        if isinstance(goal := grid.get(x, y), Goal)
    }

    def single_rollout(start_x: int, start_y: int) -> str | None:
        current_x, current_y = start_x, start_y

        while 0 <= current_x < policy.shape[0] and 0 <= current_y < policy.shape[1]:
            if (current_x, current_y) in goal_colors and not require_confirmation:
                # if start_x == 2 and start_y == 5:
                #     print(path)
                return goal_colors[(current_x, current_y)]

            action_probs = policy[current_x, current_y].softmax(dim=-1)
            action = Action(torch.multinomial(action_probs, 1).item())

            match action:
                case Action.left:
                    current_x -= 1
                case Action.right:
                    current_x += 1
                case Action.up:
                    current_y -= 1
                case Action.down:
                    # if current_x != 5 or current_y != 4:
                    current_y += 1
                case Action.confirm:
                    if require_confirmation and isinstance(
                        grid.get(current_x, current_y), Goal
                    ):
                        return goal_colors[(current_x, current_y)]

        return None

    def blend_colors(colors: list[str]) -> str:
        if not colors:
            return "#808080"  # Default to gray if no colors

        import matplotlib.colors as mcolors

        # Count occurrences of #DC2626 (red)
        red_count = colors.count("#DC2626")
        total_count = len(colors)
        red_ratio = red_count / total_count if total_count > 0 else 0

        # Define the border colors
        red_color = mcolors.to_rgb("#DC2626")
        blue_color = mcolors.to_rgb("#0284C7")

        # Linearly interpolate between blue and red
        interpolated_color = tuple(
            ((1 - red_ratio) * blue**2 + red_ratio * red**2) ** 0.5
            for blue, red in zip(blue_color, red_color)
        )

        # if red_ratio > 0.6:
        #     return mcolors.to_rgb("#DC2626")
        # elif red_ratio < 0.4:
        #     return mcolors.to_rgb("#0284C7")
        # else:
        #     return mcolors.to_hex(interpolated_color)

        # Convert RGB to hex
        return mcolors.to_hex(interpolated_color)

    colors: dict[tuple[int, int], str] = {}
    for x, y in product(range(grid.width), range(grid.height)):
        if not isinstance(grid.get(x, y), (Goal, Wall)):
            rollout_colors = []
            while len(rollout_colors) < num_rollouts:
                if (color := single_rollout(x, y)) is not None:
                    rollout_colors.append(color)

            if rollout_colors:
                colors[(x, y)] = blend_colors(rollout_colors)

    return colors


def draw_end_episode(x: int, y: int, probability: float, color: str, ax: Axes):
    _ = ax.add_patch(
        Circle(
            (x + 0.5, y + 0.5),
            radius=probability / 2,
            facecolor=color,
            edgecolor=color,
            linewidth=3,
            alpha=0.8,
        )
    )


def fill_cell_sink_color(x: int, y: int, color: str, ax: Axes):
    rect = Rectangle(
        (x + 0.05, y + 0.05),
        0.9,
        0.9,
        facecolor=color,
        edgecolor="none",
        alpha=0.2,
    )
    _ = ax.add_patch(rect)


def draw_policy_arrow(x: int, y: int, dx: float, dy: float, color: str, ax: Axes):
    ox, oy = x + 0.5, y + 0.5
    arrow = Arrow(ox, oy, dx * 0.9, dy * 0.9, width=0.25, color=color, alpha=1)
    _ = ax.add_patch(arrow)


def draw_object(x: int, y: int, cell: WorldObj, ax: Axes):
    color = cell.color if hasattr(cell, "color") else "gray"
    if isinstance(cell, Goal):
        draw_goal(x, y, color, ax)
    else:
        draw_wall(x, y, color, ax)


def draw_goal(x: int, y: int, color: str, ax: Axes):
    if color == "blue":
        gemstone = Polygon(
            [
                (x + 0.65, y + 0.25),  # top right
                (x + 0.85, y + 0.4),  # mid right
                (x + 0.5, y + 0.75),  # bottom
                (x + 0.15, y + 0.4),  # mid left
                (x + 0.35, y + 0.25),  # top left
            ],
            closed=True,
            facecolor="#DC2626" if color == "red" else "#0284C7",
            edgecolor="none",
            alpha=1,
        )
        _ = ax.add_patch(gemstone)
    else:
        draw_ghost_skull(ax, x + 0.5, y + 0.45, 0.5)


def draw_ghost_skull(ax, x, y, size):
    # Main skull shape (circle)
    skull = Circle((x, y), size / 2, facecolor="red", edgecolor="none")
    ax.add_patch(skull)

    # Bottom part of the skull (rectangle)
    bottom = Rectangle(
        (x - size / 2, y + size / 9), size, size / 2, facecolor="red", edgecolor="none"
    )
    ax.add_patch(bottom)

    # Left eye (white circle)
    left_eye = Circle(
        (x - size / 5, y + size / 10), size / 7, facecolor="white", edgecolor="none"
    )
    ax.add_patch(left_eye)

    # Right eye (white circle)
    right_eye = Circle(
        (x + size / 5, y + size / 10), size / 7, facecolor="white", edgecolor="none"
    )
    ax.add_patch(right_eye)


def draw_wall(x: int, y: int, color: str, ax: Axes):
    rect = Rectangle((x, y), 1, 1, facecolor=color, edgecolor="none", alpha=0.33)
    _ = ax.add_patch(rect)


def draw_oversight_frame(x: int, y: int, ax: Axes):
    rect = Rectangle(
        (x + 0.04, y + 0.04),
        0.92,
        0.92,
        facecolor="none",
        edgecolor="gold",
        linewidth=2,
        fill=False,
        alpha=0.5,
    )
    _ = ax.add_patch(rect)


@torch.inference_mode()
def evaluate(
    agent: Agent,
    envs: gymnasium.vector.SyncVectorEnv,
    device: torch.device,
    gamma: float,
    n: int = 100,
):
    n_shards = agent.n_shards if hasattr(agent, "n_shards") else 0
    data = []
    for _ in range(n):
        init_obs, _ = envs.reset()
        data += collect_evaluation_data(agent, init_obs, envs, device, n_shards, gamma)

    df = pl.DataFrame(data)
    logs = _format_evaluation_logs(df, n_shards)
    return logs


def collect_evaluation_data(
    agent: Agent,
    init_obs: np.ndarray,
    init_envs: gymnasium.vector.SyncVectorEnv,
    device: torch.device,
    n_shards: int,
    gamma: float,
):
    _ = agent.eval()
    results: list[dict[str, Any]] = []

    null_alphas = (torch.Tensor([-1] * n_shards), "none")
    shard_alphas = [
        (F.one_hot(torch.Tensor([i]).long(), n_shards).float(), str(i))
        for i in range(n_shards)
    ]

    for steering_alphas, steering_label in [null_alphas] + shard_alphas:
        obs = torch.Tensor(init_obs).to(device)
        done = False

        envs = deepcopy(init_envs)
        num_steps = 0
        while not done:
            num_steps += 1
            action, *_ = agent.get_action_and_value(
                obs, true_alphas=steering_alphas.to(device)
            )
            obs, _, terminated, truncated, info = envs.step(action.cpu().numpy())  # pyright: ignore[reportAny]
            done = np.logical_or(terminated, truncated).any()
            obs = torch.Tensor(obs).to(device)

        metadata = info["final_info"][0]["metadata"]  # pyright: ignore[reportPossiblyUnboundVariable]
        metrics = info["final_info"][0]["metrics"]  # pyright: ignore[reportPossiblyUnboundVariable]
        assert isinstance(metadata, dict)
        assert isinstance(metrics, dict)

        results.append(
            {
                "steered_to": steering_label,
                "num_steps": num_steps,
                "discounted_return": metadata.get("reward", 0) * (gamma**num_steps),
            }
            | metadata
            | metrics
        )

    return results


def _format_evaluation_logs(df: pl.DataFrame, n_shards: int) -> dict[str, float]:
    logs: dict[str, float] = {}
    metrics = [c for c in df.columns if c.startswith("env/")] + [
        "num_steps",
        "discounted_return",
    ]
    for steering_target in ["none"] + [str(i) for i in range(n_shards)]:
        for metric in metrics:
            success_name = f"eval/steer={steering_target}_{metric.removeprefix('env/')}"
            value = df.filter(c("steered_to") == steering_target)[metric].mean()
            assert isinstance(value, float)
            logs[success_name] = value

        mean_num_steps = (
            df.filter(c("steered_to") == steering_target)
            .select(c(metrics).exclude("num_steps"))
            .sum_horizontal()
            .mean()
        )
        assert isinstance(mean_num_steps, float)
        logs[f"eval/{steering_target}_overall"] = mean_num_steps

    return logs
