import os
import random
from itertools import product
from typing import Any, Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.record_video import RecordVideo
from jaxtyping import Float

from projects.minigrid.src.factrep.environments.partial_oversight import (
    PartialOversightEnv,
)
from projects.minigrid.src.factrep.models import Agent


def seed_everything(seed: int):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mean(x: Float[torch.Tensor, "batch"]) -> Float[torch.Tensor, ""]:
    return x.sum() / x.numel()


def wrap_in_loggers(
    get_env: Callable[[], gym.Env[Any, Any]],
    run_name: str | None = None,
    video_schedule: Callable[[int], bool] | None = None,
) -> Callable[[], gym.Env[Any, Any]]:
    wrapped = RecordEpisodeStatistics(get_env())
    if run_name is not None and video_schedule is not None:
        wrapped = RecordVideo(
            wrapped,
            f"projects/minigrid/videos/{run_name}",
            episode_trigger=video_schedule,
        )
    _ = wrapped.reset()  # pyright: ignore[reportUnknownVariableType]

    return lambda: wrapped


def policy_after_steering(
    env: PartialOversightEnv,
    agent: Agent,
    alphas: Float[torch.Tensor, "n_experiments n_shards"],
    device: torch.device,
) -> Float[torch.Tensor, "w h n_experiments n_actions"]:
    policy = torch.zeros(
        env.height,
        env.width,
        alphas.shape[0],
        env.action_space.n,  # pyright: ignore[reportUnknownArgumentType,reportAttributeAccessIssue]
        device=device,
    )

    for x, y in product(range(1, env.width - 1), range(1, env.height - 1)):
        obs = env.transport_agent(x, y)
        obs = torch.Tensor(obs)
        obs = einops.repeat(
            obs, "c h w -> n_experiments c h w", n_experiments=alphas.shape[0]
        ).to(device)
        policy[x, y] = agent.get_action_distribution(obs, alphas)

    return policy.softmax(-1)
