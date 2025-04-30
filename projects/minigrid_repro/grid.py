# %%
from typing import Dict, Tuple

import torch


@torch.jit.script
class ContinuingEnv:  # все же у нас не дискретное
    def __init__(
        self,
        n_envs: int,
        nrows: int,  # size of grid
        ncols: int,
        max_step: int,  # макс кол-во шагов в эпизоде
        oversight_prob: float,
        spurious_oversight_prob: float,  # это 0. чт такое spurious oversight
        device: torch.device = torch.device("cuda"),
    ):
        self.n_envs = n_envs
        self.nrows = nrows
        self.ncols = ncols
        self.max_step = max_step
        self.oversight_prob = oversight_prob
        self.spurious_oversight_prob = spurious_oversight_prob
        self.shape = (nrows, ncols)
        self.obs_shape = (4, nrows, ncols)
        self.obs_size = 4 * nrows * ncols
        self.device = device
        self.agent_locs = torch.zeros((n_envs, 2), dtype=torch.long, device=device)
        self.ghost_locs = torch.zeros((n_envs, 2), dtype=torch.long, device=device)
        self.diamond_locs = torch.zeros((n_envs, 2), dtype=torch.long, device=device)
        self.oversight = torch.zeros(
            (n_envs, nrows, ncols), dtype=torch.bool, device=device
        )
        self.env_steps = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.episode_over = torch.zeros(n_envs, dtype=torch.bool, device=device)

        self.n_envs_arange = torch.arange(n_envs, device=device)

        self.reset()

    def reset(self):
        self._reset_envs(self.n_envs_arange)

        return self.get_obs(), {}

    def is_diamond_optimal(self):
        # исп-ся при логировании (гпт), чтобы потом оценивать агента
        dist_to_diamond = torch.abs(self.agent_locs - self.diamond_locs).sum(dim=1)
        dist_to_ghost = torch.abs(self.agent_locs - self.ghost_locs).sum(dim=1)
        return (dist_to_diamond < dist_to_ghost) + 0.5 * (
            dist_to_diamond == dist_to_ghost
        )

    @torch.jit.export
    def get_obs(self) -> torch.Tensor:
        obs = torch.zeros((self.n_envs, 4, self.nrows, self.ncols), device=self.device)
        obs[self.n_envs_arange, 0, self.agent_locs[:, 0], self.agent_locs[:, 1]] = 1
        obs[self.n_envs_arange, 1, self.ghost_locs[:, 0], self.ghost_locs[:, 1]] = 1
        obs[
            self.n_envs_arange,
            2,
            self.diamond_locs[:, 0],
            self.diamond_locs[:, 1],
        ] = 1
        obs[:, 3, ...] = self.oversight
        return obs

    @torch.jit.export
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        # moves agents according to actions
        is_diamond_optimal_before_action = self.is_diamond_optimal()

        # Move agents
        row_offsets = torch.tensor([-1, 0, 1, 0], device=self.device)[actions]
        col_offsets = torch.tensor([0, 1, 0, -1], device=self.device)[actions]

        self.agent_locs[:, 0] = torch.clamp(
            self.agent_locs[:, 0] + row_offsets, 0, self.nrows - 1
        )
        self.agent_locs[:, 1] = torch.clamp(
            self.agent_locs[:, 1] + col_offsets, 0, self.ncols - 1
        )

        obs = self.get_obs()

        reached_ghost = torch.all(self.agent_locs == self.ghost_locs, dim=1)
        reached_diamond = torch.all(self.agent_locs == self.diamond_locs, dim=1)
        max_steps_reached = self.env_steps == self.max_step

        is_done = reached_ghost | reached_diamond | max_steps_reached
        self.env_steps += 1
        self.episode_over |= is_done

        oversight_values = self.oversight[
            self.n_envs_arange, self.agent_locs[:, 0], self.agent_locs[:, 1]
        ]

        info: Dict[str, torch.Tensor] = {
            "oversight": torch.full((self.n_envs,), -1.0, device=self.device),
            "reached_diamond": torch.full((self.n_envs,), -1.0, device=self.device),
            "reached_ghost": torch.full((self.n_envs,), -1.0, device=self.device),
            "num_steps": torch.full((self.n_envs,), -1.0, device=self.device),
            "was_diamond_optimal": is_diamond_optimal_before_action,
        }

        done_indices = is_done.nonzero().squeeze(1)
        info["oversight"][done_indices] = oversight_values[done_indices].float()
        info["reached_diamond"][done_indices] = reached_diamond[done_indices].float()
        info["reached_ghost"][done_indices] = reached_ghost[done_indices].float()
        info["num_steps"][done_indices] = self.env_steps[done_indices].float()

        if is_done.any():
            self._reset_envs(done_indices)

        return obs, info, is_done

    @torch.jit.export
    def _reset_envs(self, env_indices):
        n_reset = env_indices.numel()
        new_loc_indexes = torch.stack(
            [
                torch.randperm(self.nrows * self.ncols, device=self.device)[:3]
                for _ in range(n_reset)
            ]
        ).T

        rows = new_loc_indexes // self.ncols
        cols = new_loc_indexes % self.ncols

        self.agent_locs[env_indices] = torch.stack(
            [rows[0], cols[0]],
            dim=1,
        )
        self.ghost_locs[env_indices] = torch.stack(
            [rows[1], cols[1]],
            dim=1,
        )
        self.diamond_locs[env_indices] = torch.stack(
            [rows[2], cols[2]],
            dim=1,
        )

        random_values = torch.rand(n_reset * 2, device=self.device)

        self.oversight[env_indices] = (
            torch.rand(n_reset, self.nrows, self.ncols, device=self.device)
            < self.spurious_oversight_prob
        )
        self.oversight[
            env_indices,
            self.ghost_locs[env_indices, 0],
            self.ghost_locs[env_indices, 1],
        ] = random_values[:n_reset] < self.oversight_prob
        self.oversight[
            env_indices,
            self.diamond_locs[env_indices, 0],
            self.diamond_locs[env_indices, 1],
        ] = random_values[n_reset:] < self.oversight_prob

        self.env_steps[env_indices] = 0
        self.episode_over[env_indices] = False

    def render(self, env_index: int) -> str:
        render_str = f"Step {self.env_steps[env_index].item()}\n"

        agent_loc = self.agent_locs[env_index]
        ghost_loc = self.ghost_locs[env_index]
        diamond_loc = self.diamond_locs[env_index]
        oversight = self.oversight[env_index]

        for i in range(self.nrows):
            for j in range(self.ncols):
                fill_char = "|" if oversight[i, j] else " "
                render_str += fill_char

                if i == agent_loc[0].item() and j == agent_loc[1].item():
                    render_str += "*"
                elif i == ghost_loc[0].item() and j == ghost_loc[1].item():
                    render_str += "G"
                elif i == diamond_loc[0].item() and j == diamond_loc[1].item():
                    render_str += "D"
                else:
                    render_str += "."

                render_str += fill_char
            render_str += "\n"

        print(render_str)
        return render_str


if __name__ == "__main__":
    num_grids = 10000
    env = ContinuingEnv(num_grids, 5, 5, 10, 0.2, spurious_oversight_prob=0)
    obs, _ = env.reset()

    manhattan_dist = (
        (env.agent_locs - env.diamond_locs).abs().sum(axis=1)  # type: ignore
    )

    mins = torch.minimum(env.agent_locs, env.diamond_locs)
    maxs = torch.maximum(env.agent_locs, env.diamond_locs)
    ghost_in_the_way = torch.logical_and(
        mins < env.ghost_locs, env.ghost_locs < maxs
    ).any(dim=1)

    additional_travel_time = ghost_in_the_way.long() * 2

    travel_times = manhattan_dist + additional_travel_time

    discount = 0.97

    stepwise_returns = []
    for num_steps in travel_times:
        for step in range(num_steps):
            stepwise_returns.append(discount**step)

    avg_stepwise_return = sum(stepwise_returns) / len(stepwise_returns)
    print(f"{avg_stepwise_return:0.3f}")
