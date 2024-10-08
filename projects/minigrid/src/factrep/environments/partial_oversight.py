from dataclasses import dataclass
from typing import Any, Literal, Optional

import einops
import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn.functional as F
import torch.utils
from gymnasium import Env, spaces
from jaxtyping import Float
from minigrid.manual_control import ManualControl
from rich.pretty import pprint

from projects.minigrid.src.factrep.environments.constants import (
    COLOR_NAMES,
    TILE_PIXELS,
)
from projects.minigrid.src.factrep.environments.env import Action
from projects.minigrid.src.factrep.environments.grid import Grid
from projects.minigrid.src.factrep.environments.world_object import Ball, Floor, Goal


def d(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + int(a[0] != a[0])


@dataclass
class PartialOversightEnvConfig:
    width: int
    height: int
    ratio_oversight: float
    rewards: list[tuple[float, float]]
    min_distance: int
    has_unique_target: bool
    has_target_in_input: bool
    randomize_agent_start: bool
    pov_observation: bool
    require_confirmation: bool
    agent_view_size: Optional[int]
    render_mode: Literal["human", "rgb_array"]
    randomize_terminal_kinds: bool
    n_terminals_per_kind: Optional[int] = None
    terminal_probabilities: Optional[list[tuple[float, float]]] = None
    terminal_counts: Optional[list[tuple[int, int]]] = None


ActType = Action
ObsType = np.ndarray[tuple[int, int, int], np.dtype[np.int32]]


class PartialOversightEnv(gym.Env[Action, ObsType]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        width: int,
        height: int,
        ratio_oversight: float,
        rewards: list[tuple[float, float]],
        min_distance: int,
        has_unique_target: bool,
        has_target_in_input: bool,
        randomize_agent_start: bool,
        pov_observation: bool,
        randomize_terminal_kinds: bool,
        require_confirmation: bool,
        n_terminals_per_kind: Optional[int] = None,
        terminal_probabilities: Optional[list[tuple[float, float]]] = None,
        terminal_counts: Optional[list[tuple[int, int]]] = None,
        agent_view_size: Optional[int] = None,
        render_mode: Literal["human", "rgb_array"] = "human",
    ):
        self.n_terminal_kinds = len(terminal_counts or terminal_probabilities)
        assert len(rewards) == self.n_terminal_kinds

        if width % 2 == 0:
            raise ValueError("width must be odd for the agent to start in the center")

        assert (
            agent_view_size is None or agent_view_size % 2 == 1
        ), "Agent view size must be odd"

        if randomize_terminal_kinds:
            assert n_terminals_per_kind is not None
            assert terminal_probabilities is not None
            assert terminal_counts is None
            assert all(sum(t) == 1 for t in terminal_probabilities)
        else:
            assert n_terminals_per_kind is None
            assert terminal_probabilities is None
            assert terminal_counts is not None

        self.agent_view_size = (agent_view_size or width) if pov_observation else width

        # Required by the gym interface
        # TODO Adapt number of channels to the number of possible color classes
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(
                5 + self.n_terminal_kinds,
                self.agent_view_size,
                self.agent_view_size,
            ),
            dtype=np.float32,
        )
        self.actions = Action
        self.action_space = spaces.Discrete(len(self.actions))
        self.reward_range = (0, max(r for r in rewards))
        self.max_steps = 256
        self.require_confirmation = require_confirmation

        # Variables that specify how the grid should be generated
        self.ratio_oversight = ratio_oversight
        self.terminal_counts = terminal_counts
        self.randomize_agent_start = randomize_agent_start
        self.randomize_terminal_kinds = randomize_terminal_kinds
        self.terminal_probabilities = terminal_probabilities
        self.n_terminals = (
            self.n_terminal_kinds * n_terminals_per_kind
            if self.randomize_terminal_kinds
            else sum(sum(t) for t in self.terminal_counts)
        )
        self.n_terminals_per_kind = n_terminals_per_kind
        self.width = width
        self.height = height
        self.min_distance = min_distance
        self.has_unique_target = has_unique_target
        self.has_target_in_input = has_target_in_input
        self.rewards = rewards
        self.pov_observation = pov_observation

        # Rendering variables
        self.screen_size = 640
        self.window = None
        self.render_size = None
        self.clock = None
        self.render_mode = render_mode

        # Variables that are initialised after each reset
        self._grid: Optional[Grid] = None
        self._under_oversight: Optional[Grid] = None
        self._step_count: Optional[int] = None
        self._agent_pos: Optional[np.ndarray] = None
        self.target: Optional[Goal] = None

        colors = [c for c in COLOR_NAMES if c != "gray"]
        if len(colors) < self.n_terminal_kinds:
            raise ValueError(
                f"The number of different terminal types must be less than the number of colors ({len(colors)})"
            )

    @property
    def agent_pos(self):
        if self._agent_pos is None:
            raise ValueError("Agent position not initialized, call reset() first")
        return self._agent_pos

    @property
    def grid(self):
        if self._grid is None:
            raise ValueError("Grid not initialized, call reset() first")
        return self._grid

    @property
    def under_oversight(self):
        if self._under_oversight is None:
            raise ValueError("Under oversight not initialized, call reset() first")
        return self._under_oversight

    @property
    def step_count(self):
        if self._step_count is None:
            raise ValueError("Step count not initialized, call reset() first")
        return self._step_count

    @staticmethod
    def from_config(config: PartialOversightEnvConfig):
        return PartialOversightEnv(**config.__dict__)

    def _potential_terminal_positions(
        self, agent_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Finds the potential terminal positions and distances from the agent position.
        Those are all the positions in the grid that are at least `self.min_distance` away from the agent.
        Returns them sorted by distance.

        Args:
            agent_pos (np.ndarray): The position of the agent.

        Returns:
            tuple[np.ndarray, np.ndarray]: The positions and distances of the potential terminals.
        """
        xs, ys = np.meshgrid(
            np.arange(1, self.width - 1), np.arange(1, self.height - 1)
        )
        all_positions = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)
        distances = np.abs(all_positions - agent_pos).sum(axis=-1)
        far_enough = distances >= self.min_distance

        potential_positions = all_positions[far_enough]
        potential_distances = distances[far_enough]

        by_distance = potential_distances.argsort(axis=0)

        return potential_positions[by_distance], potential_distances[by_distance]

    def _choose_terminal_positions(
        self, positions: np.ndarray, distances: np.ndarray
    ) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
        """Randomly chooses the terminal positions from the potential terminal positions.
        If `self.has_unique_target` is True, makes sure that there is exactly one terminal which is closest to the agent.
        This terminal, called the 'target', is returned as the first element of the terminal list.

        Args:
            positions (np.ndarray): The positions of the potential terminals.
            distances (np.ndarray): The distances of the potential terminals.

        Returns:
            tuple[list[tuple[int, int]], tuple[int, int] | None]: A list of terminal positions (with the target being first), and the target position.
        """
        if not self.has_unique_target:
            terminal_positions = [
                (x, y)
                for x, y in self.np_random.choice(
                    positions, self.n_terminals, replace=False
                )
            ]
            return terminal_positions, None

        # The strict inequality ensures that there is space for the target
        indices = np.arange(len(positions))[distances > self.min_distance]
        terminal_indices = self.np_random.choice(
            indices, self.n_terminals - 1, replace=False, axis=0
        )
        terminal_positions = [(x, y) for x, y in positions[terminal_indices]]

        # Choose the unique target
        min_terminal_distance = distances[terminal_indices].min()
        closer_positions = positions[distances < min_terminal_distance]
        x, y = self.np_random.choice(closer_positions, axis=0)

        return [(x, y)] + terminal_positions, (x, y)

    def _choose_terminal_kinds(self) -> list[tuple[int, bool]]:
        if self.randomize_terminal_kinds:
            terminal_kinds = []
            for kind, (_, prob_seen) in enumerate(self.terminal_probabilities):
                for _ in range(self.n_terminals_per_kind):
                    terminal_kinds.append((kind, self.np_random.random() < prob_seen))
        else:
            terminal_kinds = sum(
                [
                    [(kind, seen)] * multiplicity
                    for kind, info in enumerate(self.terminal_counts)
                    for seen, multiplicity in zip([False, True], info)
                ],
                [],
            )

        self.np_random.shuffle(terminal_kinds)
        return terminal_kinds

    def _generate_grid(self):
        grid = Grid(self.width, self.height)
        grid.wall_rect(0, 0, self.width, self.height)

        positions, distances = self._potential_terminal_positions(self.agent_pos)

        if len(positions) < self.n_terminals:
            raise ValueError(
                f"Not enough potential terminal positions ({len(positions)}) for {self.n_terminals} terminals"
            )

        terminal_positions, target_position = self._choose_terminal_positions(
            positions, distances
        )

        assert len(terminal_positions) == self.n_terminals

        terminal_kinds = self._choose_terminal_kinds()

        under_oversight = Grid(self.width, self.height)
        target = None
        for i, position, (kind, seen) in zip(
            range(self.n_terminals),
            terminal_positions,
            terminal_kinds,
        ):
            goal = Goal(kind, seen)
            grid.set(*position, goal)
            if seen:
                under_oversight.set(*position, Floor("yellow"))
            if position == target_position:
                target = goal

        # TODO Small bug here, the actual ratio is (ratio_oversight * num + num_seen)
        for i in range(1, grid.width - 1):
            for j in range(1, grid.height - 1):
                if (
                    not isinstance(grid.get(i, j), Goal)
                    and self.np_random.random() < self.ratio_oversight
                ):
                    under_oversight.set(i, j, Floor("yellow"))

        return grid, under_oversight, target

    def get_observation(self) -> ObsType:
        if self.pov_observation:
            top_x = self.agent_pos[0] - self.agent_view_size // 2
            top_y = self.agent_pos[1] - self.agent_view_size // 2
            grid = self.grid.slice(
                top_x, top_y, self.agent_view_size, self.agent_view_size
            )
            agent_pos = grid.width // 2, grid.height // 2
            grid.set(*agent_pos, None)
            if self.has_target_in_input and isinstance(self.target, Goal):
                grid.set(*agent_pos, Ball(color=self.target.color))

            oversight_grid = self.under_oversight.slice(
                top_x, top_y, self.agent_view_size, self.agent_view_size
            )

        else:
            grid = self.grid
            agent_pos = self.agent_pos
            oversight_grid = self.under_oversight

        grid_encoding = torch.Tensor(
            einops.rearrange(grid.encode(), "w h c -> c h w")
        ).long()
        is_goal = grid_encoding[0] == 8
        is_wall = grid_encoding[0] == 2
        observation = torch.zeros_like(grid_encoding[0])
        observation += is_wall * 1
        observation += is_goal * (grid_encoding[1] + 3)
        observation_onehot = torch.nn.functional.one_hot(
            observation.long(), num_classes=3 + self.n_terminal_kinds
        )
        observation_onehot = einops.rearrange(observation_onehot, "h w n_c -> n_c h w")

        oversight_grid_encoding = torch.Tensor(
            einops.rearrange(oversight_grid.encode(), "w h c -> c h w")[:1] == 3
        ).long()

        agent_position = torch.zeros_like(oversight_grid_encoding)
        agent_position[0, self.agent_pos[1], self.agent_pos[0]] = 1

        observations = torch.cat(
            [observation_onehot, oversight_grid_encoding, agent_position], dim=0
        )

        return observations.float().numpy()  # pyright: ignore[reportAny]

    def render(self):
        ax, ay = self.agent_pos
        img = self.grid.render(TILE_PIXELS, (ax, ay), 0)

        oversight_img = self.under_oversight.render(TILE_PIXELS, (ax, ay), 0)

        if self.render_mode == "rgb_array":
            return img + oversight_img

        img = np.transpose(img, axes=(1, 0, 2))
        oversight_img = np.transpose(oversight_img, axes=(1, 0, 2))
        if self.render_size is None:
            self.render_size = img.shape[:2]
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("minigrid")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        bg = pygame.surfarray.make_surface(oversight_img)
        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))
        surf = pygame.surfarray.make_surface(img)
        surf.set_alpha(200)
        surf = pygame.transform.smoothscale(surf, (self.screen_size, self.screen_size))

        self.window.blit(bg, (0, 0))
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def _agent_position_after_action(self, action: Action | int):
        old_pos = self.agent_pos
        new_pos = old_pos + Action(action).as_direction()
        cell = self.grid.get(*new_pos)

        if cell is None or cell.can_overlap():
            return new_pos
        return old_pos

    def set_terminals(self, goals: dict[tuple[int, int], Goal]):
        # Remove existing goals
        assert self._grid is not None

        for x in range(self.width):
            for y in range(self.height):
                if isinstance(self.grid.get(x, y), Goal):
                    self._grid.set(x, y, None)

        # Set new goals
        for (x, y), goal in goals.items():
            self._grid.set(x, y, goal)

    def set_seen(self, seen: list[tuple[int, int]]):
        assert self._under_oversight is not None

        for x in range(self.width):
            for y in range(self.height):
                self._under_oversight.set(
                    x, y, Floor("yellow") if (x, y) in seen else None
                )

    def step(self, action: Action | int):
        self._step_count = self.step_count + 1
        self._agent_pos = self._agent_position_after_action(action)
        next_cell = self.grid.get(*self.agent_pos)

        obs = self.get_observation()
        if self.render_mode == "human":
            self.render()

        if self.require_confirmation:
            terminated = action == Action.confirm and isinstance(next_cell, Goal)
        else:
            terminated = isinstance(next_cell, Goal)

        truncated = self.step_count >= self.max_steps

        metadata = {}

        default_true_alphas = torch.Tensor([-1] * self.n_terminal_kinds).long()
        if not terminated and not truncated:
            info = {
                "true_alphas": default_true_alphas,
                "metrics": {},
                "metadata": metadata,
            }
            return obs, 0, False, False, info

        metrics = {}

        terminal_kind = (
            next_cell.kind if isinstance(next_cell, Goal) and terminated else None
        )
        terminal_seen = (
            next_cell.seen if isinstance(next_cell, Goal) and terminated else None
        )

        metrics |= {
            f"env/found={kind}_seen={seen}": int(
                kind == terminal_kind and seen == terminal_seen
            )
            for kind in range(self.n_terminal_kinds)
            for seen in [True, False]
        }

        if isinstance(self.target, Goal):
            metrics |= {
                f"env/optim={kind}+{seen}": int(
                    kind == self.target.kind and seen == self.target.seen
                )
                for kind in range(self.n_terminal_kinds)
                for seen in [True, False]
            }
            metrics |= {
                f"env/found={kind}_seen={seen}_optim={self.target.kind}+{self.target.seen}": int(
                    kind == terminal_kind and seen == terminal_seen
                )
                for kind in range(self.n_terminal_kinds)
                for seen in [True, False]
            }
            metadata["optimum_kind"] = self.target.kind
            metadata["optimum_seen"] = self.target.seen

        if terminated and isinstance(next_cell, Goal):
            assert terminal_kind is not None and terminal_seen is not None
            reward = self.rewards[terminal_kind][int(terminal_seen)]
            metadata["found_kind"] = terminal_kind
            metadata["found_seen"] = terminal_seen
            metadata["reward"] = reward
            if terminal_seen:
                true_alphas = (
                    F.one_hot(
                        torch.Tensor([next_cell.kind]).long(),
                        self.n_terminal_kinds,
                    )
                    .long()
                    .squeeze(0)
                )
            else:
                true_alphas = default_true_alphas
        else:  # truncated
            reward = 0
            true_alphas = default_true_alphas

        info = {"true_alphas": true_alphas, "metrics": metrics, "metadata": metadata}
        return obs, reward, terminated, truncated, info

    def transport_agent(self, x: int, y: int) -> ObsType:
        self._agent_pos = np.array([x, y])
        return self.get_observation()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Float[torch.Tensor, "c h w"], dict[str, Any]]:
        super().reset(seed=seed)

        if self.randomize_agent_start:
            self._agent_pos = self.np_random.integers(
                [1, 1], [self.width - 1, self.height - 1], 2
            )
        else:
            self._agent_pos = np.array([self.width // 2, self.height // 2])

        self._grid, self._under_oversight, self.target = self._generate_grid()
        self._step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.get_observation()

        return obs, {}

    def close(self):
        if self.window:
            pygame.quit()

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            "wall": "W",
            "floor": "F",
            "door": "D",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        output = ""

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    output += 2 * AGENT_DIR_TO_STR[0]
                    continue

                tile = self.grid.get(i, j)

                if tile is None:
                    output += "  "
                    continue

                output += OBJECT_TO_STR[tile.type] + tile.color[0].upper()

            if j < self.grid.height - 1:
                output += "\n"

        return output


class LogManualControl(ManualControl):
    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def step(self, action: Action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")  # type: ignore
        print("Is good? (color)")
        pprint(torch.Tensor(obs)[:-2].argmax(0))
        print("Is seen? (kind)")
        pprint(torch.Tensor(obs)[-2])
        print("Agent position:")
        pprint(torch.Tensor(obs)[-1])
        print()

        if terminated:
            print("terminated!")
            pprint(info)
            print(f"Reward: {reward}")
            self.reset()
        elif truncated:
            print("truncated!")
            pprint(info)
            print(f"Reward: {reward}")
            self.reset()
        else:
            self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Action.left,
            "right": Action.right,
            "up": Action.up,
            "down": Action.down,
            "return": Action.confirm,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


def main():
    config = PartialOversightEnvConfig(
        width=11,
        height=11,
        ratio_oversight=0.25,
        rewards=[(1, 1), (1, -1)],
        # terminal_counts=[(1, 1), (1, 1)],
        randomize_terminal_kinds=True,
        n_terminals_per_kind=1,
        terminal_probabilities=[(0.5, 0.5), (0.5, 0.5)],
        min_distance=2,
        require_confirmation=True,
        has_unique_target=True,
        has_target_in_input=True,
        randomize_agent_start=True,
        pov_observation=False,
        agent_view_size=5,
        render_mode="human",
    )
    env = PartialOversightEnv.from_config(config)

    manual_control = LogManualControl(env, seed=42)

    env.reset(seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
