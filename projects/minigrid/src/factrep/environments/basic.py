from dataclasses import dataclass
from typing import Any, Literal, Optional

import einops
import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
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
from projects.minigrid.src.factrep.environments.world_object import Ball, Goal


@dataclass
class BasicEnvConfig:
    width: int
    height: int
    n_terminals_by_kind: list[int]
    randomize_terminal_kinds: bool
    min_distance: int
    has_unique_target: bool
    has_target_in_input: bool
    reward_for_target: float
    reward_for_other: float
    randomize_agent_start: bool
    pov_observation: bool
    render_mode: Literal["human", "rgb_array"]
    agent_view_size: Optional[int] = 5


class BasicEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    @staticmethod
    def from_config(config: BasicEnvConfig):
        return BasicEnv(**config.__dict__)

    def __init__(
        self,
        width: int,
        height: int,
        n_terminals_by_kind: list[int],
        randomize_terminal_kinds: bool,
        min_distance: int,
        has_unique_target: bool,
        has_target_in_input: bool,
        reward_for_target: float,
        reward_for_other: float,
        randomize_agent_start: bool,
        pov_observation: bool,
        agent_view_size: Optional[int] = 5,
        render_mode: str = "human",
    ):
        assert (
            agent_view_size is None or agent_view_size % 2 == 1
        ), "Agent view size must be odd"
        self.agent_view_size = (agent_view_size or width) if pov_observation else width

        # Required by the gym interface
        # TODO Adapt number of channels to the number of possible color classes
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(
                3 + len(n_terminals_by_kind),
                self.agent_view_size,
                self.agent_view_size,
            ),
            dtype=np.float32,
        )
        self.actions = Action
        self.action_space = spaces.Discrete(len(self.actions))
        self.reward_range = (0, max(reward_for_target, reward_for_other))
        self.max_steps = 256

        # Variables that specify how the grid should be generated
        self.n_terminals_by_kind = n_terminals_by_kind
        self.randomize_terminal_kinds = randomize_terminal_kinds
        self.width = width
        self.height = height
        self.min_distance = min_distance
        self.has_unique_target = has_unique_target
        self.has_target_in_input = has_target_in_input
        self.reward_for_target = reward_for_target
        self.reward_for_other = reward_for_other
        self.randomize_agent_start = randomize_agent_start
        self.pov_observation = pov_observation

        # Rendering variables
        self.screen_size = 640
        self.window = None
        self.render_size = None
        self.clock = None
        self.render_mode = render_mode

        # Variables that are initialised after each reset
        self._grid: Optional[Grid] = None
        self._step_count: Optional[int] = None
        self._agent_pos: Optional[np.ndarray] = None
        self.target: Optional[Goal] = None

        colors = [c for c in COLOR_NAMES if c != "gray"]
        if len(colors) < len(self.n_terminals_by_kind):
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
    def step_count(self):
        if self._step_count is None:
            raise ValueError("Step count not initialized, call reset() first")
        return self._step_count

    def _potential_terminal_positions(
        self, agent_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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
        n_terminals = sum(self.n_terminals_by_kind)

        if not self.has_unique_target:
            terminal_positions = [
                (x, y)
                for x, y in self.np_random.choice(positions, n_terminals, replace=False)
            ]
            return terminal_positions, None

        # The strict inequality ensures that there is space for the target
        indices = np.arange(len(positions))[distances > self.min_distance]
        terminal_indices = self.np_random.choice(
            indices, n_terminals - 1, replace=False, axis=0
        )
        terminal_positions = [(x, y) for x, y in positions[terminal_indices]]

        # Choose the unique target
        min_terminal_distance = distances[terminal_indices].min()
        closer_positions = positions[distances < min_terminal_distance]
        x, y = self.np_random.choice(closer_positions, axis=0)

        return [(x, y)] + terminal_positions, (x, y)

    def _choose_terminal_kinds(self) -> list[int]:
        n_terminals = sum(self.n_terminals_by_kind)

        if not self.randomize_terminal_kinds:
            terminal_kinds = sum(
                [n * [t] for t, n in enumerate(self.n_terminals_by_kind)], []
            )
            self.np_random.shuffle(terminal_kinds)

            return terminal_kinds

        terminal_kinds = self.np_random.choice(
            len(self.n_terminals_by_kind),
            n_terminals,
            p=np.array(self.n_terminals_by_kind) / n_terminals,
            replace=True,
        ).tolist()

        return terminal_kinds

    def _generate_grid(self) -> tuple[Grid, Goal | None]:
        grid = Grid(self.width, self.height)
        grid.wall_rect(0, 0, self.width, self.height)

        n_terminals = sum(self.n_terminals_by_kind)
        positions, distances = self._potential_terminal_positions(self.agent_pos)

        if len(positions) < n_terminals:
            raise ValueError(
                f"Not enough potential terminal positions ({len(positions)}) for {n_terminals} terminals"
            )

        terminal_positions, target_position = self._choose_terminal_positions(
            positions, distances
        )

        assert len(terminal_positions) == n_terminals

        terminal_kinds = self._choose_terminal_kinds()

        target = None
        for position, kind in zip(terminal_positions, terminal_kinds):
            goal = Goal(kind)
            grid.set(*position, goal)
            if position == target_position:
                target = goal

        return grid, target

    def _agent_position_after_action(self, action: Action | int):
        old_pos = self.agent_pos
        new_pos = old_pos + Action(action).as_direction()
        cell = self.grid.get(*new_pos)

        if cell is None or cell.can_overlap():
            return new_pos
        return old_pos

    def step(self, action: Action | int):
        self._step_count = self.step_count + 1

        self._agent_pos = self._agent_position_after_action(action)
        next_cell = self.grid.get(*self.agent_pos)

        obs = self.get_observation()
        if self.render_mode == "human":
            self.render()

        terminated = isinstance(next_cell, Goal)
        truncated = self.step_count >= self.max_steps

        if not terminated and not truncated:
            info = {"true_alphas": torch.Tensor([-1, -1]).long(), "metrics": {}}
            return obs, 0, False, False, info

        terminal_kind = next_cell.kind if terminated else None

        metrics = {
            f"env/found={kind}": int(kind == terminal_kind)
            for kind in range(len(self.n_terminals_by_kind))
        }
        if isinstance(self.target, Goal):
            metrics |= {
                f"env/found={kind}_optim={self.target.kind}": int(kind == terminal_kind)
                for kind in range(len(self.n_terminals_by_kind))
            }

        if terminated:
            reached_target = self.target is None or self.target == next_cell
            reward = self.reward_for_target if reached_target else self.reward_for_other
            true_alphas = (
                F.one_hot(
                    torch.Tensor([next_cell.kind]).long(), len(self.n_terminals_by_kind)
                )
                .long()
                .squeeze(0)
            )
        else:  # truncated
            reward = 0
            true_alphas = torch.Tensor([-1] * len(self.n_terminals_by_kind)).long()

        info = {"true_alphas": true_alphas, "metrics": metrics}
        return obs, reward, terminated, truncated, info

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

    def get_observation(self):
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

        else:
            grid = self.grid
            agent_pos = self.agent_pos

        grid_encoding = torch.Tensor(
            einops.rearrange(grid.encode(), "w h c -> c h w")
        ).long()
        is_goal = grid_encoding[0] == 8
        is_wall = grid_encoding[0] == 2
        observation = torch.zeros_like(grid_encoding[0])
        observation += is_wall * 1
        observation += is_goal * (grid_encoding[1] + 3)
        observation[agent_pos[1], agent_pos[0]] = 2
        observation_onehot = torch.nn.functional.one_hot(
            observation.long(), num_classes=3 + len(self.n_terminals_by_kind)
        )
        observation_onehot = einops.rearrange(observation_onehot, "h w n_c -> n_c h w")

        return observation_onehot.float().numpy()

    def render(self):
        assert self.render_mode == "rgb_array" or self.render_mode == "human"

        ax, ay = self.agent_pos
        img = self.grid.render(TILE_PIXELS, (ax, ay), 0)

        if self.render_mode == "rgb_array":
            return img

        img = np.transpose(img, axes=(1, 0, 2))
        if self.render_size is None:
            self.render_size = img.shape[:2]
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("minigrid")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(img)

        # Create background with mission description
        bg = pygame.Surface((int(surf.get_size()[0]), int(surf.get_size()[1])))
        bg.convert()
        bg.blit(surf, (0, 0))

        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

        self.window.blit(bg, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

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

    def transport_agent(self, x: int, y: int):
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

        self._grid, self.target = self._generate_grid()
        self._step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.get_observation()

        return obs, {}


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
        pprint(torch.Tensor(obs).argmax(0))
        # pprint(obs[0])
        print()

        if terminated:
            print("terminated!")
            pprint(info)
            self.reset()
        elif truncated:
            print("truncated!")
            pprint(info)
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
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


def main():
    config = BasicEnvConfig(
        width=11,
        height=11,
        n_terminals_by_kind=[2, 2],
        randomize_terminal_kinds=False,
        render_mode="human",
        min_distance=3,
        has_unique_target=True,
        has_target_in_input=False,
        reward_for_target=1.0,
        reward_for_other=0.1,
        randomize_agent_start=False,
        pov_observation=False,
        agent_view_size=None,
    )
    env = BasicEnv.from_config(config)

    manual_control = LogManualControl(env, seed=42)

    env.reset(seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
