# evaluation_env.py  ── NEW FILE ───────────────────────────────────────────────
from typing import List, Tuple

import torch

from projects.minigrid_repro.grid import ContinuingEnv

# 1.  Specification of a single episode
# Each episode is fully determined by four int64 tensors:
#   • agent_start : shape (2,)       row, col  ∈ {0,…,nrows-1}
#   • diamond_pos : shape (2,)
#   • diamond_pos : shape (2,)
#   • ghost_pos   : shape (2,)
EpisodeSpec = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def sample_episode_specs(
    n_episodes: int,
    nrows: int,
    ncols: int,
    oversight_prob: float,
    *,
    rng: torch.Generator,
) -> List[EpisodeSpec]:
    specs: List[EpisodeSpec] = []
    num_cells = nrows * ncols
    for _ in range(n_episodes):
        perm = torch.randperm(num_cells, generator=rng)
        start, diamond, ghost = perm[:3]
        start_rc = torch.stack((start // ncols, start % ncols))
        diamond_rc = torch.stack((diamond // ncols, diamond % ncols))
        ghost_rc = torch.stack((ghost // ncols, ghost % ncols))
        probs = torch.full((2,), oversight_prob)  # no generator here
        mask = torch.bernoulli(probs, generator=rng).to(torch.bool)
        specs.append((start_rc, ghost_rc, diamond_rc, mask))
    return specs


# 2.  The deterministic-reset environment
class ReplayEnv(ContinuingEnv):
    """
    A drop-in replacement for ContinuingEnv whose `_reset_envs` method *replays*
    a pre-generated table of EpisodeSpec objects instead of sampling fresh
    randomness.  Everything else (step(), tensors, JIT speed) is inherited.
    """

    def __init__(self, episode_specs: List[EpisodeSpec], *args, **kwargs):
        self._specs: List[EpisodeSpec] = episode_specs
        self._cursor: int = 0  # round-robin over the table
        super().__init__(*args, **kwargs)

    # Overrides only the reset logic.  All tensor fields exist already
    # because they were initialised by the base-class constructor.
    def _reset_envs(self, env_indices):
        n_reset = env_indices.numel()
        specs_slice = []
        for _ in range(n_reset):
            specs_slice.append(self._specs[self._cursor % len(self._specs)])
            self._cursor += 1

        # unpack batched tensors for vectorised assignment
        agent_rc = torch.stack([s[0] for s in specs_slice]).to(self.device)
        ghost_rc = torch.stack([s[1] for s in specs_slice]).to(self.device)
        diamond_rc = torch.stack([s[2] for s in specs_slice]).to(self.device)
        masks = torch.stack([s[3] for s in specs_slice]).to(self.device)

        self.agent_locs[env_indices] = agent_rc
        self.diamond_locs[env_indices] = diamond_rc
        self.ghost_locs[env_indices] = ghost_rc

        # first wipe any old oversight flags, then set new ones
        self.oversight[env_indices].zero_()
        self.oversight[env_indices, ghost_rc[:, 0], ghost_rc[:, 1]] = masks[:, 0]
        self.oversight[env_indices, diamond_rc[:, 0], diamond_rc[:, 1]] = masks[:, 1]

        self.env_steps[env_indices] = 0
        self.episode_over[env_indices] = False
