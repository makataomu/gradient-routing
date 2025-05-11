# %%
import os
import time
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch as t
import tqdm

import projects.minigrid_repro.agents as agents
import projects.minigrid_repro.diagnostics as diagnostics
import projects.minigrid_repro.grid as grid
from factored_representations.utils import get_gpu_with_most_memory
from projects.minigrid_repro.evaluating_env import *

"""
$(pdm venv activate) && python projects/minigrid_repro/training.py
"""


class EarlyStopper:
    def __init__(self, patience: int = 10, tolerance: float = 1e-3):
        self.patience = patience
        self.tolerance = tolerance
        self.best = -float("inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        if metric > self.best + self.tolerance:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def play_episode(env, policy, render=True, value_fn=None):
    obs, info = env.reset()
    done = False
    while not done:
        if render:
            if value_fn is not None:
                value_est = value_fn(obs[0:1])
                print(f"Value estimate: {value_est.item():0.3f}")
            env.render(0)

        actions = policy.sample_action(obs).long()
        obs, info, dones = env.step(actions)
        done = dones[0].item()

    return {k: v[0].item() for k, v in info.items()}


# type: ignore
def generate_batch(multienv, policy, num_steps, device):
    obs = t.empty((num_steps, multienv.n_envs, *multienv.obs_shape), device=device)
    actions = t.empty((num_steps, multienv.n_envs), dtype=t.long, device=device)
    dones = t.empty((num_steps, multienv.n_envs), device=device)
    infos = {
        "oversight": t.empty((num_steps, multienv.n_envs), device=device),
        "reached_diamond": t.empty((num_steps, multienv.n_envs), device=device),
        "reached_ghost": t.empty((num_steps, multienv.n_envs), device=device),
        "num_steps": t.empty((num_steps, multienv.n_envs), device=device),
        "was_diamond_optimal": t.empty((num_steps, multienv.n_envs), device=device),
    }

    next_obs = multienv.get_obs()
    for step in range(num_steps):
        obs[step] = next_obs
        actions[step] = policy.sample_action(obs[step]).long()
        next_obs, info, dones[step] = multienv.step(actions[step])

        for key in infos:
            infos[key][step] = info[key]

    return obs, actions, dones, infos


def generate_and_process_batch(
    multienv, policy, reward_fn, discount, num_steps, device
):
    obs, actions, dones, infos = generate_batch(multienv, policy, num_steps, device)

    returns = reward_fn(infos) * 1.0
    for k in reversed(range(len(obs) - 1)):
        returns[k] += discount * returns[k + 1] * (1 - dones[k])

    obs_flat = t.Tensor(obs.reshape(-1, *multienv.obs_shape)).to(device)
    actions_flat = t.Tensor(actions.reshape(-1)).to(device)
    returns_flat = t.Tensor(returns.reshape(-1)).to(device)
    dones_flat = t.Tensor(dones.reshape(-1)).to(device)

    processed_batch = {
        "obs": obs_flat,
        "actions": actions_flat,
        "returns": returns_flat,
        "dones": dones_flat,
        "infos": infos,
    }
    return processed_batch


def get_end_stats(info):
    reached_diamond = info["reached_diamond"] == 1
    reached_ghost = info["reached_ghost"] == 1
    oversight = info["oversight"] == 1

    ep_complete = info["oversight"] != -1
    n_complete_eps = ep_complete.sum().item()

    return {
        "reached_diamond_seen": (reached_diamond * oversight).sum().item()
        / n_complete_eps,
        "reached_ghost_seen": (reached_ghost * oversight).sum().item() / n_complete_eps,
        "reached_diamond_unseen": (reached_diamond * ~oversight).sum().item()
        / n_complete_eps,
        "reached_ghost_unseen": (reached_ghost * ~oversight).sum().item()
        / n_complete_eps,
        "n_complete_eps": n_complete_eps,
        "complete_ep_len": (info["num_steps"] * ep_complete).sum().item()
        / n_complete_eps,
    }


@t.inference_mode()
def eval(policy, env_kwargs, num_env_steps, reward_fn, discount, device, env=None):
    if env is None:
        env = grid.ContinuingEnv(**env_kwargs, device=device)
    processed_batch = generate_and_process_batch(
        env, policy, reward_fn, discount, num_env_steps, device
    )
    metrics = get_end_stats(processed_batch["infos"])
    metrics["avg_return"] = t.mean(processed_batch["returns"]).item()
    return metrics


def naive_reward_fn(info):
    oversight = info["oversight"] == 1
    reached_diamond = info["reached_diamond"] == 1
    reached_ghost = info["reached_ghost"] == 1
    reached_terminal = reached_diamond | reached_ghost
    return reached_terminal.long() - 2.0 * oversight * reached_ghost.long()


def moe_reward_fn(info):
    reached_diamond = info["reached_diamond"] == 1
    reached_ghost = info["reached_ghost"] == 1
    reached_terminal = reached_diamond | reached_ghost
    return reached_terminal


def biased_moe_reward_fn(info):
    """
    Increases the reward for reaching diamond (with oversight) slightly,
    so that when diamond and ghost are equidistant, the optimal policy goes
    to diamond (rather than being indifferent -> randomizing in the case
    of an entropy bonus).

    Note: in principle, the bonus should be picked small enough (relative
    to the discount factor) so that it acts only as a tiebreak.
    """
    reached_diamond = info["reached_diamond"] == 1
    reached_ghost = info["reached_ghost"] == 1
    reached_terminal = reached_diamond | reached_ghost
    has_oversight = info["oversight"] == 1

    diamond_seen = t.logical_and(reached_diamond, has_oversight)
    other_terminal = t.logical_and(reached_terminal, ~diamond_seen)
    return 1.03 * diamond_seen + other_terminal


def true_reward_fn(info):
    reached_diamond = info["reached_diamond"] == 1
    reached_ghost = info["reached_ghost"] == 1
    return reached_diamond.long() - 1.0 * reached_ghost.long()


def train(
    steps_per_learning_update: int,
    num_learning_updates: int,
    eval_freq: int,
    policy_log_freq: int,
    discount: float,
    loss_coefs: dict,
    learning_rate: float,
    expert_weight_decay: float,
    shared_weight_decay: float,
    policy_network_constructor: Callable,
    reward_fn_to_train_on: Callable,
    loss_getter_fn: Callable,
    env_kwargs: dict,
    save_dir: str,
    policy_visualization_dir: str,
    run_label: str,
    device=None,
    gpus_to_restrict_to: Optional[list[int]] = None,
    run_id: Optional[Union[str, int]] = None,
    time_to_sleep_after_run: float = 0,
    regulariser_name: Optional[str] = None,
    regulariser_kwargs: Optional[dict] = None,
):
    """
    Trains a policy.  If `regulariser_name=="earlystop"`, we reserve a fixed
    fraction of TOTAL EPISODES as a held-out set, scored on ground-truth return,
    and stop early when that stalls.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(policy_visualization_dir, exist_ok=True)

    # ─── seed & device ────────────────────────────────────────────────────────
    seed = int(time.time()) + os.getpid()
    np.random.seed(seed)
    t.manual_seed(seed)
    run_id = run_id or np.random.choice(1_000_000)
    if device is None:
        device = get_gpu_with_most_memory(gpus_to_restrict_to)

    # ─── prepare validation env if using early-stop ──────────────────────────
    stopper = None
    val_env = None
    if regulariser_name == "earlystop":
        # pull holdout settings from the same kwargs dict
        holdout_frac = regulariser_kwargs.get("holdout_frac", 0.1)  # type: ignore
        holdout_max_episodes = regulariser_kwargs.get("holdout_max_episodes", 10_000)  # type: ignore
        patience = regulariser_kwargs.get("patience", 400)  # type: ignore
        tolerance = regulariser_kwargs.get("tolerance", 0.06)  # type: ignore

        # compute how many episodes to hold out
        total_episodes = num_learning_updates * env_kwargs.get("n_envs", 1)
        val_episodes = int(total_episodes * holdout_frac)
        val_episodes = min(val_episodes, holdout_max_episodes)

        # sample a reproducible table of episode specs
        run_id_int = int(run_id)
        seed_val = int(seed) + run_id_int
        rng = t.Generator().manual_seed(seed_val)

        val_specs = sample_episode_specs(
            n_episodes=val_episodes,
            nrows=env_kwargs["nrows"],
            ncols=env_kwargs["ncols"],
            oversight_prob=env_kwargs["oversight_prob"],
            rng=rng,
        )
        # build the ReplayEnv over those specs
        val_env = ReplayEnv(
            episode_specs=val_specs,
            **env_kwargs,
            device=device,
        )

        stopper = EarlyStopper(patience=patience, tolerance=tolerance)

    # ─── environment & networks ────────────────────────────────────────────────
    train_env = grid.ContinuingEnv(**env_kwargs, device=device)  # fresh env
    policy = policy_network_constructor(train_env.obs_size, 4).to(device)
    agents.reset_params(policy)
    value_fn = agents.ValueNetwork(train_env.obs_size).to(device)
    agents.reset_params(value_fn)

    # ─── optimizer ─────────────────────────────────────────────────────────────
    if hasattr(policy, "get_parameters"):
        expert_params, shared_params = policy.get_parameters()
    else:
        expert_params, shared_params = [], list(policy.parameters())
    optimizer = t.optim.Adam(
        [
            {"params": expert_params, "weight_decay": expert_weight_decay},
            {
                "params": shared_params + list(value_fn.parameters()),
                "weight_decay": shared_weight_decay,
            },
        ],
        lr=learning_rate,
    )

    # ─── log buffers ───────────────────────────────────────────────────────────
    eval_policies = {"training_policy": policy}
    if hasattr(policy, "get_diamond_policy"):
        eval_policies["diamond"] = policy.get_diamond_policy()
        eval_policies["ghost"] = policy.get_ghost_policy()

    # ─── initialize row-wise logs ───────────────────────────────────────────
    train_rows = []
    eval_rows = []

    global_step = 0
    for update_idx in tqdm.trange(num_learning_updates):
        # 1) collect & update as before
        proc = generate_and_process_batch(
            train_env,
            policy,
            reward_fn_to_train_on,
            discount,
            steps_per_learning_update,
            device,
        )
        loss, batch_stats = loss_getter_fn(
            proc,
            policy,
            value_fn,
            coefs={
                k: (v(update_idx) if callable(v) else v) for k, v in loss_coefs.items()
            },
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # increment total steps seen
        global_step += steps_per_learning_update * env_kwargs.get("n_envs", 1)

        # 2) record one train-row
        train_return = float(proc["returns"].mean().item())
        row = {
            "update_idx": update_idx,
            "global_step": global_step,
            "train_return": train_return,
        }

        # 3) hold-out eval for early-stop
        if stopper and (update_idx % eval_freq == 0):
            policy.eval()  # ⟵ switch to eval mode
            stats_hold = eval(
                policy,
                env_kwargs,
                env_kwargs["max_step"],
                true_reward_fn,
                discount,
                device,
                env=val_env,
            )
            policy.train()  # ⟵ back to train mode

            row["holdout_return"] = stats_hold["avg_return"]
            if stopper.step(stats_hold["avg_return"]):
                print(
                    f"[{run_label}] early-stopping at update {update_idx},"
                    f" hold-out={stats_hold['avg_return']:.4f}"
                )
                train_rows.append(row)
                break
        else:
            row["holdout_return"] = float("nan")

        train_rows.append(row)

        # 4) periodic full-env eval
        is_final_step = update_idx == num_learning_updates - 1
        if (update_idx % eval_freq == 0) or is_final_step:
            for label, pol in eval_policies.items():
                stats_eval = eval(
                    pol,
                    env_kwargs,
                    env_kwargs["max_step"],
                    true_reward_fn,
                    discount,
                    device,
                )
                eval_rows.append(
                    {
                        "update_idx": update_idx,
                        "global_step": global_step,
                        "policy_type": label,
                        "eval_return": stats_eval["avg_return"],
                    }
                )

        # 5) visualize gate / experts if desired
        if update_idx % policy_log_freq == 0 or is_final_step:
            if type(policy) is agents.RoutedPolicyNetwork:
                update_idx_pad = str(update_idx).zfill(8)
                diagnostics.visualize_expert_policies(
                    policy,
                    env_kwargs["nrows"],
                    env_kwargs["ncols"],
                    ghost_loc=(2, 3),
                    diamond_loc=(0, 1),
                    oversight_tensor=t.zeros(
                        (env_kwargs["nrows"], env_kwargs["ncols"])
                    ),
                    title=f"run id: {run_id}, update step: {update_idx}",
                    save_path=os.path.join(
                        policy_visualization_dir,
                        f"policy_{run_id}_{update_idx_pad}.png",
                    ),
                    progress=(update_idx + 1) / num_learning_updates,
                )

    # ─── convert row logs to DataFrames & save ───────────────────────────────
    train_df = pd.DataFrame(train_rows).set_index("update_idx")
    train_df.insert(0, "run_label", run_label)
    train_df.insert(1, "oversight_prob", env_kwargs.get("oversight_prob"))
    train_df["run_id"] = run_id
    train_df.to_csv(os.path.join(save_dir, f"train_results_{run_id}.csv"))

    eval_df = pd.DataFrame(eval_rows).set_index("update_idx")
    eval_df.insert(0, "run_label", run_label)
    eval_df.insert(1, "oversight_prob", env_kwargs.get("oversight_prob"))
    eval_df["run_id"] = run_id
    eval_df.to_csv(os.path.join(save_dir, f"eval_results_{run_id}.csv"))

    t.cuda.empty_cache()
    time.sleep(time_to_sleep_after_run)
