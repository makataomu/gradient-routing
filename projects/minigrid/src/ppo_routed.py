# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import dotenv
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import wandb
from jaxtyping import Float
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tyro.conf import subcommand

from projects.minigrid.src.factrep import utils
from projects.minigrid.src.factrep.environments.basic import BasicEnv, BasicEnvConfig
from projects.minigrid.src.factrep.environments.partial_oversight import (
    PartialOversightEnv,
    PartialOversightEnvConfig,
)
from projects.minigrid.src.factrep.evaluation import evaluate, plot_policy
from projects.minigrid.src.factrep.models import Agent, AgentConfig


def default_agent_config() -> AgentConfig:
    return AgentConfig(
        action_space_dim=4,
        architecture="original",
        teacher_forcing=False,
        shifting_alphas=False,
        negative_gradient=False,
        gating_enabled=True,
        routing_enabled=True,
        n_shards=2,
    )


@dataclass
class Config:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1337
    """seed for the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: int | None = 1
    """if toggled, cuda will be enabled by default"""
    wandb_project_name: str = "rl-v2.0"
    """the wandb's project name"""
    wandb_entity: str = "team-shard"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save the model or not"""
    save_model_every: int = 128
    """how often to save the model"""
    eval_every_n_iterations: int = 4
    """how often to evaluate the agent"""
    only_train_when_under_oversight: bool = False
    """if toggled, the agent will only train on episodes under oversight"""

    oversight_ordering: Literal["oversight_first", "oversight_last", "random"] = (
        "random"
    )
    ordering_flip: float = 0.5

    # Model specific arguments
    agent: AgentConfig = field(default_factory=default_agent_config)

    # Logging
    log_field_every_n_iterations: int = 16
    log_field_video_every_n_iterations: int = 1000
    log_field_for_seeds: list[int] = field(default_factory=lambda: [7, 13, 1337])

    envs: list[
        Annotated[BasicEnvConfig, subcommand("basic")]
        | Annotated[PartialOversightEnvConfig, subcommand("partial_oversight")]
    ] = field(
        default_factory=lambda: [
            PartialOversightEnvConfig(
                width=7,
                height=7,
                ratio_oversight=0.25,
                rewards=[(1, 1), (1, 1)],
                terminal_counts=None,
                randomize_terminal_kinds=True,
                n_terminals_per_kind=1,
                terminal_probabilities=[(0.5, 0.5), (0.5, 0.5)],
                min_distance=3,
                require_confirmation=False,
                has_unique_target=True,
                has_target_in_input=False,
                randomize_agent_start=True,
                pov_observation=False,
                agent_view_size=5,
                render_mode="rgb_array",
            )
        ]
    )
    envs_marks: list[float] = field(default_factory=lambda: [])

    eval_env: (
        Annotated[BasicEnvConfig, subcommand("basic")]
        | Annotated[PartialOversightEnvConfig, subcommand("partial_oversight")]
    ) = field(
        default_factory=lambda: PartialOversightEnvConfig(
            width=7,
            height=7,
            ratio_oversight=0.25,
            rewards=[(1, 1), (-1, -1)],
            terminal_counts=None,
            n_terminals_per_kind=1,
            terminal_probabilities=[(0.5, 0.5), (0.5, 0.5)],
            randomize_terminal_kinds=True,
            min_distance=3,
            require_confirmation=False,
            has_unique_target=True,
            has_target_in_input=False,
            randomize_agent_start=True,
            pov_observation=False,
            agent_view_size=5,
            render_mode="rgb_array",
        )
    )

    # Loss arguments, have to be named as `<loss_name>_coef`, where
    # <loss_name> matches loss logged by the model
    alpha_coef: float = 0.3
    l1_coef: float = 0.0
    l2_coef: float = 0.0

    # Algorithm specific arguments
    total_timesteps: int = 1_500_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.97
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def video_schedule(episode_id: int) -> bool:
    rounded_episode_id: float = episode_id ** (1.0 / 3)
    return int(round(rounded_episode_id)) ** 3 == episode_id


def safe_mean(x: Float[torch.Tensor, "batch"]) -> Float[torch.Tensor, ""]:
    return x.sum() / x.numel()


def main(config: Config):
    seed = config.seed
    config.batch_size = int(config.num_envs * config.num_steps)
    config.minibatch_size = int(config.batch_size // config.num_minibatches)
    config.num_iterations = config.total_timesteps // config.batch_size
    run_name = f"{config.exp_name}__{seed}__{int(time.time())}"

    _ = dotenv.load_dotenv()
    wandb.require("core")

    _ = wandb.init(
        project=config.wandb_project_name,
        entity=config.wandb_entity,
        config=vars(config),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device(
        f"cuda:{config.cuda}"
        if torch.cuda.is_available() and config.cuda is not None
        else "cpu"
    )

    env_config = config.envs.pop(0)
    eval_env_config = config.eval_env

    if isinstance(env_config, BasicEnvConfig) and isinstance(
        eval_env_config, BasicEnvConfig
    ):
        make_env = lambda: BasicEnv.from_config(env_config)
        make_eval_env = lambda: BasicEnv.from_config(eval_env_config)
    elif isinstance(env_config, PartialOversightEnvConfig) and isinstance(
        eval_env_config, PartialOversightEnvConfig
    ):
        make_env = lambda: PartialOversightEnv.from_config(env_config)
        make_eval_env = lambda: PartialOversightEnv.from_config(eval_env_config)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            utils.wrap_in_loggers(
                make_env, run_name, video_schedule if i == 0 else lambda _: False
            )
            for i in range(config.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    n_shards = envs.envs[0].n_terminal_kinds

    config.agent.action_space_dim = int(envs.single_action_space.n)  # type: ignore
    config.agent.n_shards = n_shards
    print(config.agent)
    agent = Agent(config.agent).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (config.num_steps, config.num_envs) + envs.single_observation_space.shape  # type: ignore
    ).to(device)
    actions = torch.zeros(
        (config.num_steps, config.num_envs) + envs.single_action_space.shape  # type: ignore
    ).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)
    # Steps from incomplete episodes are not considered for backpropagation
    complete_episodes = torch.zeros(config.num_steps, config.num_envs).bool().to(device)
    # Source -1 (default) means that the step does not have a traceable reward
    # its gradient will propagate through normally without masking
    true_alphas = -torch.ones(config.num_steps, config.num_envs, n_shards).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.num_envs).to(device)

    action_field_dir = Path(wandb.run.dir) / "action_field"  # type: ignore

    for iteration in range(1, config.num_iterations + 1):
        if (
            config.envs_marks
            and iteration > config.envs_marks[0] * config.num_iterations
        ):
            _ = config.envs_marks.pop(0)
            make_env = lambda: PartialOversightEnv.from_config(env_config)

            # env setup
            envs = gym.vector.SyncVectorEnv(
                [
                    utils.wrap_in_loggers(make_env, run_name, video_schedule)
                    for _ in range(config.num_envs)
                ]
            )
            next_obs, _ = envs.reset(seed=seed)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(config.num_envs).to(device)

        # Log metrics
        is_last_iteration = iteration == config.num_iterations
        if is_last_iteration or iteration % config.eval_every_n_iterations == 0:
            eval_envs = gym.vector.SyncVectorEnv([make_eval_env])
            _ = agent.eval()
            wandb.log(
                evaluate(agent, eval_envs, device, gamma=config.gamma, n=100),
                step=global_step,
            )
            _ = agent.train()

        # Log field
        if is_last_iteration or iteration % config.log_field_every_n_iterations == 0:
            agent.eval()
            env = make_eval_env()
            for eval_seed in config.log_field_for_seeds:
                env.reset(seed=eval_seed)
                field = plot_policy(
                    env,
                    agent,
                    device,
                    require_confirmation=env_config.require_confirmation,
                )
                image_path = (
                    action_field_dir / f"seed={eval_seed}" / f"iter={iteration}.png"
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                wandb.log(
                    {f"action_field/snapshot/seed={eval_seed}": wandb.Image(field)},
                    step=global_step,
                )
                field.savefig(image_path)
            agent.train()

        # Log field timelapse
        if (
            is_last_iteration
            or iteration % config.log_field_video_every_n_iterations == 0
        ):
            agent.eval()
            for eval_seed in config.log_field_for_seeds:
                image_files = [
                    str(p)
                    for p in sorted(
                        (action_field_dir / f"seed={eval_seed}").glob("*.png"),
                        key=lambda p: p.stat().st_ctime,
                    )
                ]
                image_files = [str(p) for p in image_files]
                clip = ImageSequenceClip(image_files, fps=4)
                video_path = (
                    action_field_dir / f"seed={eval_seed}" / f"iter={iteration}.mp4"
                )
                clip.write_videofile(str(video_path))
                wandb.log(
                    {
                        f"action_field/timelapse/seed={eval_seed}": wandb.Video(
                            str(video_path)
                        )
                    },
                    step=global_step,
                )
            agent.train()

        # Annealing the rate if instructed to do so.
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.num_iterations
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Do args.num_steps steps in the environment
        true_alphas[:] = -1
        complete_episodes[:] = False
        subtrajectory_beginning = torch.zeros(config.num_envs).int().to(device)

        if config.only_train_when_under_oversight:
            assert (
                config.num_envs == 1
            ), "num_envs must be 1 for only_train_when_under_oversight"

        step = 0
        while step < config.num_steps:
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            # if terminations.all().item():
            #     print(f"{subtrajectory_beginning[0]}–{step=}, {reward=}, ")
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            # Setting true_alphas during the episode
            # for env_idx, env_true_alphas in enumerate(infos.get("true_alphas", [])):
            #     # This checks for whether the true_alphas is set in the specific environment
            #     if infos["_true_alphas"][env_idx]:
            #         span = slice(subtrajectory_beginning[env_idx], step + 1)
            #         true_alphas[span, env_idx] = env_true_alphas
            #         subtrajectory_beginning[env_idx] = step + 1

            # Setting true_alphas and logging at the end of the episode
            if "final_info" in infos:
                for env_idx, env_info in enumerate(infos["final_info"]):
                    if env_info is None:
                        continue

                    # under_oversight = env_info["metadata"].get("found_seen", True)
                    # if config.only_train_when_under_oversight and not under_oversight:
                    #     step = subtrajectory_beginning[env_idx] - 1
                    #     next_done = dones[step + 1]
                    #     # print("=== NO OVERSIGHT, skipping episode ===")
                    #     # print(f"Going back to {subtrajectory_beginning[0]}")
                    #     # print(env_info)
                    #     # print()
                    #     continue

                    if "episode" in env_info:
                        print(
                            f"global_step={global_step}, episodic_return={env_info['episode']['r']}"
                        )
                        metrics = env_info["metrics"] | {
                            "env/episodic_return": env_info["episode"]["r"],
                            "env/episodic_discounted_return": env_info["episode"]["r"]
                            * (config.gamma ** env_info["episode"]["l"]),
                            "env/episodic_length": env_info["episode"]["l"],
                        }
                        if (env_true_alphas := env_info.get("true_alphas")) is not None:
                            span = slice(subtrajectory_beginning[env_idx], step + 1)
                            true_alphas[span, env_idx] = env_true_alphas
                            # print("=== Finished episode and used it for training ===")
                            # print(f"Used {span} for training")
                            # print(env_info)
                            # print()

                        wandb.log(metrics, step=global_step)

                    complete_episodes[: step + 1] = complete_episodes[: step + 1] | (
                        next_done.bool()
                    )
                    subtrajectory_beginning = torch.where(
                        next_done.bool(),
                        torch.ones_like(subtrajectory_beginning).int() * (step + 1),
                        subtrajectory_beginning,
                    )

            step += 1

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)  # type: ignore
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)  # type: ignore
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_true_alphas = true_alphas.reshape(-1, n_shards)
        b_complete_episodes = complete_episodes.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config.batch_size)
        b_inds = b_inds[b_complete_episodes.cpu().numpy()]

        clipfracs = []
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]
                if len(mb_inds) == 0:
                    continue

                _, newlogprob, entropy, newvalue, agent_info = (
                    agent.get_action_and_value(
                        b_obs[mb_inds],
                        true_alphas=b_true_alphas[mb_inds],
                        action=b_actions.long()[mb_inds],
                    )
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv and len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = (
                    pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
                )
                for name, val in agent_info["loss"].items():
                    coef = config.__dict__.get(f"{name}_coef")
                    loss += coef * val

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None and approx_kl > config.target_kl:  # type: ignore
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if "v_loss" not in vars():
            v_loss = torch.Tensor([torch.nan])
            pg_loss = torch.Tensor([torch.nan])
            entropy_loss = torch.Tensor([torch.nan])
            old_approx_kl = torch.Tensor([torch.nan])
            approx_kl = torch.Tensor([torch.nan])
            agent_info = {"loss": {}, "metrics": {}}
            clipfracs = []
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        log_data = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),  # type: ignore
            "losses/policy_loss": pg_loss.item(),  # type: ignore
            "losses/entropy": entropy_loss.item(),  # type: ignore
            "losses/old_approx_kl": old_approx_kl.item(),  # type: ignore
            "losses/approx_kl": approx_kl.item(),  # type: ignore
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "env/true_alpha=def": (b_true_alphas[:, 0] == -1).sum().item(),
            "env/true_alpha=0": (b_true_alphas[:, 0] == 1).sum().item(),
            "env/true_alpha=1": (b_true_alphas[:, 1] == 1).sum().item(),
        }
        for name, val in agent_info["loss"].items():  # type: ignore
            log_data[f"losses/{name}"] = val.item()
        for name, val in agent_info["metrics"].items():  # type: ignore
            log_data[f"model/{name}"] = val

        wandb.log(log_data, step=global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

        if (iteration + 1) % config.save_model_every == 0 and config.save_model:
            model_path = f"projects/minigrid/models/{run_name}/{config.exp_name}_iter={iteration + 1}.cleanrl_model"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            agent.save(model_path, metadata={"env_config": asdict(env_config)})
            print(f"model saved to {model_path}")

    if config.save_model:
        model_path = f"projects/minigrid/models/{run_name}/{config.exp_name}_iter={iteration + 1}.cleanrl_model"  # type: ignore
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        agent.save(model_path, metadata={"env_config": asdict(env_config)})
        print(f"model saved to {model_path}")

    envs.close()

    wandb.finish()


if __name__ == "__main__":
    original_config = tyro.cli(Config)

    experiments = [
        {
            "exp_name": "001p",
            "envs.0.terminal_probabilities": [(0.999, 0.001), (0.999, 0.001)],
        },
        {
            "exp_name": "003p",
            "envs.0.terminal_probabilities": [(0.997, 0.003), (0.997, 0.003)],
        },
        {
            "exp_name": "01p",
            "envs.0.terminal_probabilities": [(0.99, 0.01), (0.99, 0.01)],
        },
        {
            "exp_name": "02p",
            "envs.0.terminal_probabilities": [(0.98, 0.02), (0.98, 0.02)],
        },
        {
            "exp_name": "03p",
            "envs.0.terminal_probabilities": [(0.97, 0.03), (0.97, 0.03)],
        },
        {
            "exp_name": "04p",
            "envs.0.terminal_probabilities": [(0.96, 0.04), (0.96, 0.04)],
        },
        {
            "exp_name": "05p",
            "envs.0.terminal_probabilities": [(0.95, 0.05), (0.95, 0.05)],
        },
        {
            "exp_name": "10p",
            "envs.0.terminal_probabilities": [(0.9, 0.1), (0.9, 0.1)],
        },
        {
            "exp_name": "20p",
            "envs.0.terminal_probabilities": [(0.8, 0.2), (0.8, 0.2)],
        },
        {
            "exp_name": "30p",
            "envs.0.terminal_probabilities": [(0.7, 0.3), (0.7, 0.3)],
        },
        {
            "exp_name": "40p",
            "envs.0.terminal_probabilities": [(0.6, 0.4), (0.6, 0.4)],
        },
        {
            "exp_name": "50p",
            "envs.0.terminal_probabilities": [(0.5, 0.5), (0.5, 0.5)],
        },
        {
            "exp_name": "60p",
            "envs.0.terminal_probabilities": [(0.4, 0.6), (0.4, 0.6)],
        },
        {
            "exp_name": "70p",
            "envs.0.terminal_probabilities": [(0.3, 0.7), (0.3, 0.7)],
        },
        {
            "exp_name": "80p",
            "envs.0.terminal_probabilities": [(0.2, 0.8), (0.2, 0.8)],
        },
        {
            "exp_name": "90p",
            "envs.0.terminal_probabilities": [(0.1, 0.9), (0.1, 0.9)],
        },
        {
            "exp_name": "999p",
            "envs.0.terminal_probabilities": [(0.001, 0.999), (0.001, 0.999)],
        },
    ]

    for seed in [1337, 42]:
        for exp in experiments:
            config = deepcopy(original_config)
            config.seed = seed
            for k, v in exp.items():
                setting = config
                *path, attr = k.split(".")
                for k in path:
                    try:
                        k = int(k)
                        if len(setting) > k:
                            setting[k] = deepcopy(setting[0])
                            setting = setting[k]
                    except ValueError:
                        setting = getattr(setting, k)
                setattr(setting, attr, v)

            main(config)
