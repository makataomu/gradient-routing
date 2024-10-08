# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical

import wandb
from projects.minigrid.src.factrep import utils
from projects.minigrid.src.factrep.environments.partial_oversight import (
    PartialOversightEnv,
    PartialOversightEnvConfig,
)
from projects.minigrid.src.factrep.evaluation import evaluate


@dataclass
class Args:
    exp_name: str = "7x7_100p_small_no_gating_newppo"
    """the name of this experiment"""
    seed: int = 1337
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: int = 1
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rl-v2.0"
    """the wandb's project name"""
    wandb_entity: str = "team-shard"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save the model or not"""
    save_model_every: int = 100

    # Algorithm specific arguments
    total_timesteps: int = 1_500_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 1024
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
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.critic = nn.Sequential(
            nn.LazyConv2d(128, 2, padding="same"),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            nn.LazyConv2d(128, 2, padding="same"),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            layer_init(nn.Linear(128, envs.single_action_space.n), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, true_alphas=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def save(self, path: Path | str, metadata={}):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "metadata": metadata,
            },
            path,
        )

    @staticmethod
    def load(path: Path | str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(path, map_location=device)
        env_config = PartialOversightEnvConfig(
            width=7,
            height=7,
            ratio_oversight=0.25,
            randomize_terminal_kinds=True,
            n_terminals_per_kind=1,
            terminal_probabilities=[(0.8, 0.2), (0.8, 0.2)],
            rewards=[(0, 1), (0, -1)],
            terminal_counts=None,
            require_confirmation=False,
            min_distance=3,
            has_unique_target=True,
            has_target_in_input=False,
            randomize_agent_start=True,
            pov_observation=False,
            agent_view_size=5,
            render_mode="rgb_array",
        )
        make_env = lambda: PartialOversightEnv.from_config(env_config)
        # env setup
        envs = gym.vector.SyncVectorEnv([make_env])
        agent = Agent(envs)
        agent.load_state_dict(model["state_dict"])
        agent.metadata = model["metadata"]
        return agent


def video_schedule(episode_id: int) -> bool:
    rounded_episode_id: float = episode_id ** (1.0 / 3)
    return int(round(rounded_episode_id)) ** 3 == episode_id


def main(args: Args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda is not None
        else "cpu"
    )

    env_config = PartialOversightEnvConfig(
        width=7,
        height=7,
        ratio_oversight=0.25,
        randomize_terminal_kinds=True,
        n_terminals_per_kind=1,
        terminal_probabilities=[(0, 1), (0, 1)],
        rewards=[(1, 1), (-1, -1)],
        terminal_counts=None,
        require_confirmation=False,
        min_distance=3,
        has_unique_target=True,
        has_target_in_input=False,
        randomize_agent_start=True,
        pov_observation=False,
        agent_view_size=5,
        render_mode="rgb_array",
    )

    eval_env_config = PartialOversightEnvConfig(
        width=7,
        height=7,
        ratio_oversight=0.25,
        randomize_terminal_kinds=True,
        n_terminals_per_kind=1,
        terminal_probabilities=[(0.5, 0.5), (0.5, 0.5)],
        rewards=[(1, 1), (-1, -1)],
        terminal_counts=None,
        require_confirmation=False,
        min_distance=3,
        has_unique_target=True,
        has_target_in_input=False,
        randomize_agent_start=True,
        pov_observation=False,
        agent_view_size=5,
        render_mode="rgb_array",
    )

    make_env = lambda: PartialOversightEnv.from_config(env_config)
    make_eval_env = lambda: PartialOversightEnv.from_config(eval_env_config)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            utils.wrap_in_loggers(make_env, run_name, video_schedule)
            for _ in range(args.num_envs)
        ]
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        is_last_iteration = iteration == args.num_iterations
        if is_last_iteration or iteration % 25 == 0:
            eval_envs = gym.vector.SyncVectorEnv([make_eval_env])
            _ = agent.eval()
            wandb.log(
                evaluate(agent, eval_envs, device, gamma=args.gamma, n=100),
                step=global_step,
            )
            _ = agent.train()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        subtrajectory_beginning = torch.zeros(args.num_envs).int().to(device)

        step = 0
        while step < args.num_steps:
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            if "final_info" in infos:
                for env_idx, info in enumerate(infos["final_info"]):
                    # under_oversight = info["metadata"].get("found_seen", True)
                    # if not under_oversight:
                    #     step = subtrajectory_beginning[env_idx] - 1
                    #     next_done = dones[step + 1]
                    #     continue

                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        metrics = info["metrics"] | {
                            "env/episodic_return": info["episode"]["r"],
                            "env/episodic_length": info["episode"]["l"],
                        }

                        wandb.log(metrics, step=global_step)

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
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

            if (iteration + 1) % args.save_model_every == 0:
                model_path = f"projects/minigrid/models/{run_name}/{args.exp_name}_iter={iteration + 1}.cleanrl_model"
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                agent.save(model_path, metadata={"env_config": asdict(env_config)})
                print(f"model saved to {model_path}")

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

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
        }

        wandb.log(log_data, step=global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

    model_path = f"projects/minigrid/models/{run_name}/{args.exp_name}_iter={iteration + 1}.cleanrl_model"  # type: ignore
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path, metadata={"env_config": asdict(env_config)})
    print(f"model saved to {model_path}")

    envs.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    for seed in [1337]:
        args.seed = seed
        main(args)
