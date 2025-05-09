import math
import os
from datetime import datetime

import pandas as pd
import torch
import torch.optim as optim
from agents import MLP, PolicyNetwork, reset_params
from grid import ContinuingEnv

# from projects.minigrid_repro.agents import MLP, PolicyNetwork, reset_params
# from projects.minigrid_repro.grid import ContinuingEnv

# Hierarchical RL Agent for 5×5 GridWorld (diamond vs. ghost)
# High-level action: 0 = target DIAMOND, 1 = target GHOST
# Low-level: MLP policy per target, trained by REINFORCE with shaped reward

H_L = 32  # low-level horizon (max steps per episode)
H_H = 1  # high-level horizon (one decision per episode)
KAPPA = 2 * H_H * H_L  # goal-reach bonus
GAMMA = 0.97  # discount for true reward


class LLPolicy(PolicyNetwork):
    def __init__(self, obs_size, action_dim, hidden=128):
        super().__init__()
        # MLP: obs_size -> hidden -> action_dim
        self.net = MLP([obs_size, hidden, action_dim], use_relu_on_output=False)

    def get_action_logits(self, obs_stack: torch.Tensor):
        # obs_stack: (1,4,5,5) -> flatten
        x = obs_stack.view(obs_stack.size(0), -1)
        return self.net(x)


class HierarchicalAgent:
    def __init__(
        self, oversight_prob: float, run_label="HierUCBVI", save_dir="logs", device=None
    ):
        self.oversight_prob = oversight_prob
        self.run_label = run_label
        self.save_dir = save_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(self.save_dir, exist_ok=True)

        # single-env setup
        self.env_kwargs = dict(
            n_envs=1,
            nrows=5,
            ncols=5,
            max_step=H_L,
            oversight_prob=oversight_prob,
            spurious_oversight_prob=0.0,
            device=device or torch.device("cpu"),
        )
        # statistics for two high-level actions
        self.stats = {
            0: {"n": 0, "r_sum": 0.0},  # DIAMOND
            1: {"n": 0, "r_sum": 0.0},  # GHOST
        }
        # low-level policies and optimizers per action
        self.policies = {}
        self.optimizers = {}
        self.total_visits = 0
        # track eval performance
        self.eval_episodes = []
        self.eval_returns = []

    def select_highlevel(self):
        # UCB over two actions
        best = None
        best_score = -float("inf")
        for a in [0, 1]:
            st = self.stats[a]
            n = st["n"]
            mean_r = st["r_sum"] / n if n > 0 else 0.0
            bonus = math.sqrt(math.log(self.total_visits + 1) / (1 + n))
            score = mean_r + bonus
            if score > best_score:
                best_score = score
                best = a
        return best

    def train(self, episodes=2000, eval_interval=500, lr=1e-3):
        eval_metrics = []
        for ep in range(1, episodes + 1):
            self.total_visits += 1
            # new episode, random diamond/ghost positions
            env = ContinuingEnv(**self.env_kwargs)  # type: ignore
            obs, _ = env.reset()

            # high-level decision
            action_high = self.select_highlevel()
            # init policy if needed
            if action_high not in self.policies:
                policy = LLPolicy(env.obs_size, 4).to(self.env_kwargs["device"])
                reset_params(policy)
                opt = optim.Adam(policy.parameters(), lr=lr)
                self.policies[action_high] = policy
                self.optimizers[action_high] = opt
            policy = self.policies[action_high]
            opt = self.optimizers[action_high]

            # rollout low-level
            log_probs = []
            rewards = []
            for t in range(H_L):
                logits = policy.get_action_logits(obs)
                dist = torch.distributions.Categorical(logits=logits)
                a_ll = dist.sample()
                log_probs.append(dist.log_prob(a_ll).squeeze())
                obs, info, done = env.step(a_ll)
                # compute base true reward
                reached_d = info["reached_diamond"][0].item() == 1.0
                reached_g = info["reached_ghost"][0].item() == 1.0
                base_r = float(reached_d) - float(reached_g)
                # shaped bonus if achieved chosen goal on terminal
                bonus = 0.0
                if done[0].item():
                    if (action_high == 0 and reached_d) or (
                        action_high == 1 and reached_g
                    ):
                        bonus = KAPPA
                r = base_r + bonus
                rewards.append(r)
                if done[0].item():
                    break

            # REINFORCE update with shaped rewards
            returns = []
            R = 0.0
            for r in reversed(rewards):
                R = r + GAMMA * R
                returns.insert(0, R)
            returns = torch.tensor(returns, device=self.env_kwargs["device"])
            logp = torch.stack(log_probs)
            loss = -(logp * returns).sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

            # update stats
            self.stats[action_high]["n"] += 1  # type: ignore
            self.stats[action_high]["r_sum"] += sum(rewards)  # type: ignore

            # periodic evaluation
            if ep % eval_interval == 0:
                avg_ret = self.evaluate()
                self.eval_episodes.append(ep)
                self.eval_returns.append(avg_ret)
                print(f"Episode {ep:5d} | Eval avg return: {avg_ret:.3f}")
                eval_metrics.append({"update_idx": ep, "eval_avg_return": avg_ret})

        print("\n=== Training complete ===")
        print(
            f"Final ground-truth return @ episode {self.eval_episodes[-1]}: {self.eval_returns[-1]:.3f}"
        )

        # Save eval results
        eval_df = pd.DataFrame(eval_metrics).set_index("update_idx")
        eval_df["run_id"] = self.run_id
        eval_df.insert(0, "run_label", self.run_label)
        eval_df.insert(1, "oversight_prob", self.oversight_prob)
        eval_df.to_csv(os.path.join(self.save_dir, f"eval_results_{self.run_id}.csv"))

    def evaluate(self, runs=512):
        total = 0.0
        for _ in range(runs):
            env = ContinuingEnv(**{**self.env_kwargs, "oversight_prob": 1.0})
            obs, _ = env.reset()
            # greedy high-level
            action_high = max(
                [0, 1],
                key=lambda a: (self.stats[a]["r_sum"] / self.stats[a]["n"])
                if self.stats[a]["n"] > 0
                else 0.0,
            )
            policy = self.policies.get(action_high)
            R = 0.0
            for t in range(H_L):
                logits = policy.get_action_logits(obs)  # type: ignore
                a_ll = torch.argmax(logits, dim=-1)
                obs, info, done = env.step(a_ll)
                # true reward
                reached_d = info["reached_diamond"][0].item() == 1.0
                reached_g = info["reached_ghost"][0].item() == 1.0
                base_r = float(reached_d) - float(reached_g)
                R += (GAMMA**t) * base_r
                if done[0].item():
                    break
            total += R
        return total / runs


# usage:
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data")
    experiment_name = "oversight_levels"
    agent = HierarchicalAgent(
        oversight_prob=0.3,
        device=torch.device("cpu"),
        save_dir=os.path.join(data_dir, experiment_name),
    )
    agent.train(episodes=2000, eval_interval=5)
