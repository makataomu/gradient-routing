#!/usr/bin/env python3
"""
bulk_runs_double_descent.py

Depth sweep for long-horizon training to look for double-descent.

- No regularisers / early stopping / holdout.
- Uses training.train() exactly as defined in your repo.
- Saves train/eval CSVs on the go in save_dir (as train() already does).
- Encodes experiment metadata into run_label.

Example (Kaggle / CLI):
  pdm run python projects/minigrid_repro/bulk_runs_double_descent.py \
    --run_types naive_outcomes routing no_routing_control \
    --oversight_probs 0.001 0.003 0.01 \
    --depths 1 2 4 6 \
    --width 256 \
    --num_steps 50000 \
    --seed_offset 0 \
    --num_runs_per_config 4
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List

import torch

try:
    import projects.minigrid_repro.agents as agents
    import projects.minigrid_repro.training as training
except ImportError:
    import agents as agents
    import training as training


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Double-descent depth sweep runner")

    p.add_argument(
        "--run_types",
        nargs="+",
        default=["naive_outcomes"],
        help="Training variants. Supported: naive_outcomes, routing, no_routing_control, filtering, oracle",
    )
    p.add_argument(
        "--oversight_probs",
        type=float,
        nargs="+",
        default=[0.01],
        help="Values assigned to env_kwargs['oversight_prob'].",
    )
    p.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Number of hidden layers in the (expert / policy) MLP.",
    )
    p.add_argument("--width", type=int, default=256, help="Hidden width per layer.")
    p.add_argument("--num_steps", type=int, default=20_000, help="num_learning_updates")
    p.add_argument("--steps_per_learning_update", type=int, default=32)
    p.add_argument("--discount", type=float, default=0.97)
    p.add_argument("--eval_freq", type=int, default=100)
    p.add_argument("--policy_log_freq", type=int, default=200)

    p.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit seeds. If omitted, uses seed_offset..",
    )
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--num_runs_per_config", type=int, default=4)

    p.add_argument("--n_envs", type=int, default=512, help="env_kwargs['n_envs']")
    p.add_argument("--nrows", type=int, default=5)
    p.add_argument("--ncols", type=int, default=5)
    p.add_argument("--max_step", type=int, default=32)

    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--expert_weight_decay", type=float, default=0.0)
    p.add_argument("--shared_weight_decay", type=float, default=0.0)

    p.add_argument(
        "--save_dir",
        type=str,
        default=str(PROJECT_DIR / "data" / "double_descent_depth"),
        help="Where train() writes CSV outputs.",
    )
    p.add_argument(
        "--fig_dir",
        type=str,
        default=str(PROJECT_DIR / "figures" / "double_descent_depth"),
        help="Where train() writes policy visualizations.",
    )

    return p.parse_args()


def make_policy_ctor(
    run_type: str, depth: int, width: int
) -> Callable[[int, int], agents.PolicyNetwork]:
    """
    Builds a constructor(obs_dim, num_actions) -> PolicyNetwork with chosen depth/width.

    - routing / no_routing_control: RoutedPolicyNetwork with gate; expert_hidden and gate_hidden deep.
    - others (naive_outcomes / filtering / oracle): single-expert diamond policy, but expert MLP is deep.
    """
    expert_hidden = tuple([width] * depth)
    gate_hidden = tuple([width] * depth) + (1,)

    if run_type in ("routing", "no_routing_control"):
        use_gate = True
        use_gradient_routing = run_type == "routing"

        def ctor(obs_dim: int, num_actions: int) -> agents.PolicyNetwork:
            return agents.RoutedPolicyNetwork(
                obs_dim=obs_dim,
                num_actions=num_actions,
                use_gate=use_gate,
                use_gradient_routing=use_gradient_routing,
                expert_hidden=expert_hidden,
                gate_hidden=gate_hidden,
            )

        return ctor

    # single expert (diamond)
    def ctor(obs_dim: int, num_actions: int) -> agents.PolicyNetwork:
        routed = agents.RoutedPolicyNetwork(
            obs_dim=obs_dim,
            num_actions=num_actions,
            use_gate=False,
            use_gradient_routing=False,
            expert_hidden=expert_hidden,
        )
        return routed.get_diamond_policy()

    return ctor


def algo_settings_by_run_type() -> Dict[str, Dict]:
    """
    Minimal mapping: which reward + loss to use per run_type.
    (Matches your earlier bulk_runs patterns.)
    """
    return {
        "routing": dict(
            reward_fn_to_train_on=training.moe_reward_fn,
            loss_getter_fn=agents.get_routed_reinforce_loss,
        ),
        "no_routing_control": dict(
            reward_fn_to_train_on=training.moe_reward_fn,
            loss_getter_fn=agents.get_routed_reinforce_loss,
        ),
        "naive_outcomes": dict(
            reward_fn_to_train_on=training.naive_reward_fn,
            loss_getter_fn=agents.get_reinforce_loss,
        ),
        "filtering": dict(
            reward_fn_to_train_on=training.true_reward_fn,
            loss_getter_fn=agents.get_filtered_reinforce_loss,
        ),
        "oracle": dict(
            reward_fn_to_train_on=training.true_reward_fn,
            loss_getter_fn=agents.get_reinforce_loss,
        ),
    }


def build_runs(args: argparse.Namespace) -> List[Dict]:
    # same coef defaults you used earlier
    loss_coefs = {
        "entropy_bonus": 0.1,
        "value_loss": 0.5,
        "gate_loss": 0.01,
        "gate_loss_no_oversight": 0,
    }

    # seeds
    if args.seeds is not None and len(args.seeds) > 0:
        seeds = list(args.seeds)
    else:
        seeds = list(
            range(args.seed_offset, args.seed_offset + args.num_runs_per_config)
        )

    base_env_kwargs = dict(
        n_envs=args.n_envs,
        nrows=args.nrows,
        ncols=args.ncols,
        max_step=args.max_step,
        oversight_prob=None,
        spurious_oversight_prob=0,
    )

    settings = algo_settings_by_run_type()
    for rt in args.run_types:
        if rt not in settings:
            raise ValueError(
                f"Unknown run_type '{rt}'. Choose from: {sorted(settings.keys())}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs: List[Dict] = []
    for ovs in args.oversight_probs:
        env_kwargs = deepcopy(base_env_kwargs)
        env_kwargs["oversight_prob"] = ovs

        for run_type in args.run_types:
            for depth in args.depths:
                policy_ctor = make_policy_ctor(run_type, depth, args.width)

                # Encode “experiment name” into run_label (as you requested)
                # run_id will be seed (unique per run)
                base_label = f"{run_type}|ovs={ovs:g}|depth={depth}|width={args.width}|steps={args.num_steps}"

                for seed in seeds:
                    kw = dict(
                        steps_per_learning_update=args.steps_per_learning_update,
                        num_learning_updates=args.num_steps,
                        eval_freq=args.eval_freq,
                        policy_log_freq=args.policy_log_freq,
                        discount=args.discount,
                        loss_coefs=loss_coefs,
                        learning_rate=args.learning_rate,
                        expert_weight_decay=args.expert_weight_decay,
                        shared_weight_decay=args.shared_weight_decay,
                        policy_network_constructor=policy_ctor,
                        reward_fn_to_train_on=settings[run_type][
                            "reward_fn_to_train_on"
                        ],
                        loss_getter_fn=settings[run_type]["loss_getter_fn"],
                        env_kwargs=deepcopy(env_kwargs),
                        save_dir=args.save_dir,
                        policy_visualization_dir=args.fig_dir,
                        run_label=base_label,
                        run_id=seed,  # unique ID
                        device=device,
                        regulariser_name=None,
                        regulariser_kwargs=None,
                        random_seed=False,
                        default_seed=seed,
                    )
                    runs.append(kw)

    return runs


def main() -> None:
    args = parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    runs = build_runs(args)
    total = len(runs)
    print(f"Prepared {total} runs.")
    print(f"Saving CSVs to: {args.save_dir}")
    print(f"Saving figures to: {args.fig_dir}")

    for i, kw in enumerate(runs, 1):
        print(f"\n=== Run {i}/{total} ===")
        print(f"run_label: {kw['run_label']}")
        print(f"run_id   : {kw['run_id']}")
        print(f"seed     : {kw['default_seed']}")
        training.train(**kw)


if __name__ == "__main__":
    main()
