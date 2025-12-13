#!/usr/bin/env python
"""
bulk_runs_double_descent.py

Sweep over model depth (and optionally oversight level / run type / seeds)
to look for double-descent behaviour.

- No regularisers / early stopping / holdout.
- Long training horizon controlled by --num_steps.
- Uses the same logging format as bulk_runs.py so the existing
  analysis scripts still work.

Drop this file into projects/minigrid_repro/ and run e.g.:

  pdm run python projects/minigrid_repro/bulk_runs_double_descent.py \
    --run_types naive_outcomes routing \
    --oversight_probs 0.001 0.003 0.01 \
    --depths 1 2 4 6 \
    --num_steps 50000 \
    --seeds 1 2 3 4
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List

import torch

try:
    import projects.minigrid_repro.agents as agents
    import projects.minigrid_repro.bulk_runs as base_bulk
    import projects.minigrid_repro.training as training
except ImportError:
    import agents as agents
    import bulk_runs as base_bulk
    import training as training


PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Depth sweep (double-descent) bulk runs for Minigrid project."
    )

    # Which algorithm(s) to run
    parser.add_argument(
        "--run_types",
        nargs="+",
        default=["naive_outcomes"],
        help=(
            "Which training methods to run. Must be keys of "
            "bulk_runs.algorithm_settings_by_run_type, e.g. "
            "naive_outcomes, routing, no_routing_control, filtering, oracle"
        ),
    )

    # Oversight levels to sweep (if relevant for your env config)
    parser.add_argument(
        "--oversight_probs",
        type=float,
        nargs="+",
        default=[0.01],
        help="Oversight probabilities to use in env_kwargs['oversight_prob'].",
    )

    # Depth sweep
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help=(
            "Number of hidden layers in each expert / policy MLP. "
            "Depth=1 means a single 256-unit hidden layer; depth=4 "
            "means 4 layers of width 256, etc."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Hidden width for each layer in the policy MLP.",
    )

    # Training horizon
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20_000,
        help="Number of learning updates (i.e. num_learning_updates). "
        "Increase this a lot for double-descent experiments.",
    )

    parser.add_argument(
        "--steps_per_learning_update",
        type=int,
        default=32,
        help="Steps per learning update (same meaning as in bulk_runs.py).",
    )

    parser.add_argument(
        "--discount",
        type=float,
        default=0.97,
        help="Discount factor gamma.",
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=100,
        help="How often to evaluate (in learning updates).",
    )

    parser.add_argument(
        "--policy_log_freq",
        type=int,
        default=200,
        help="How often to log policy snapshots (in learning updates).",
    )

    # Seeds
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Explicit list of seeds. If not given, will use "
            "range(seed_offset, seed_offset + num_runs_per_config)."
        ),
    )

    parser.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="Starting seed if --seeds is not provided.",
    )

    parser.add_argument(
        "--num_runs_per_config",
        type=int,
        default=4,
        help="Number of runs per (run_type, oversight_prob, depth) if --seeds not given.",
    )

    # Env size tweaks (optional)
    parser.add_argument(
        "--n_envs",
        type=int,
        default=None,
        help="Optional override of env_kwargs['n_envs']. If None, keep bulk_runs default.",
    )

    # Optimisation hyperparams
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for optimizer.",
    )

    parser.add_argument(
        "--expert_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay on expert parameters.",
    )

    parser.add_argument(
        "--shared_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay on shared parameters.",
    )

    # Experiment naming / dirs
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="double_descent_depth",
        help="Experiment name used in config and directory structure.",
    )

    return parser.parse_args()


def make_policy_ctor(
    run_type: str,
    depth: int,
    width: int,
) -> Callable[[int, int], agents.PolicyNetwork]:
    """
    Build a policy_network_constructor that creates deeper networks.

    For routing / no_routing_control:
        - full RoutedPolicyNetwork with gate + gradient routing toggled as usual.
    For naive_outcomes / filtering / oracle:
        - single-expert (diamond only) policy with deeper expert MLP.
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

    # Everything else uses a single expert (diamond) without gating.
    # This mimics get_single_expert_policy, just with a deeper expert MLP.
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


def build_training_kwargs_list(args: argparse.Namespace) -> List[Dict]:
    """
    Build a list of kwargs dicts to feed into training.train(**kwargs).

    We reuse bulk_runs.algorithm_settings_by_run_type for:
      - loss_coefs
      - reward functions
      - oracle env constructor
      - run_type_score
    and we override only the policy_network_constructor + some training hyperparams.
    """
    algo_settings = base_bulk.algorithm_settings_by_run_type
    base_env_kwargs = deepcopy(base_bulk.env_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = PROJECT_ROOT / "data"
    fig_root = PROJECT_ROOT / "figures"

    data_root.mkdir(exist_ok=True)
    fig_root.mkdir(exist_ok=True)

    training_kwargs_list: List[Dict] = []

    # Decide seeds per config
    if args.seeds is not None and len(args.seeds) > 0:
        seeds_for_all = list(args.seeds)
    else:
        seeds_for_all = list(
            range(args.seed_offset, args.seed_offset + args.num_runs_per_config)
        )

    for oversight_prob in args.oversight_probs:
        env_kwargs = deepcopy(base_env_kwargs)
        # Oversight probability is typically a field in env_kwargs; if not, this is a no-op.
        env_kwargs["oversight_prob"] = oversight_prob

        if args.n_envs is not None:
            env_kwargs["n_envs"] = args.n_envs

        for run_type in args.run_types:
            if run_type not in algo_settings:
                raise ValueError(
                    f"Unknown run_type '{run_type}'. "
                    f"Available: {sorted(algo_settings.keys())}"
                )

            algo_cfg = algo_settings[run_type]

            for depth in args.depths:
                policy_ctor = make_policy_ctor(run_type, depth, args.width)

                base_kwargs = dict(
                    experiment_name=args.experiment_name,
                    save_dir=str(data_root),
                    policy_visualization_dir=str(fig_root),
                    env_kwargs=env_kwargs,
                    device=device,
                    discount=args.discount,
                    steps_per_learning_update=args.steps_per_learning_update,
                    num_learning_updates=args.num_steps,
                    eval_freq=args.eval_freq,
                    policy_log_freq=args.policy_log_freq,
                    loss_coefs=algo_cfg["loss_coefs"],
                    reward_fn_to_train_on=algo_cfg["reward_fn_to_train_on"],
                    reward_fn_for_evaluating=algo_cfg["reward_fn_for_evaluating"],
                    oracle_env_constructor=algo_cfg["oracle_env_constructor"],
                    reward_fn_for_oracle_policy_eval=algo_cfg[
                        "reward_fn_for_oracle_policy_eval"
                    ],
                    run_type_score=algo_cfg["run_type_score"],
                    policy_network_constructor=policy_ctor,
                    value_network_constructor=agents.ValueNetwork,
                    # No regulariser / early stopping:
                    regulariser_name=None,
                    regulariser_kwargs=None,
                    stopper=None,
                    val_env=None,
                    # Data / sampling options (fall back to defaults if not present):
                    max_num_spurious_rows=algo_cfg.get(
                        "max_num_spurious_rows", 600_000
                    ),
                    max_num_filtered_rows=algo_cfg.get("max_num_filtered_rows", None),
                    oversampling_fraction=algo_cfg.get("oversampling_fraction", 0.0),
                    effective_num_outcomes=algo_cfg.get("effective_num_outcomes", None),
                    # Optimizer hyperparams:
                    learning_rate=args.learning_rate,
                    expert_weight_decay=args.expert_weight_decay,
                    shared_weight_decay=args.shared_weight_decay,
                    # GPU pinning (leave None = default behaviour)
                    gpus_to_restrict_to=None,
                )

                for seed in seeds_for_all:
                    run_label = f"{run_type}+baseline_depth{depth}"
                    run_id = seed  # nice to keep run_id == seed

                    kw = dict(base_kwargs)
                    kw["run_label"] = run_label
                    kw["run_id"] = run_id
                    kw["random_seed"] = False
                    kw["default_seed"] = seed
                    kw["env_kwargs"] = deepcopy(env_kwargs)
                    # keep depth for pretty printing (will remove before train())
                    kw["_model_depth"] = depth

                    training_kwargs_list.append(kw)

    print(
        f"Prepared {len(training_kwargs_list)} runs "
        f"for experiment '{args.experiment_name}'."
    )
    return training_kwargs_list


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    training_kwargs_list = build_training_kwargs_list(args)

    total = len(training_kwargs_list)
    for idx, kw in enumerate(training_kwargs_list, start=1):
        depth = kw.pop("_model_depth", None)
        run_label = kw["run_label"]
        run_id = kw["run_id"]
        oversight_prob = kw["env_kwargs"].get("oversight_prob", None)

        print(
            f"\n=== Run {idx}/{total} ===\n"
            f"  run_label      : {run_label}\n"
            f"  run_id         : {run_id}\n"
            f"  run_type       : {run_label.split('+')[0]}\n"
            f"  depth          : {depth}\n"
            f"  oversight_prob : {oversight_prob}\n"
            f"  num_steps      : {args.num_steps}\n"
            f"  steps/update   : {args.steps_per_learning_update}\n"
            f"  seeds          : "
            f"{'explicit' if args.seeds is not None else 'offset-based'}\n"
        )

        training.train(**kw)


if __name__ == "__main__":
    main()
