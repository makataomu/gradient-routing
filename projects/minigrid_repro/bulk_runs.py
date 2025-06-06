# %%
import argparse
import glob
import inspect
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import partial

import numpy as np

try:
    import projects.minigrid_repro.agents as agents
    import projects.minigrid_repro.training as training
except ImportError:
    import agents as agents
    import training as training

from factored_representations.utils import Timer

"""
$(pdm venv activate) && python projects/minigrid_repro/bulk_runs.py
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_types", nargs="+", default=["naive_outcomes"])
    p.add_argument("--oversight_probs", nargs="+", type=float, default=[0.01, 0.02])
    p.add_argument(
        "--regularisers",
        nargs="+",
        default=["baseline", "dropout", "entropy0p05", "kl1e-3", "earlystop"],
    )
    # p.add_argument("--holdout_fracs", nargs="+", type=float, default=[0.10, 0.25, 0.5])
    p.add_argument("--seeds", nargs="+", type=int, default=[])
    p.add_argument("--num_paral_runs", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=20000)
    p.add_argument(
        "--reg_kwargs", type=json.loads, default={}
    )  # --reg_kwargs '{"holdout_frac": 0.1, "patience": 500, "tolerance": 0.05, "min_steps": 800}' - kaggle
    # "{\"holdout_frac\":0.003,\"patience\":20,\"tolerance\":0.04,\"min_steps\":100}" - cmd
    return p.parse_args()


def check_reg_args(train_func, kwargs: dict):
    sig = inspect.signature(train_func)
    valid_keys = sig.parameters.keys()

    for k, v in kwargs.items():
        if k not in valid_keys:
            raise KeyError(f"{k} is not in train")


def get_function_defaults(func):
    sig = inspect.signature(func)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }


if __name__ == "__main__":
    args = parse_args()

    # check_reg_args(training.train, args.reg_kwargs)
    # train_defaults = get_function_defaults(training.train)

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data")
    policy_visualization_dir = os.path.join(parent_dir, "policy_visualization")

    to_remove = {
        data_dir: ["*.csv", "*.pt"],
        policy_visualization_dir: ["*.png", "*.gif"],
    }

    # BULK RUN SETTINGS
    num_max_paral_runs = 8
    num_paral_runs = defaultdict(lambda: args.num_paral_runs)
    experiment_name = "oversight_levels"

    overwrite = False
    if overwrite:
        for dirname, patterns in to_remove.items():
            for pattern in patterns:
                matching_paths = glob.glob(
                    os.path.join(dirname, experiment_name, pattern)
                )
                if pattern == "*.csv":
                    assert len(matching_paths) < 20, (
                        "Are you sure you want to delete 20+ files?"
                    )
                for path in matching_paths:
                    os.remove(path)

    loss_coefs = {
        "entropy_bonus": 0.1,
        "value_loss": 0.5,
        "gate_loss": 0.01,
        "gate_loss_no_oversight": 0,
    }

    num_envs = 512
    num_learning_updates = args.num_steps

    env_kwargs = dict(
        n_envs=num_envs,
        nrows=5,
        ncols=5,
        max_step=32,
        oversight_prob=None,
        spurious_oversight_prob=0,
    )

    algorithm_settings_by_run_type = {
        "routing": dict(
            policy_network_constructor=partial(
                agents.RoutedPolicyNetwork, use_gate=True, use_gradient_routing=True
            ),
            reward_fn_to_train_on=training.moe_reward_fn,
            loss_getter_fn=agents.get_routed_reinforce_loss,
        ),
        "no_routing_control": dict(
            policy_network_constructor=partial(
                agents.RoutedPolicyNetwork, use_gate=True, use_gradient_routing=False
            ),
            reward_fn_to_train_on=training.moe_reward_fn,
            loss_getter_fn=agents.get_routed_reinforce_loss,
        ),
        "naive_outcomes": dict(
            policy_network_constructor=agents.get_single_expert_policy,
            reward_fn_to_train_on=training.naive_reward_fn,
            loss_getter_fn=agents.get_reinforce_loss,
        ),
        "filtering": dict(
            policy_network_constructor=agents.get_single_expert_policy,
            reward_fn_to_train_on=training.true_reward_fn,
            loss_getter_fn=agents.get_filtered_reinforce_loss,
        ),
        "oracle": dict(
            policy_network_constructor=agents.get_single_expert_policy,
            reward_fn_to_train_on=training.true_reward_fn,
            loss_getter_fn=agents.get_reinforce_loss,
        ),
    }

    # oversight_probs = [0.01]
    # run_types = ["naive_outcomes"]

    basic_training_kwargs = dict(
        steps_per_learning_update=32,
        num_learning_updates=num_learning_updates,
        policy_log_freq=1000,  # max(num_learning_updates // 200, 50),
        loss_coefs=loss_coefs,
        discount=0.97,
        learning_rate=5e-5,
        expert_weight_decay=0,
        shared_weight_decay=0,
        save_dir=os.path.join(data_dir, experiment_name),
        policy_visualization_dir=os.path.join(
            policy_visualization_dir, experiment_name
        ),
        gpus_to_restrict_to=None,
    )

    training_kwargs_list = []
    for reg in args.regularisers:
        for oversight_prob in args.oversight_probs:
            for run_type in args.run_types:
                alg_settings = algorithm_settings_by_run_type[run_type]
                env_kwargs_to_use = deepcopy(env_kwargs)
                env_kwargs_to_use["oversight_prob"] = oversight_prob  # type: ignore
                eval_freq = 100
                eval_freq_actual = math.ceil(eval_freq * oversight_prob)

                run_type_training_kwargs = basic_training_kwargs.copy()
                run_type_training_kwargs.update(
                    dict(
                        eval_freq=eval_freq_actual,
                        policy_network_constructor=alg_settings[
                            "policy_network_constructor"
                        ],
                        reward_fn_to_train_on=alg_settings["reward_fn_to_train_on"],
                        loss_getter_fn=alg_settings["loss_getter_fn"],
                        env_kwargs=env_kwargs_to_use,
                    )
                )

                reg_name = reg
                reg_kw = {}
                if reg == "earlystop":
                    reg_kw = args.reg_kwargs
                    if "holdout_frac" in args.reg_kwargs:
                        reg_name += f"_{args.reg_kwargs['holdout_frac']}"

                    # reg_kw = {"holdout_frac": h, "patience": 400, "tolerance": 0.06}
                    training_kwargs = run_type_training_kwargs.copy()
                    training_kwargs.update(
                        dict(
                            run_label=f"{run_type}+{reg_name}",
                            regulariser_name=reg,
                            regulariser_kwargs=args.reg_kwargs,
                        )
                    )
                    for _ in range(num_paral_runs[run_type]):
                        training_kwargs_list.append(training_kwargs)
                elif reg == "baseline":
                    training_kwargs = run_type_training_kwargs.copy()
                    training_kwargs.update(dict(run_label=f"{run_type}+{reg_name}"))

                    for _ in range(num_paral_runs[run_type]):
                        training_kwargs_list.append(training_kwargs)
                else:
                    print(f"{reg} not implemented.")
                    pass

    print(
        f"Experiment '{experiment_name}' running {len(training_kwargs_list)} total iterates across {num_max_paral_runs} processes..."
    )
    timer = Timer(num_tasks=len(training_kwargs_list))
    if num_max_paral_runs == 1:
        for training_kwargs in training_kwargs_list:
            training.train(**training_kwargs)  # type: ignore
            timer.increment()
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=num_max_paral_runs) as executor:
            for iterate_idx, training_kwargs in enumerate(training_kwargs_list):
                time.sleep(2)

                if len(args.seeds) > 0:
                    cli_seed = args.seeds[iterate_idx % len(args.seeds)]
                    unique_suffix = np.random.randint(0, 1000)
                    filename_id = cli_seed * 10000 + unique_suffix  # 70483
                    training_kwargs.update(
                        {
                            "run_id": filename_id,
                            "random_seed": False,
                            "default_seed": cli_seed,
                        }
                    )

                future = executor.submit(
                    training.train,
                    time_to_sleep_after_run=2,
                    **training_kwargs,  # type: ignore
                )
                futures.append(future)

            for future in as_completed(futures):
                future.result()
                timer.increment()

# %%
