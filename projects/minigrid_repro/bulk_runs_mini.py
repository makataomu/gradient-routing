# %%
import argparse
import glob
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import partial

import torch.multiprocessing as mp

import projects.minigrid_repro.agents as agents
import projects.minigrid_repro.training as training
from factored_representations.utils import Timer

"""
$(pdm venv activate) && python projects/minigrid_repro/bulk_runs.py
"""

if __name__ == "__main__":
    try:
        # Set the start method to 'spawn'. This is necessary for PyTorch/CUDA
        # when using multiprocessing on systems (like default Linux/Kaggle)
        # where the default is 'fork'.
        mp.set_start_method("spawn")
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        # start_method can only be set once per process.
        # This might happen if your environment or another library
        # has already set it. You might want to check if it's
        # already 'spawn' if you hit this.
        # For most cases, just passing is fine.
        print("Multiprocessing start method already set.")
        pass  # Or handle the error if needed

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data")
    policy_visualization_dir = os.path.join(parent_dir, "policy_visualization")

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="oversight_levels")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--num_learning_updates", type=int, default=20000)
    parser.add_argument(
        "--run_types",
        nargs="+",
        default=["routing", "oracle", "filtering", "naive_outcomes"],
    )
    parser.add_argument(
        "--oversight_probs",
        nargs="+",
        type=float,
        default=[0.003, 0.01, 0.025, 0.03, 0.05, 0.1],
    )
    args = parser.parse_args()

    experiment_name = args.experiment_name
    num_envs = args.num_envs
    num_learning_updates = args.num_learning_updates
    run_types = args.run_types
    oversight_probs = args.oversight_probs

    to_remove = {
        data_dir: ["*.csv", "*.pt"],
        policy_visualization_dir: ["*.png", "*.gif"],
    }

    # BULK RUN SETTINGS
    num_parallel_runs = 4
    num_iterates = defaultdict(lambda: 1)

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

    print(run_types, oversight_probs, num_envs)

    training_kwargs_list = []
    for oversight_prob in oversight_probs:
        for run_type in run_types:
            alg_settings = algorithm_settings_by_run_type[run_type]
            env_kwargs_to_use = deepcopy(env_kwargs)
            env_kwargs_to_use["oversight_prob"] = oversight_prob  # type: ignore
            num_learning_updates_actual = (
                int(num_learning_updates * oversight_prob)
                if run_type == "oracle"
                else num_learning_updates
            )
            eval_freq = 100
            eval_freq_actual = math.ceil(eval_freq * oversight_prob)

            training_kwargs = dict(
                steps_per_learning_update=32,
                num_learning_updates=num_learning_updates_actual,
                eval_freq=eval_freq_actual,
                policy_log_freq=1000,  # max(num_learning_updates // 200, 50),
                discount=0.97,
                loss_coefs=loss_coefs,
                learning_rate=5e-5,
                expert_weight_decay=0,
                shared_weight_decay=0,
                policy_network_constructor=alg_settings["policy_network_constructor"],
                reward_fn_to_train_on=alg_settings["reward_fn_to_train_on"],
                loss_getter_fn=alg_settings["loss_getter_fn"],
                env_kwargs=env_kwargs_to_use,
                save_dir=os.path.join(data_dir, experiment_name),
                policy_visualization_dir=os.path.join(
                    policy_visualization_dir, experiment_name
                ),
                run_label=run_type,
                gpus_to_restrict_to=None,
            )
            for _ in range(num_iterates[run_type]):
                training_kwargs_list.append(training_kwargs)

    print(
        f"Experiment '{experiment_name}' running {len(training_kwargs_list)} total iterates across {num_parallel_runs} processes..."
    )
    timer = Timer(num_tasks=len(training_kwargs_list))
    if num_parallel_runs == 1:
        for training_kwargs in training_kwargs_list:
            training.train(**training_kwargs)  # type: ignore
            timer.increment()
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=num_parallel_runs) as executor:
            for iterate_idx, training_kwargs in enumerate(training_kwargs_list):
                time.sleep(2)
                future = executor.submit(
                    training.train,
                    time_to_sleep_after_run=2,
                    **training_kwargs,  # type: ignore
                )
                futures.append(future)

            for future in as_completed(futures):
                future.result()
                timer.increment()
