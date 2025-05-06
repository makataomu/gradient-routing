# %%
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial

import projects.minigrid_repro.agents as agents
import projects.minigrid_repro.training as training

"""
$(pdm venv activate) && python projects/minigrid_repro/bulk_runs_ablations.py
"""

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data")
    policy_visualization_dir = os.path.join(parent_dir, "policy_visualization")

    to_remove = {data_dir: ["*.csv"], policy_visualization_dir: ["*.png", "*.gif"]}

    # BULK RUN SETTINGS
    num_parallel_runs = 2
    num_iterates = 4
    experiment_name = "ablations"

    overwrite = True
    if overwrite:
        for dirname, patterns in to_remove.items():
            for pattern in patterns:
                for path in glob.glob(os.path.join(dirname, experiment_name, pattern)):
                    os.remove(path)

    loss_coefs = {
        "entropy_bonus": 0.1,
        "value_loss": 0.5,
        "gate_loss": 0.01,
        "gate_loss_no_oversight": 0,
    }

    num_envs = 512
    num_learning_updates = 20000

    env_kwargs = dict(
        n_envs=num_envs,
        nrows=5,
        ncols=5,
        max_step=32,
        oversight_prob=0.1,
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
        "no_gate": dict(
            policy_network_constructor=partial(
                agents.RoutedPolicyNetwork, use_gate=True, use_gradient_routing=False
            ),
            reward_fn_to_train_on=training.moe_reward_fn,
            loss_getter_fn=agents.get_routed_reinforce_loss,
        ),
    }

    run_types = ["routing", "no_routing_control", "no_gate"]

    training_kwargs_list = []
    for run_type in run_types:
        alg_settings = algorithm_settings_by_run_type[run_type]
        env_kwargs_to_use = deepcopy(env_kwargs)
        num_learning_updates_actual = (
            int(num_learning_updates * env_kwargs["oversight_prob"])
            if run_type == "oracle"
            else num_learning_updates
        )

        training_kwargs = dict(
            steps_per_learning_update=32,
            num_learning_updates=num_learning_updates_actual,
            eval_freq=50,
            policy_log_freq=max(num_learning_updates // 200, 50),
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
        training_kwargs_list.append(training_kwargs)

    training_kwargs_list = num_iterates * training_kwargs_list

    print(
        f"Experiment '{experiment_name}' running {len(training_kwargs_list)} total iterates across {num_parallel_runs} processes..."
    )
    if num_parallel_runs == 1:
        for training_kwargs in training_kwargs_list:
            training.train(**training_kwargs)  # type: ignore
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=num_parallel_runs) as executor:
            for training_kwargs in training_kwargs_list:
                time.sleep(2)
                future = executor.submit(
                    training.train,
                    **training_kwargs,  # type: ignore
                )
                futures.append(future)
        for future in futures:
            future.result()
