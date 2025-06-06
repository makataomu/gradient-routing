# ─────────────────────────────────────────────────────────────────────────────
# File: optuna_hpo.py
#
# A self-contained Optuna example that prunes unpromising trials early using
# Hyperband/ASHA, by wrapping your `training.train` function.
# ─────────────────────────────────────────────────────────────────────────────

import os

# Make sure Python can import your training module:
import sys

import numpy as np
import optuna
import torch as t
from optuna.pruners import HyperbandPruner

sys.path.append("projects/minigrid_repro")  # adjust if necessary
import agents
import training

# ─────────────────────────────────────────────────────────────────────────────
# 1) GLOBAL “knobs” for this study
# ─────────────────────────────────────────────────────────────────────────────
OVERSIGHT_PROB = 0.007  # NOTE: 0.7% for holdou+trai, 0.3% for val.
HOLDOUT_FRAC = 0.3 * OVERSIGHT_PROB  # tune on one, use another
HOLDOUT_FRAC = 0.003  # tune on it and use it as early stopper

SEED = 100

# We’ll treat each “resource” unit as one gradient-update (i.e. one 'update_idx'):
MAX_UPDATES = 20000  # <-- for a quick smoke-test, use 2000 (instead of 20000)
MIN_UPDATES = 400  # <-- NOTE: 5-10% of MAX_UPDATES
ETA = 3  # <-- keep top one-third at each rung

# Number of Optuna trials you want to run (for a quick check, use ~20–40)
N_TRIALS = 20  # NOTE: 10 for each HP

# We'll run single‐threaded (n_jobs=1) so you see exactly which trials get pruned.
N_JOBS = -1

DEVICE = "cuda" if t.cuda.is_available() else "cpu"
# DEVICE = get_gpu_with_most_memory()


parent_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(parent_dir, "data")
policy_visualization_dir = os.path.join(parent_dir, "policy_visualization")
experiment_name = "optuna"

reg = "earlystop"
run_type = "naive_outcomes"
# ─────────────────────────────────────────────────────────────────────────────
# 2) A small helper to “inject” the trial into training.train
# ─────────────────────────────────────────────────────────────────────────────


def objective(trial: optuna.Trial) -> float:
    """
    Each call to `training.train()` will:
      1. run for up to MAX_UPDATES gradient-steps,
      2. evaluate on the hold-out every `eval_freq` updates,
      3. call trial.report(<hold-out score>, step=update_idx),
      4. if trial.should_prune() is True, we raise an exception to stop early.

    We expose three hyperparameters for tuning:
      - patience (early-stopping patience),
      - tolerance (min improvement on hold-out),
      - learning_rate (just as an example).
    """
    real_trange = training.tqdm.trange

    # ─ 2.1) Sample hyperparameters to tune:
    patience = trial.suggest_int("patience", 50, 300)  # wait 50–300 updates
    tolerance = trial.suggest_float("tolerance", 0.01, 0.10)  # min 0.01–0.1 margin
    # holdout_frac = trial.suggest_categorical(
    #     "holdout_frac", [0.002, 0.003, 0.004, 0.005]
    # )
    # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)

    # We will call training.train _in‐process_, but monkey‐patch an early-stop callback:
    # So we override training.train’s internal EarlyStopper to call trial.report / prune.
    #
    # We do this by temporarily swapping out training.EarlyStopper.step with a wrapper,
    # then restoring it at the end.  This is a quick hack—if you were building a real
    # pipeline, you’d pass trial or a callback into training.train directly.
    original_step_fn = training.EarlyStopper.step

    def patched_step_fn(self, current_return: float, update_idx: int) -> bool:
        """
        Now matches the real signature: (self, current_return, update_idx).
        We call the original EarlyStopper.step with both args, then report to Optuna.
        """
        # 1) call original so it updates its internal counters:
        stop_flag = original_step_fn(self, current_return, update_idx)

        # 2) report to Optuna at resource = update_idx
        trial.report(current_return, step=update_idx)

        # 3) ask Optuna to prune if needed
        if trial.should_prune():
            raise optuna.TrialPruned()

        return stop_flag

    # Patch it in:
    training.EarlyStopper.step = patched_step_fn

    eval_freq = 100
    # eval_freq_actual = math.ceil(eval_freq * OVERSIGHT_PROB)
    eval_freq_actual = 50
    run_id = np.random.choice(1_000_000)
    run_id = f"{trial.number}_{run_id}"
    print(run_id)
    # ─ 2.2) Call training.train with our chosen hyperparameters:
    try:
        # We need to build a dictionary identical to what bulk_runs.py would pass, but
        # overriding the important pieces: num_learning_updates, learning_rate, early-stop params.
        reg_name = reg
        # reg_name += f"_{holdout_frac}"
        run_label = f"{run_type}+{reg_name}"

        training_kwargs = {
            "steps_per_learning_update": 32,
            "num_learning_updates": MAX_UPDATES,
            "eval_freq": eval_freq_actual,
            "policy_log_freq": 1000,
            "discount": 0.97,
            "loss_coefs": {
                "entropy_bonus": 0.1,
                "value_loss": 0.5,
                "gate_loss": 0.01,
                "gate_loss_no_oversight": 0,
            },
            "learning_rate": 5e-5,
            "expert_weight_decay": 0,
            "shared_weight_decay": 0,
            # NOTE: it's now hard coded for naive_outcomes
            "policy_network_constructor": agents.get_single_expert_policy,
            "reward_fn_to_train_on": training.true_reward_fn,
            "loss_getter_fn": agents.get_reinforce_loss,
            "env_kwargs": {
                "n_envs": 512,
                "nrows": 5,
                "ncols": 5,
                "max_step": 32,
                "oversight_prob": OVERSIGHT_PROB,
                "spurious_oversight_prob": 0,
            },
            "save_dir": os.path.join(data_dir, experiment_name),
            "policy_visualization_dir": os.path.join(
                policy_visualization_dir, experiment_name
            ),
            "run_label": run_label,
            # "device": DEVICE,
            "regulariser_name": "earlystop",
            "regulariser_kwargs": {
                # 512 x 32 × 0.007 = 114 episodes
                # 512 x 32 × 0.007 × 0.3 = 34 episodes
                "holdout_frac": HOLDOUT_FRAC,
                "patience": patience,
                "tolerance": tolerance,
                "min_steps": MIN_UPDATES,
            },
            "gpus_to_restrict_to": None,
            "run_id": run_id,
            "random_seed": False,
            "default_seed": SEED,
        }

        # Create directories (so no “file not found” errors):
        os.makedirs(training_kwargs["save_dir"], exist_ok=True)
        os.makedirs(training_kwargs["policy_visualization_dir"], exist_ok=True)

        # Store current update index in trial.user_attrs so patched_step_fn can see it:
        trial.set_user_attr("current_update", 0)

        # Now we wrap the core training loop with a small custom loop that updates
        # trial.user_attrs["current_update"] at each step.  Unfortunately training.train
        # itself doesn’t expose update‐by‐update hooks, so we have to poke into it.
        #
        # Easiest hack: monkey‐patch `tqdm.trange` so that each loop iteration advances
        # trial.user_attrs["current_update"] before calling the real body:
        real_trange = training.tqdm.trange

        def fake_trange(num_updates, *args, **kwargs):
            """
            This generator wraps the real `range(num_updates)` so that each iteration:
              - increments trial.user_attrs["current_update"],
              - yields the update_idx to the real code.
            """
            for update_idx in real_trange(num_updates, *args, **kwargs):
                trial.set_user_attr("current_update", update_idx)
                yield update_idx

        training.tqdm.trange = fake_trange

        # Finally, call the actual training:
        training.train(**training_kwargs)
        #
        # If training completes all MAX_UPDATES without pruning, we end up here.
        # Let’s grab the final hold-out performance (from the last row of `eval_results_*.csv`).
        #
        # We know `training.train` writes: save_dir/train_results_<run_id>.csv
        #                                                  eval_results_<run_id>.csv
        # We can just read back the final eval_results CSV and take its "avg_return".

        import pandas as pd

        # Read the largest (most recent) eval-results:
        f = os.path.join(training_kwargs["save_dir"], f"holdout_results_{run_id}.csv")
        print(f"Trial {trial.number} wrote: {f}")

        df_holdout = pd.read_csv(f)
        final_avg = df_holdout["avg_return"].max()

        print(run_id, " - ", final_avg)
        return float(final_avg)

    except optuna.TrialPruned:
        # Trial was pruned early; communicate that back to Optuna:
        raise

    finally:
        # ─ 2.3) UN‐patch everything so future trials aren’t affected:
        training.EarlyStopper.step = original_step_fn
        training.tqdm.trange = real_trange


# ─────────────────────────────────────────────────────────────────────────────
# 3) Launch the study
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 3.1  Choose HyperbandPruner (synchronous) or ASHAPruner (asynchronous):
    pruner = HyperbandPruner(
        min_resource=MIN_UPDATES,
        max_resource=MAX_UPDATES,
        reduction_factor=ETA,
    )
    # (If you prefer asynchronous stepping, you could do:
    #    pruner = ASHAPruner(min_resource=MIN_UPDATES, max_resource=MAX_UPDATES, reduction_factor=ETA)
    # )

    study = optuna.create_study(
        direction="maximize",  # we want to maximize avg_return
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(),  # standard TPE
        study_name="minigrid_hpo",
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)

    # 3.2  Print a summary:
    print("============================================")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"  Best value: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    print(
        f"  Pruned trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}"
    )
    print(
        f"  Complete trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))}"
    )
    print("============================================")
