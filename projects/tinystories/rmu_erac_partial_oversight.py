import argparse
import glob
import sys

import numpy as np

import shared_configs.model_store as model_store
from factored_representations.utils import get_gpu_with_most_memory
from projects.tinystories.rmu import do_rmu_training_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_gpus", type=int)
    parser.add_argument("gpu_on", type=int)
    parser.add_argument("--dry_run", default=False)
    args = parser.parse_args()
    gpu_on = args.gpu_on
    num_gpus = args.num_gpus
    device = get_gpu_with_most_memory()

    DRY_RUN = args.dry_run

    model_save_dir = "rmu_erac_partial_oversight"

    model_dir = model_store.MODEL_STORE_PATH / "bulk_runs_for_paper"
    experiment_tag = "11"

    # gotten from ls bulk_runs_for_paper/e11-o_ERAC*oversight_00.pt and then picking out the seeds
    all_seeds = [
        226624,
        272144,
        308618,
        404079,
        431623,
        466992,
        470967,
        502501,
        553028,
        581963,
        59734,
        600782,
        731777,
        743312,
        81169,
        852336,
    ]
    print(f"Seeds: {all_seeds}")
    all_seeds = np.array_split(all_seeds, num_gpus)[gpu_on]

    for oversight_pct in [1]:
        for seed in all_seeds:
            run_prefix = (
                f"e{experiment_tag}-o{oversight_pct}" if not DRY_RUN else "dry_run"
            )
            oversight_int = f"{int(oversight_pct * 100):02}"
            model_save_name = (
                f"{run_prefix}_rmu_erac_seed{seed}_oversight{oversight_int}"
            )
            do_rmu_training_run(
                random_shuffle_seed=seed + 1,
                dry_run=DRY_RUN,
                model_to_load=f"bulk_runs_for_paper/e{experiment_tag}-o_ERAC_seed{seed}_oversight{oversight_int}",
                model_save_dir=model_save_dir,
                model_save_name=model_save_name,
                num_training_steps=500,
                layers_to_train=[0, 1, 2, 3, 4, 5],
                param_pattern="W_out",
                target_layer=5,
                steering_coef=100.0,
                retain_wt=200.0,  # this is alpha in RMU
                num_validation_stories=1000,
                eval_every=1000000,  # effectively don't eval for rmu
                learning_rate=5e-4,
                num_steps_forget_set_retraining=40,
                retrain_all_steps=False,
                num_stories_to_retrain=[64],
                forget_data_labeling_pct=oversight_pct,
                num_times_to_retrain=5,
                additional_fields_to_save_to_df={
                    "oversight_pct": oversight_pct,
                    "seed": seed,
                },
                run_type="rmu_erac",
                device=device,
            )
            if DRY_RUN:
                break
        if DRY_RUN:
            break
