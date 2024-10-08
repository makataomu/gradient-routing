# %%
import os
import random
import sys

import factored_representations.utils as utils
import projects.tinystories.shared_settings as shared_settings
import projects.tinystories.tinystories_era as tinystories_era
from factored_representations.files import ensure_shared_dir_exists

"""
$(pdm venv activate) && python projects/tinystories/bulk_runs_for_paper.py

$(pdm venv activate) && python projects/tinystories/bulk_runs_for_paper.py dry_run
"""

era_cfg = shared_settings.ERAConfig(
    layers_to_mask=[0, 1, 2, 3, 4],
    to_expand={"d_mlp": 64},
    masking_scheme="full_seq",
    masking_type="ddbp",
    expanded_vs_original_dim_learning_rates=dict(
        expanded_dim_lr_target=1.0,
        original_dim_lr_target=-0.75,
        expanded_dim_lr_off_target=1.0,
        original_dim_lr_off_target=1.0,
    ),
    include_conditional_bias_term=False,
)
era_steps = 5_000
coherence_finetuning = 40
forget_set_retraining = 40
erac_l1_coeff = 1e-4

erac_model_cfg = shared_settings.RunTypeConfig(
    label="ERAC",
    pretrained_model_to_load=None,
    anneal_gradient_mask_weights=False,
    mask_weight_increase_steps=0,
    expand_model=True,
    use_gradient_routing=True,
    forget_data_labeling_percentage=1.0,
    drop_labeled_forget_data=False,
    drop_unlabeled_forget_data=False,
    sort_forget_data_by_label=False,
    num_steps_era_training=era_steps,
    num_steps_coherence_finetuning=coherence_finetuning,
    num_steps_forget_set_retraining=forget_set_retraining,
    l1_coeff=erac_l1_coeff,
)

base_model_cfg = shared_settings.RunTypeConfig(
    label="base",
    pretrained_model_to_load=None,
    anneal_gradient_mask_weights=False,
    mask_weight_increase_steps=0,
    expand_model=False,
    use_gradient_routing=False,
    forget_data_labeling_percentage=1.0,
    drop_labeled_forget_data=False,
    drop_unlabeled_forget_data=False,
    sort_forget_data_by_label=False,
    num_steps_era_training=era_steps,
    num_steps_coherence_finetuning=0,
    num_steps_forget_set_retraining=0,
    l1_coeff=0,
)

pure_model_cfg = shared_settings.RunTypeConfig(
    label="pure",
    pretrained_model_to_load=None,
    anneal_gradient_mask_weights=False,
    mask_weight_increase_steps=0,
    expand_model=False,
    use_gradient_routing=False,
    forget_data_labeling_percentage=1,
    drop_labeled_forget_data=True,
    drop_unlabeled_forget_data=False,
    sort_forget_data_by_label=False,
    num_steps_era_training=era_steps,
    num_steps_coherence_finetuning=0,
    num_steps_forget_set_retraining=forget_set_retraining,
    l1_coeff=0,
)

expanded_base_model_cfg = shared_settings.RunTypeConfig(
    label="expanded_base",
    pretrained_model_to_load=None,
    anneal_gradient_mask_weights=False,
    mask_weight_increase_steps=0,
    expand_model=True,
    use_gradient_routing=False,
    forget_data_labeling_percentage=1.0,
    drop_labeled_forget_data=False,
    drop_unlabeled_forget_data=False,
    sort_forget_data_by_label=False,
    num_steps_era_training=era_steps,
    num_steps_coherence_finetuning=coherence_finetuning,
    num_steps_forget_set_retraining=0,
    l1_coeff=erac_l1_coeff,
)

if __name__ == "__main__":
    DRY_RUN = len(sys.argv) > 1 and sys.argv[1] == "dry_run"
    OVERWRITE_MODELS = True

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, "data")

    device = utils.get_gpu_with_most_memory()
    experiment_id = "11"

    model_save_dir = "bulk_runs_for_paper"
    ensure_shared_dir_exists(f"models/{model_save_dir}")

    num_runs_per_type = 12
    run_configs = [
        erac_model_cfg,
        # base_model_cfg,
        # pure_model_cfg,
        # expanded_base_model_cfg,
    ]

    experiment_tag = f"e{experiment_id}" if not DRY_RUN else "dry_run"
    print("Starting bulk runs...")
    timer = utils.Timer(num_runs_per_type * len(run_configs) if not DRY_RUN else 1)
    for run_idx in range(num_runs_per_type):
        random_shuffle_seed = random.randint(0, 1_000_000)
        for run_config in run_configs:
            run_name = f"{experiment_tag}_{run_config.label}_seed{random_shuffle_seed}"
            print(f"Starting run {run_name}...")
            tinystories_era.do_era_training_run(
                experiment_cfg=shared_settings.cfg,
                run_type_cfg=run_config,
                era_cfg=era_cfg,
                random_shuffle_seed=random_shuffle_seed,
                num_validation_stories=1000,
                num_stories_to_retrain=[1, 4, 16, 64],
                device=device,
                model_save_dir=model_save_dir,
                model_save_name=run_name,
                overwrite_model_saves=OVERWRITE_MODELS,
                validation_data_save_dir=data_dir,
                dry_run=DRY_RUN,
            )
            timer.increment()
        if DRY_RUN:  # Remove this to test multiple runs
            break
