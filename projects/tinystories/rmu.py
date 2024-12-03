# %%
import glob
import os
import sys
from functools import partial
from typing import Optional

import numpy as np
import torch as t
import torch.utils.data as data
import tqdm
import wandb

import factored_representations.string_utils as string_utils
import shared_configs.model_store as model_store
from factored_representations.utils import get_gpu_with_most_memory
from projects.tinystories.retraining_evals import (
    RetrainExperimentConfig,
    run_retrain_evals,
)
from projects.tinystories.shared_settings import cfg as experiment_cfg
from projects.tinystories.tinystories_era import _eval_to_dict

"""
$(pdm venv activate) && python projects/tinystories/rmu.py

$(pdm venv activate) && python projects/tinystories/rmu.py dry_run
"""

"""
Based on code at:
    https://github.com/centerforaisafety/wmdp/blob/main/rmu/unlearn.py

Sample hyperparameters here:
    https://github.com/centerforaisafety/wmdp/blob/main/run_rmu_zephyr.ipynb

    alpha=[1200.0, 1200.0]
    steering_coeffs=6.5,6.5
    lr=5e-05
    min_len=0
    max_len=2000
    batch_size=4
    max_num_batches=150
    layer_id=7
    layer_ids=[5, 6, 7]
    param_ids=[6]
    seed=42
    verbose=True
    steering_coeff_list=[6.5, 6.5]
"""


def get_params_at_layers(model, layer_ids: list[int], param_pattern: str):
    params = []
    param_strs = []
    for layer_id in layer_ids:
        for name, p in model.blocks[layer_id].named_parameters():
            if param_pattern in name:
                params.append(p)
                param_strs.append(f"layer {layer_id}: {name}")

    print("Training " + ", ".join(param_strs))
    return params


def masked_mse(pred, target, mask):
    mask = mask.unsqueeze(-1)
    return t.nn.functional.mse_loss(pred * mask, target * mask)


def get_rmu_loss(
    forget_stories,
    retain_stories,
    tokenize_fn,
    model,
    frozen_model,
    target_layer: int,
    steering_coef: float,
    retain_wt: float,  # this is alpha in RMU
):
    forget_toks, forget_mask = tokenize_fn(forget_stories)
    retain_toks, retain_mask = tokenize_fn(retain_stories)

    # Forget
    random_vector = t.rand(1, 1, model.cfg.d_model, device=model.cfg.device)
    control_vec = random_vector / t.norm(random_vector) * steering_coef
    forget_activations = model(forget_toks, stop_at_layer=target_layer)
    forget_loss = masked_mse(forget_activations, control_vec, forget_mask)

    # Retain
    with t.inference_mode():
        frozen_retain_activations = frozen_model(
            retain_toks, stop_at_layer=target_layer
        )
    retain_activations = model(retain_toks, stop_at_layer=target_layer)
    retain_mse = masked_mse(retain_activations, frozen_retain_activations, retain_mask)
    retain_loss = retain_mse * retain_wt

    loss = forget_loss + retain_loss
    info = {
        "rmu_loss": loss.item(),
        "rmu_loss_retain": retain_loss.item(),
        "rmu_loss_forget": forget_loss.item(),
    }
    return loss, info


def do_rmu_training_run(
    random_shuffle_seed: int,
    dry_run: bool,
    model_to_load: str,
    model_save_dir: str,
    model_save_name: str,
    num_training_steps: int,
    layers_to_train: list[int],  # what layers to train in RMU
    param_pattern: str,  # what params to train in RMU
    target_layer: int,  # RMU param
    steering_coef: float,  # RMU param
    retain_wt: float,  # RMU param: alpha
    num_validation_stories: int,
    eval_every: int,
    learning_rate: float,
    num_steps_forget_set_retraining: int,
    retrain_all_steps: bool,
    num_stories_to_retrain: list[int],
    forget_data_labeling_pct: float,
    num_times_to_retrain: int,
    additional_fields_to_save_to_df: Optional[dict] = None,
    run_type="rmu",
    device=None,
):
    model_save_path = os.path.join(model_save_dir, model_save_name)
    model_store.ensure_save_path_exists(model_save_path, overwrite=True)

    device = get_gpu_with_most_memory() if device is None else device
    data_rng = np.random.default_rng(random_shuffle_seed)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    wandb.login()
    wandb.init(
        project=f"tinystories-{experiment_cfg.wandb_project_subname}-unlearning",
        mode="disabled" if dry_run else "online",
        name=f"tinystories-rmu-{model_save_name}",
        config={
            "experiment_cfg": experiment_cfg.__dict__,
        },
        settings=wandb.Settings(code_dir=project_dir),
        dir=project_dir,
    )
    wandb.run.log_code(  # type: ignore
        project_dir,
        include_fn=lambda path, root: (path.endswith(str(__file__))),
    )

    all_stories = string_utils.load_dataset_with_split(
        "delphi-suite/stories",
        "train",
        max_stories=1000
        if dry_run
        else num_training_steps * experiment_cfg.batch_size * 10 + 2000,
    )
    data_rng.shuffle(all_stories)

    # Load data
    truncated_stories = string_utils.truncate_stories_by_chars(
        all_stories, experiment_cfg.truncate_story_chars_at
    )
    forget_stories, retain_stories = string_utils.split_and_label_stories_by_concept(
        truncated_stories, experiment_cfg.words_to_localize
    )

    # partial oversight
    num_forget_labeled = int(len(forget_stories) * forget_data_labeling_pct)
    forget_stories_labeled = forget_stories[:num_forget_labeled]
    forget_stories_unlabeled = [
        (story, 1) for story, _ in forget_stories[num_forget_labeled:]
    ]
    retain_stories += forget_stories_unlabeled
    forget_stories = forget_stories_labeled

    validation_stories = string_utils.load_dataset_with_split(
        "delphi-suite/stories", "validation", max_stories=1000 if dry_run else None
    )
    data_rng.shuffle(validation_stories)
    truncated_validation_stories = string_utils.truncate_stories_by_chars(
        validation_stories, experiment_cfg.truncate_story_chars_at
    )
    forget_validation_stories, retain_validation_stories = (
        string_utils.split_and_label_stories_by_concept(
            truncated_validation_stories,
            experiment_cfg.words_to_localize,
        )
    )
    num_val_to_use = experiment_cfg.batch_size if dry_run else num_validation_stories
    forget_validation_stories = forget_validation_stories[
        : num_val_to_use + max(num_stories_to_retrain)
    ]
    retain_validation_stories = retain_validation_stories[:num_val_to_use]

    forget_dataloader, retain_dataloader = [
        data.DataLoader(
            string_utils.ListDataset(stories),
            shuffle=True,
            batch_size=experiment_cfg.batch_size,
        )
        for stories in [forget_stories, retain_stories]
    ]

    model = model_store.load_model(
        model_to_load,
        experiment_cfg.transformer_lens_model_name,
        device,
    )

    frozen_model = model_store.load_model(
        model_to_load,
        experiment_cfg.transformer_lens_model_name,
        device,
    )

    params_to_train = get_params_at_layers(model, layers_to_train, param_pattern)
    optim = t.optim.AdamW(params_to_train, lr=learning_rate)

    num_steps = min(num_training_steps, len(forget_dataloader), len(retain_dataloader))

    tokenize_fn = partial(
        string_utils.tokenize_batch,
        tokenizer=model.tokenizer,
        prepend_bos=True,
        truncate_at=experiment_cfg.truncate_batch_tokens_at,
        padding_side="right",
        device=device,
    )

    def eval_on_validation(model):
        post_training = _eval_to_dict(
            model,
            forget_validation_stories,
            retain_validation_stories,
            truncate_at=experiment_cfg.truncate_batch_tokens_at,
        )
        wandb.log(
            {
                "validation_forget_loss": post_training["forget_loss"],
                "validation_retain_loss": post_training["retain_loss"],
            }
        )

    for step, (forget_batch, retain_batch) in enumerate(
        pbar := tqdm.tqdm(
            zip(forget_dataloader, retain_dataloader),
            total=num_steps,
        )
    ):
        if step >= num_steps:
            break

        if step == 0:
            eval_on_validation(model)

        loss, info = get_rmu_loss(
            forget_batch[0],
            retain_batch[0],
            tokenize_fn,
            model,
            frozen_model,
            target_layer,
            steering_coef,
            retain_wt,
        )
        loss.backward()

        optim.step()
        optim.zero_grad()

        wandb.log(info)
        pbar.set_postfix({"loss": info["rmu_loss"]})

        eval_every = 8 if dry_run else eval_every

        if step % eval_every == 0:
            eval_on_validation(model)

    did_eval_on_last_step = (num_steps - 1) % eval_every == 0
    if not did_eval_on_last_step:
        eval_on_validation(model)

    if model_save_path is not None:
        model_store.save_model(model, model_save_path)

    # RETRAINING
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    validation_data_save_dir = os.path.join(parent_dir, "data")

    retrain_cfg = RetrainExperimentConfig(
        words_to_localize=experiment_cfg.words_to_localize,
        num_stories_to_retrain=[1] if dry_run else num_stories_to_retrain,
        num_steps=1 if dry_run else num_steps_forget_set_retraining,
        num_times_to_retrain=num_times_to_retrain,
        eval_batch_size=experiment_cfg.batch_size,
        max_tokens=experiment_cfg.truncate_batch_tokens_at,
        pure_model_path=None,
        base_model_path=None,
        retrain_all_steps=retrain_all_steps,
        model_save_path=model_save_path,
        model_type=experiment_cfg.transformer_lens_model_name,
        prompt=experiment_cfg.unlearning_eval_prompt,
        dry_run=dry_run,
        eval_interval=1,
        test_retain_stories=num_validation_stories,
        test_forget_stories=num_validation_stories,
    )
    figs, res_df = run_retrain_evals(
        forget_validation_stories, retain_validation_stories, retrain_cfg, device
    )
    res_df.insert(0, "run_type", run_type)
    res_df.insert(1, "model_save_name", model_save_name)
    res_df.insert(2, "random_seed", random_shuffle_seed)
    if additional_fields_to_save_to_df is not None:
        for idx, (key, value) in enumerate(additional_fields_to_save_to_df.items()):
            res_df.insert(idx + 3, key, value)

    if validation_data_save_dir is not None:
        res_df.to_csv(
            os.path.join(validation_data_save_dir, f"{model_save_name}_relearn.csv")
        )

    wandb.finish()


if __name__ == "__main__":
    DRY_RUN = len(sys.argv) > 1 and sys.argv[1] == "dry_run"

    model_save_dir = "rmu"

    model_dir = model_store.MODEL_STORE_PATH / "bulk_runs_for_paper"
    experiment_tag = "11"

    file_prefix = f"{model_dir}/e{experiment_tag}_base_seed"
    model_paths = glob.glob(f"{file_prefix}*.pt")
    all_seeds = sorted([int(m[len(file_prefix) : -3]) for m in model_paths])
    print(f"Seeds: {all_seeds}")

    for seed in all_seeds:
        run_prefix = f"e{experiment_tag}" if not DRY_RUN else "dry_run"
        model_save_name = f"{run_prefix}_rmu_seed{seed}"
        do_rmu_training_run(
            random_shuffle_seed=seed + 1,
            dry_run=DRY_RUN,
            model_to_load=f"bulk_runs_for_paper/e{experiment_tag}_base_seed{seed}",
            model_save_dir=model_save_dir,
            model_save_name=model_save_name,
            num_training_steps=500,
            layers_to_train=[0, 1, 2, 3, 4, 5],
            param_pattern="W_out",
            target_layer=5,
            steering_coef=100.0,
            retain_wt=200.0,  # this is alpha in RMU
            num_validation_stories=100,
            eval_every=10,
            learning_rate=5e-4,
            num_steps_forget_set_retraining=40,
            retrain_all_steps=True,
            num_stories_to_retrain=[4, 16, 64],
            num_times_to_retrain=5,
            forget_data_labeling_pct=1.0,
        )
        if DRY_RUN:
            break
