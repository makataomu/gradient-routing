# %%
import json
import math
import os
import sys
import tempfile
from copy import deepcopy
from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np
import torch as t
import torch.utils.data as data
import tqdm
import wandb
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_pretrained_model_config

import factored_representations.model_expansion as model_expansion
import factored_representations.string_utils as string_utils
import factored_representations.training as training
import factored_representations.utils as utils
import projects.tinystories.shared_settings as shared_settings
import shared_configs.model_store as model_store
from factored_representations import masklib
from projects.tinystories.retraining_evals import (
    RetrainExperimentConfig,
    eval_model,
    run_retrain_evals,
)
from projects.tinystories.unlearning_eval import unlearning_eval

MASKING_SCHEMES = ["concept", "freq", "full_seq", "no_routing"]
MASKING_TYPES = ["ddbp", "slgr", "demix"]
SCHEMES_WITH_PARTIAL_WEIGHTS = ["freq", "full_seq"]


def eval_on_validation(
    model,
    validation_forget_data: list[tuple],
    validation_retain_data: list[tuple],
    truncate_at: int,
    ids_to_set=None,  # super hacky thing because DEMIX modifies forward pass
) -> Tuple[dict, dict]:
    if ids_to_set is not None:
        ids_to_set[0] = t.tensor(0)
    validation_forget_loss = eval_model(
        model,
        dataset=validation_forget_data,
        truncate_at=truncate_at,
    )
    if ids_to_set is not None:
        ids_to_set[0] = t.tensor(1)
    validation_retain_loss = eval_model(
        model,
        dataset=validation_retain_data,
        truncate_at=truncate_at,
    )
    if ids_to_set is not None:
        ids_to_set[0] = None
    return validation_forget_loss, validation_retain_loss


def _eval_to_dict(
    model,
    validation_forget_data: list[tuple],
    validation_retain_data: list[tuple],
    truncate_at: int,
    ids_to_set=None,
):
    validation_forget_loss, validation_retain_loss = eval_on_validation(
        model, validation_forget_data, validation_retain_data, truncate_at, ids_to_set
    )
    return {
        "forget_loss": validation_forget_loss["loss"],
        "retain_loss": validation_retain_loss["loss"],
    }


def eval_and_log(
    model,
    mask_applier,
    words_to_localize,
    eval_str,
    n_samples,
    step,
    include_table=False,
    include_clbin=False,
):
    df, all_formatted_stories = unlearning_eval(
        model, mask_applier, words_to_localize, eval_str, n_samples
    )
    to_log = (
        df.groupby(["label"])
        .agg({"num_extra_localized_words": "sum"})
        .to_dict()["num_extra_localized_words"]
    )
    if step is not None:
        wandb.log(to_log)
        if include_table:
            wandb.log({f"Localized word counts ({step=})": wandb.Table(dataframe=df)})
    if include_clbin:
        return utils.upload_to_clbin(all_formatted_stories)
    else:
        return ""


def full_seq_mask_rule(labels, seq_length, device):
    """
    0 means should be in forget set
    1 means should be in retain set
    """
    return labels.unsqueeze(1).repeat(1, seq_length).to(device)


LABEL_SETTINGS = ["ignore", "retain_always_unmasked"]


def convert_to_labeled_mask_rule(mask_rule: Callable, label_setting: str) -> Callable:
    """Take a mask rule defined on tokens and convert it to a mask rule defined on tokens and labels."""
    assert label_setting in LABEL_SETTINGS

    if label_setting == "ignore":

        def ignore_label_mask_rule(input_ids_and_labels):
            input_ids, labels = input_ids_and_labels
            return mask_rule(input_ids)

        return ignore_label_mask_rule

    elif label_setting == "retain_always_unmasked":

        def unmasked_retain_mask_rule(input_ids_and_labels):
            input_ids, labels = input_ids_and_labels
            original_mask = mask_rule(input_ids)
            is_retain = full_seq_mask_rule(
                labels, input_ids.shape[1] - 1, original_mask.device
            )
            return t.maximum(is_retain, original_mask)

        return unmasked_retain_mask_rule
    else:
        raise ValueError(f"Unknown label setting: {label_setting}")


def do_era_training_run(
    experiment_cfg: shared_settings.SharedExperimentConfig,
    run_type_cfg: shared_settings.RunTypeConfig,
    era_cfg: shared_settings.ERAConfig,
    random_shuffle_seed: int,
    num_validation_stories: int,
    num_stories_to_retrain: list[int],
    device: t.device,
    model_save_dir: str,
    model_save_name: str,
    overwrite_model_saves: bool,
    dry_run: bool,
    num_iterations_per_eval: Optional[int] = None,
    validation_data_save_dir: Optional[str] = None,
    additional_fields_to_save_to_df: Optional[dict] = None,
):
    assert era_cfg.masking_scheme in MASKING_SCHEMES
    assert era_cfg.masking_type in MASKING_TYPES
    assert (
        run_type_cfg.drop_labeled_forget_data + run_type_cfg.drop_unlabeled_forget_data
        <= 1
    ), "Can't drop both labeled and unlabeled forget data"

    if num_iterations_per_eval is None:
        num_iterations_per_eval = 750

    # Do this so not setting random state globally
    data_rng = np.random.default_rng(random_shuffle_seed)

    model_save_path = os.path.join(model_save_dir, model_save_name)
    model_store.ensure_save_path_exists(model_save_path, overwrite_model_saves)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    wandb.login()
    wandb.init(
        project=f"tinystories-{experiment_cfg.wandb_project_subname}-unlearning",
        mode="disabled" if dry_run else "online",
        name=f"tinystories-{run_type_cfg.label}-{model_save_name}",
        config={
            "experiment_cfg": experiment_cfg.__dict__,
            "run_type_cfg": run_type_cfg.__dict__,
            "era_cfg": era_cfg.__dict__,
        },
        settings=wandb.Settings(code_dir=project_dir),
        dir=project_dir,
    )
    wandb.run.log_code(  # type: ignore
        project_dir,
        include_fn=lambda path, root: (path.endswith(str(__file__))),
    )

    shared_settings.print_info_for_model_tracker(
        model_save_name,
        model_save_dir,
        "dry run" if dry_run else wandb.run.url,  # type: ignore
        hash_str=experiment_cfg.hash() + f"-s{random_shuffle_seed}",
    )
    print(f"run type: {run_type_cfg.label} / {era_cfg}")
    print()

    num_training_steps = min(
        experiment_cfg.total_num_stories_to_load // experiment_cfg.batch_size,
        run_type_cfg.num_steps_era_training,
    )

    def get_lr(it):
        # Copied from NanoGPT. Hardcoding some values...
        min_lr = experiment_cfg.learning_rate / 10
        warmup_iters = 100
        lr_decay_iters = num_training_steps

        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return experiment_cfg.learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (experiment_cfg.learning_rate - min_lr)

    if run_type_cfg.pretrained_model_to_load is None:
        config = get_pretrained_model_config(
            experiment_cfg.transformer_lens_model_name, device=device
        )
        old_model = HookedTransformer(config)
    else:
        old_model = model_store.load_model(
            run_type_cfg.pretrained_model_to_load,
            experiment_cfg.transformer_lens_model_name,
            device,
        )
    original_model_config = old_model.cfg

    all_stories = string_utils.load_dataset_with_split(
        "delphi-suite/stories",
        "train",
        max_stories=1000 if dry_run else experiment_cfg.total_num_stories_to_load,
    )
    data_rng.shuffle(all_stories)

    # Load data
    truncated_stories = string_utils.truncate_stories_by_chars(
        all_stories, experiment_cfg.truncate_story_chars_at
    )
    forget_stories, retain_stories = string_utils.split_and_label_stories_by_concept(
        truncated_stories, experiment_cfg.words_to_localize
    )

    # Determine training data / partial oversight
    num_forget_labeled = int(
        len(forget_stories) * run_type_cfg.forget_data_labeling_percentage
    )
    forget_stories_labeled = forget_stories[:num_forget_labeled]
    forget_stories_unlabeled = [
        (story, 1) for story, _ in forget_stories[num_forget_labeled:]
    ]

    if run_type_cfg.drop_labeled_forget_data:
        training_stories = retain_stories + forget_stories_unlabeled
        data_rng.shuffle(training_stories)
    elif run_type_cfg.drop_unlabeled_forget_data:
        training_stories = retain_stories + forget_stories_labeled
        data_rng.shuffle(training_stories)
    elif run_type_cfg.sort_forget_data_by_label:
        # This makes it so you train on labeled forget data first
        is_forget_indicators = [True] * len(forget_stories) + [False] * len(
            retain_stories
        )
        data_rng.shuffle(is_forget_indicators)
        retain_stories_copy = (
            retain_stories.copy()
        )  # Hacky-- want to refer to retain later for coherence training.
        training_stories = []
        for is_forget in is_forget_indicators:
            if is_forget:
                if len(forget_stories_labeled) > 0:
                    training_stories.append(forget_stories_labeled.pop())
                else:
                    training_stories.append(forget_stories_unlabeled.pop())
            else:
                training_stories.append(retain_stories_copy.pop())
        assert len(forget_stories_labeled) == 0
        assert len(forget_stories_unlabeled) == 0
        assert len(retain_stories_copy) == 0
    else:
        training_stories = (
            retain_stories + forget_stories_labeled + forget_stories_unlabeled
        )
        data_rng.shuffle(training_stories)

    validation_stories = string_utils.load_dataset_with_split(
        "delphi-suite/stories", "validation", max_stories=None
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
    num_val_to_use = num_validation_stories
    forget_validation_stories = forget_validation_stories[
        : num_val_to_use + max(num_stories_to_retrain)
    ]
    retain_validation_stories = retain_validation_stories[:num_val_to_use]

    num_token_freq_calculate_stories = 25000
    token_freq_kwargs = dict(
        retain_stories=[story for story, _ in retain_stories],
        forget_stories=[story for story, _ in forget_stories],
        num_stories=num_token_freq_calculate_stories,
        truncate_at=None,
        num_synthetic_tokens_retain=20,
        num_synthetic_tokens_forget=1,
        scale=1.0,
        bias=-4.0,
        tokenizer=old_model.tokenizer,
        device=device,
    )
    if era_cfg.masking_type == "ddbp":
        if era_cfg.masking_scheme == "concept":
            token_mask_rule = masklib.get_concept_masking_rule(
                [(experiment_cfg.words_to_localize, 0)],
                old_model.tokenizer,
            )
        elif era_cfg.masking_scheme == "freq":
            token_mask_rule, _ = masklib.get_token_freq_masking_rule(
                **token_freq_kwargs  # type: ignore
            )
        elif era_cfg.masking_scheme == "full_seq":
            token_mask_rule, _ = masklib.get_token_freq_masking_rule(
                **token_freq_kwargs  # type: ignore
            )
        elif era_cfg.masking_scheme == "no_routing":
            token_mask_rule = lambda toks: t.ones_like(toks[:, :-1], dtype=t.long).to(
                device
            )  # just put everything in the retain set; good for testing a control
        else:
            raise ValueError(f"Unknown masking scheme: {era_cfg.masking_scheme}")

        # Change mask rule so it operates on (input_ids, labels) instead of just input_ids
        mask_rule = convert_to_labeled_mask_rule(
            token_mask_rule, "retain_always_unmasked"
        )

    elif era_cfg.masking_type == "slgr":
        mask_rule, info = masklib.get_token_freq_masking_rule(**token_freq_kwargs)  # type: ignore
        mask_weights_for_tokens: t.Tensor = (
            t.tensor(info["mask_weight"]).float().to(device)
        )
        assert mask_weights_for_tokens.shape == (old_model.cfg.d_vocab,)
    elif era_cfg.masking_type == "demix":
        print(
            "using demix, ignoring layers_to_mask, masking_scheme, to_expand, l1, etc."
        )
        mask_rule = partial(full_seq_mask_rule, device=device)
        mask_applier = None  # satisfy type checker
    else:
        raise ValueError(f"Unknown masking type: {era_cfg.masking_type}")

    # ERA setup
    def no_mask_rule(input_ids_and_labels):
        input_ids, labels = input_ids_and_labels
        return t.ones_like(input_ids[:, :-1], dtype=t.long).to(device)

    model, specs = model_expansion.expand_and_get_mask_specs(
        old_model,
        era_cfg.to_expand if run_type_cfg.expand_model else {},
        layers_to_mask=era_cfg.layers_to_mask,
        masking_rule=mask_rule if run_type_cfg.use_gradient_routing else no_mask_rule,
        suppress_warnings=False,
        **era_cfg.expanded_vs_original_dim_learning_rates,
        weight_initialization_coeff=1.0,
    )
    if run_type_cfg.pretrained_model_to_load is None:
        model.init_weights()

    use_bias = (
        run_type_cfg.use_gradient_routing and era_cfg.include_conditional_bias_term
    )
    forget_data_bias_term = t.nn.Parameter(
        t.zeros((model.cfg.d_model,), device=device),
        requires_grad=use_bias,
    )

    dataloader = data.DataLoader(
        string_utils.ListDataset(training_stories),
        shuffle=False,
        batch_size=experiment_cfg.batch_size,
    )

    print("Setting up masks")
    mask_to_use_in_ff = None  # type: ignore
    if era_cfg.masking_type == "ddbp":
        mask_applier = masklib.MaskApplier(
            model,
            specs,
            use_partial_boolean_masks=era_cfg.masking_scheme
            in SCHEMES_WITH_PARTIAL_WEIGHTS,
        )
    elif era_cfg.masking_type == "slgr":
        mask_applier = masklib.MaskApplier(
            model,
            specs,
            use_partial_boolean_masks=era_cfg.masking_scheme
            in SCHEMES_WITH_PARTIAL_WEIGHTS,
        )  # this one is for evals
        slgr_good_specs = model_expansion.get_expanded_mask_specs(
            old_model,
            attributes_to_expand=era_cfg.to_expand,
            layers_to_mask=era_cfg.layers_to_mask,
            masking_rule=lambda x: t.ones_like(x, dtype=t.long).to(device),
            new_model=model,
            **era_cfg.expanded_vs_original_dim_learning_rates,
        )
        slgr_bad_specs = model_expansion.get_expanded_mask_specs(
            old_model,
            attributes_to_expand=era_cfg.to_expand,
            layers_to_mask=era_cfg.layers_to_mask,
            masking_rule=lambda x: t.zeros_like(x, dtype=t.long).to(device),
            new_model=model,
            **era_cfg.expanded_vs_original_dim_learning_rates,
        )
        slgr_good_mask_applier = masklib.MaskApplier(
            model, slgr_good_specs, use_partial_boolean_masks=False
        )
        slgr_bad_mask_applier = masklib.MaskApplier(
            model, slgr_bad_specs, use_partial_boolean_masks=False
        )
        for param in old_model.parameters():
            param.requires_grad = False
    elif era_cfg.masking_type == "demix":
        mask_to_use_in_ff: Optional[list[t.Tensor]] = [t.tensor(0)]

        class DEMIXMLP(t.nn.Module):
            def __init__(self, old_mlp):
                super().__init__()
                self.good_mlp = deepcopy(old_mlp)
                self.bad_mlp = deepcopy(old_mlp)

            def forward(self, x):
                mask = mask_to_use_in_ff[0]  # type: ignore
                if len(mask.size()) == 0 and mask.item() == 1:
                    return self.good_mlp(x)
                elif len(mask.size()) == 0 and mask.item() == 0:
                    return self.bad_mlp(x)
                else:
                    # TODO we can parallelize this to get some easy speedups
                    # but for simplicity now, we don't
                    good_output = self.good_mlp(x) * mask[:, :, None]
                    bad_output = self.bad_mlp(x) * (1 - mask)[:, :, None]
                    # mask of shape [batch, seq]
                    # *_output of shape [batch, seq, d_model]
                    mixed = good_output + bad_output
                    return mixed

        # Initialize the custom MLPs
        for layer in range(model.cfg.n_layers):
            # make the MLP our custom demix MLP
            model.blocks[layer].mlp = DEMIXMLP(model.blocks[layer].mlp)
        model.init_weights()  # should reinitialize all the new params we added
    else:
        raise ValueError(f"Unknown masking type: {era_cfg.masking_type}")

    optim = t.optim.AdamW(
        list(model.parameters()) + [forget_data_bias_term],
        lr=experiment_cfg.learning_rate,
        **experiment_cfg.optimizer_kwargs,
    )

    mask_weight = 0 if run_type_cfg.anneal_gradient_mask_weights else 1

    total_steps = min(run_type_cfg.num_steps_era_training, len(dataloader))
    print("PHASE ONE: EXPAND AND ROUTE")
    for step, batch in (pbar := tqdm.tqdm(enumerate(dataloader), total=total_steps)):
        lr = (
            get_lr(step)
            if experiment_cfg.decay_learning_rate
            else experiment_cfg.learning_rate
        )
        for param_group in optim.param_groups:
            param_group["lr"] = lr

        if step > total_steps - 1:
            break
        stories, labels = batch
        input_ids, attention_mask = string_utils.tokenize_batch(
            stories,
            model.tokenizer,
            prepend_bos=True,
            truncate_at=experiment_cfg.truncate_batch_tokens_at,
            padding_side="right",
            device=device,
        )
        l1_norm = [t.tensor(0.0, device=device)]

        def l1_norm_hook(value, hook):
            max_tokens_per_batch = (
                experiment_cfg.batch_size * experiment_cfg.truncate_batch_tokens_at
            )
            l1_norm[0] += value.norm(p=1) / max_tokens_per_batch  # type: ignore

        for layer in era_cfg.layers_to_mask:
            for model_part in era_cfg.to_expand.keys():
                hook_name = model_expansion.get_hook_name_from_model_part(model_part)
                model.add_hook(
                    f"blocks.{layer}.{hook_name}",
                    l1_norm_hook,  # type: ignore
                )

        if era_cfg.masking_type == "ddbp":
            with mask_applier((input_ids, labels), mask_weight=mask_weight):  # type: ignore
                mask_weights = mask_rule((input_ids, labels))  # type: ignore

                if use_bias:

                    def add_bias_before_unembed_hook(tensor, hook):
                        bias = forget_data_bias_term[None, None, :].expand(
                            tensor.shape[0], tensor.shape[1], -1
                        )
                        bias = bias * (1 - mask_weights)[:, :, None]
                        return tensor + bias

                    model.add_hook(
                        "ln_final.hook_normalized", add_bias_before_unembed_hook
                    )
                loss = training.compute_preds_and_get_ce_loss(
                    model, input_ids, attention_mask, None
                )

            loss = loss / experiment_cfg.grad_accum_steps
            if run_type_cfg.l1_coeff > 0.0:
                loss += run_type_cfg.l1_coeff * l1_norm[0]
            loss.backward()

        elif era_cfg.masking_type == "slgr":
            assert run_type_cfg.l1_coeff == 0.0  # we didn't implement this in slgr yet
            softmax_temp = 1.0  # TODO refactor this into settings if we keep using it
            with t.inference_mode():
                teacher_model_logits = old_model(
                    input_ids, attention_mask=attention_mask
                )
                teacher_model_log_softmax = t.nn.functional.log_softmax(
                    teacher_model_logits / softmax_temp, dim=-1
                )
            with slgr_good_mask_applier((input_ids, labels)):  # type: ignore
                good_loss = training.compute_slgr_kl_div_loss(
                    teacher_model_log_softmax,
                    model,
                    input_ids,
                    attention_mask,
                    t.tensor(1.0) - mask_weights_for_tokens,  # type: ignore
                    temp=softmax_temp,
                )
                good_loss.backward()
            with slgr_bad_mask_applier((input_ids, labels)):  # type: ignore
                bad_loss = training.compute_slgr_kl_div_loss(
                    teacher_model_log_softmax,
                    model,
                    input_ids,
                    attention_mask,
                    mask_weights_for_tokens,  # type: ignore
                    temp=softmax_temp,
                )
                bad_loss.backward()
            with t.inference_mode():
                loss = good_loss + bad_loss  # keep for logging
        elif era_cfg.masking_type == "demix":
            mask_to_use_in_ff[0] = mask_rule(labels, input_ids.shape[1] - 1)  # type: ignore
            loss = training.compute_preds_and_get_ce_loss(
                model, input_ids, attention_mask, None
            )
            loss.backward()
            mask_to_use_in_ff[0] = None  # type: ignore
        else:
            raise ValueError(f"Unknown masking type: {era_cfg.masking_type}")
        if step % experiment_cfg.grad_accum_steps == 0:
            optim.step()
            optim.zero_grad()

        if run_type_cfg.anneal_gradient_mask_weights:
            mask_weight = min(
                1, mask_weight + 1 / run_type_cfg.mask_weight_increase_steps
            )
        else:
            mask_weight = 1

        effective_loss = loss.item() * experiment_cfg.grad_accum_steps
        wandb.log(
            {
                "loss": effective_loss - run_type_cfg.l1_coeff * l1_norm[0].item(),
                "l1_norm": l1_norm[0].item(),
            }
        )
        pbar.set_postfix(
            {"loss": effective_loss - run_type_cfg.l1_coeff * l1_norm[0].item()}
        )

        eval_every = 8 if dry_run else num_iterations_per_eval

        if step % eval_every == 0:
            # print the unembed of the bias
            if use_bias:
                print(forget_data_bias_term)
                bias_unembedded = (
                    model.W_U.T @ forget_data_bias_term[None, :].T
                ).squeeze()
                top_10_tokens = t.argsort(bias_unembedded, descending=True)[:10]
                print(top_10_tokens)
                print(
                    "top 10 bias unembed tokens:",
                    [model.tokenizer.decode([tok.item()]) for tok in top_10_tokens],  # type: ignore
                )
            validation_forget_loss, validation_retain_loss = eval_on_validation(
                model,
                forget_validation_stories[: 5 * experiment_cfg.batch_size],
                retain_validation_stories[: 5 * experiment_cfg.batch_size],
                experiment_cfg.truncate_batch_tokens_at,
                ids_to_set=mask_to_use_in_ff,
            )
            wandb.log(
                {
                    "validation_forget_loss": validation_forget_loss["loss"],
                    "validation_retain_loss": validation_retain_loss["loss"],
                }
            )

    pre_ablation = _eval_to_dict(
        model,
        forget_validation_stories,
        retain_validation_stories,
        truncate_at=experiment_cfg.truncate_batch_tokens_at,
        ids_to_set=mask_to_use_in_ff,
    )
    wandb.log(
        {
            "validation_forget_loss": pre_ablation["forget_loss"],
            "validation_retain_loss": pre_ablation["retain_loss"],
        }
    )

    if model_save_path is not None:
        model_store.save_model(model, model_save_path + "_pre_ablation")

    print("PHASE TWO: COHERENCE TRAINING")
    if era_cfg.masking_type != "demix":
        contracted_model = model_expansion.contract_model(model, original_model_config)
    else:
        for layer in range(model.cfg.n_layers):
            model.blocks[layer].mlp = model.blocks[layer].mlp.good_mlp
        contracted_model = model
    del model  # don't want to get confused
    del mask_applier  # don't need mask_applier after contraction # type: ignore
    del optim
    del dataloader
    del specs

    post_ablation = _eval_to_dict(
        contracted_model,
        forget_validation_stories,
        retain_validation_stories,
        truncate_at=experiment_cfg.truncate_batch_tokens_at,
    )
    wandb.log(
        {
            "validation_forget_loss": post_ablation["forget_loss"],
            "validation_retain_loss": post_ablation["retain_loss"],
        }
    )
    pre_post_ablation_evals = {
        "pre_ablation": pre_ablation,
        "post_ablation": post_ablation,
    }
    if validation_data_save_dir is not None:
        pre_post_ablation_path = os.path.join(
            validation_data_save_dir, f"{model_save_name}_pre_post_ablation.json"
        )
        with open(pre_post_ablation_path, "w") as file:
            json.dump(pre_post_ablation_evals, file)

    optim = t.optim.AdamW(
        contracted_model.parameters(),
        lr=5e-5,
        **experiment_cfg.optimizer_kwargs,
    )

    contracted_model.tokenizer.padding_side = "left"  # type: ignore
    data_rng.shuffle(retain_stories)
    n_retain_train = 64
    n_retain_test = 1000 if not dry_run else experiment_cfg.batch_size
    retain_retrain_stories = retain_stories[:n_retain_train]
    retain_in_sample_validation = retain_stories[
        n_retain_train : n_retain_train + n_retain_test
    ]
    input_ids, attention_mask = string_utils.tokenize_batch(
        [story for story, _ in retain_retrain_stories],
        tokenizer=contracted_model.tokenizer,
        prepend_bos=True,
        truncate_at=experiment_cfg.truncate_batch_tokens_at,
        padding_side="left",
        device=device,
    )

    best_loss = eval_model(
        contracted_model,
        dataset=retain_in_sample_validation,
        truncate_at=experiment_cfg.truncate_batch_tokens_at,
    )["loss"]
    best_model_weights = deepcopy(contracted_model.state_dict())
    best_step = 0

    for step in (
        pbar := tqdm.tqdm(range(run_type_cfg.num_steps_coherence_finetuning + 1))
    ):
        loss = training.compute_preds_and_get_ce_loss(
            contracted_model, input_ids, attention_mask, None
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        wandb.log({"loss": loss.item()})

        test_retain_loss = eval_model(
            contracted_model,
            dataset=retain_in_sample_validation,
            truncate_at=experiment_cfg.truncate_batch_tokens_at,
        )["loss"]
        if test_retain_loss < best_loss:
            best_loss = test_retain_loss
            best_model_weights = deepcopy(contracted_model.state_dict())
            best_step = step
        wandb.log({"coherence_test_loss": test_retain_loss})

        if step % 15 == 10 or step == total_steps - 1:
            validation_forget_loss, validation_retain_loss = eval_on_validation(
                contracted_model,
                forget_validation_stories[: 5 * experiment_cfg.batch_size],
                retain_validation_stories[: 5 * experiment_cfg.batch_size],
                experiment_cfg.truncate_batch_tokens_at,
            )
            wandb.log(
                {
                    "validation_forget_loss": validation_forget_loss["loss"],
                    "validation_retain_loss": validation_retain_loss["loss"],
                }
            )
    wandb.run.summary["best_coherence_step"] = best_step  # type: ignore
    print(
        f"Best coherence training loss {best_loss:0.3f} achieved at step {best_step}."
    )

    contracted_model.load_state_dict(best_model_weights)
    if model_save_path is not None:
        model_store.save_model(contracted_model, model_save_path)

    # STAGE 3
    print("PHASE THREE: RETRAINING")
    retrain_cfg = RetrainExperimentConfig(
        words_to_localize=experiment_cfg.words_to_localize,
        num_stories_to_retrain=[1] if dry_run else num_stories_to_retrain,
        num_steps=1 if dry_run else run_type_cfg.num_steps_forget_set_retraining,
        eval_batch_size=experiment_cfg.batch_size,
        max_tokens=experiment_cfg.truncate_batch_tokens_at,
        pure_model_path=None,
        base_model_path=None,
        model_save_path=model_save_path,
        model_type=experiment_cfg.transformer_lens_model_name,
        prompt=experiment_cfg.unlearning_eval_prompt,
        dry_run=dry_run,
        eval_interval=1,
        num_times_to_retrain=5,
    )
    figs, res_df = run_retrain_evals(
        forget_validation_stories, retain_validation_stories, retrain_cfg, device
    )
    res_df.insert(0, "run_type", run_type_cfg.label)
    res_df.insert(1, "model_save_name", model_save_name)
    res_df.insert(2, "random_seed", random_shuffle_seed)
    if additional_fields_to_save_to_df is not None:
        for idx, (key, value) in enumerate(additional_fields_to_save_to_df.items()):
            res_df.insert(idx + 3, key, value)

    if validation_data_save_dir is not None:
        res_df.to_csv(
            os.path.join(validation_data_save_dir, f"{model_save_name}_relearn.csv")
        )

    for i, fig in enumerate(figs):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.savefig(tmpfile.name, format="png")
            wandb.log({f"fig_{i}": wandb.Image(tmpfile.name)})

    wandb.finish()
    return res_df


if __name__ == "__main__":
    from factored_representations.utils import get_gpu_with_most_memory

    device = get_gpu_with_most_memory()
    print(f"Training single tinystories_era run on {device=}.")

    """
    $(pdm venv activate) && python projects/tinystories/tinystories_era.py
    """

    use_era = True
    model_save_name = "demix_debug_era" if use_era else "demix_debug_demix"
    model_save_dir = "debugging_demix"
    DRY_RUN = len(sys.argv) > 1 and sys.argv[1] == "dry_run"

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
    demix_cfg = shared_settings.ERAConfig(
        layers_to_mask=[],
        to_expand={},
        masking_scheme="full_seq",
        masking_type="demix",
        expanded_vs_original_dim_learning_rates=dict(
            expanded_dim_lr_target=1.0,
            original_dim_lr_target=1.0,
            expanded_dim_lr_off_target=1.0,
            original_dim_lr_off_target=1.0,
        ),
        include_conditional_bias_term=False,
    )

    era_steps = 12_500
    coherence_finetuning = (
        -1  # want to see if that weird bug also happens when doing era with no coherence, doing -1 because the function above does +1 for some reason
    )
    forget_set_retraining = 40

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
        l1_coeff=1e-4,
    )

    res_df = do_era_training_run(
        experiment_cfg=shared_settings.cfg,
        run_type_cfg=erac_model_cfg,
        era_cfg=era_cfg if use_era else demix_cfg,
        random_shuffle_seed=0,
        num_validation_stories=100,
        num_stories_to_retrain=[64],
        device=device,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name,
        overwrite_model_saves=True,
        validation_data_save_dir="data_debugging",
        dry_run=DRY_RUN,
    )
    res_df.to_csv(f"data_debugging/{model_save_name}_retrain.csv")
