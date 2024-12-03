# %%

from transformer_lens.loading_from_pretrained import get_pretrained_model_config
from transformer_lens import HookedTransformer
import torch as t
import matplotlib.pyplot as plt
import numpy as np

import factored_representations.string_utils as string_utils
import factored_representations.utils as utils
import shared_configs.model_store as model_store
from projects.tinystories.shared_settings import cfg as experiment_cfg
import factored_representations.model_expansion as model_expansion
import projects.tinystories.bulk_runs_for_paper as bulk_runs

if __name__ == "__main__":
    all_stories = string_utils.load_dataset_with_split(
        "delphi-suite/stories",
        "validation",
        max_stories=1000,
    )

    # Load data
    truncated_stories = string_utils.truncate_stories_by_chars(
        all_stories, max_character_len=experiment_cfg.truncate_story_chars_at
    )
    forget_stories, retain_stories = string_utils.split_and_label_stories_by_concept(
        truncated_stories, experiment_cfg.words_to_localize
    )

    device = utils.get_gpu_with_most_memory()

    model_save_name = "e11-save-pre-ablation_ERAC_seed578498_pre_ablation"

    config = get_pretrained_model_config(
        experiment_cfg.transformer_lens_model_name, device=device
    )
    old_model = HookedTransformer(config)
    model, specs = model_expansion.expand_and_get_mask_specs(
        old_model,
        bulk_runs.era_cfg.to_expand if bulk_runs.erac_model_cfg.expand_model else {},
        layers_to_mask=bulk_runs.era_cfg.layers_to_mask,
        masking_rule=None,  # type: ignore
        suppress_warnings=False,
        **bulk_runs.era_cfg.expanded_vs_original_dim_learning_rates,
        weight_initialization_coeff=1.0,
    )

    model_store.load_weights(
        model,
        f"bulk_runs_for_paper/{model_save_name}",
    )

    num_samples = 40
    forget = [story for story, _ in forget_stories[:num_samples]]
    retain = [story for story, _ in retain_stories[:num_samples]]

    input_ids, attention_mask = string_utils.tokenize_batch(
        forget + retain,
        model.tokenizer,
        prepend_bos=True,
        truncate_at=experiment_cfg.truncate_batch_tokens_at,
        padding_side="right",
        device=device,
    )

    mlp_out = model.blocks[0].mlp.W_out.detach()

    mlp_post_unablated = [t.tensor(0)]

    def mlp_out_hook(value, hook):
        mlp_post_unablated[0] = value

    block_idx = 0
    model.add_hook(
        f"blocks.{block_idx}.mlp.hook_post",
        mlp_out_hook,  # type: ignore
    )

    with t.inference_mode():
        _ = model.forward(input_ids, stop_at_layer=block_idx + 1)  # type: ignore

    exp_idx = -64
    original_dim_out = mlp_post_unablated[0][..., :exp_idx] @ mlp_out[:exp_idx, :]
    expanded_dim_out = mlp_post_unablated[0][..., exp_idx:] @ mlp_out[exp_idx:, :]

    original_dim_out_avg = original_dim_out[:num_samples].mean(dim=(0, 1)).cpu().numpy()
    expanded_dim_out_avg = expanded_dim_out[:num_samples].mean(dim=(0, 1)).cpu().numpy()

    mlp_out_forget_avg = (
        mlp_post_unablated[0][:num_samples].mean(dim=(0, 1)).cpu().numpy()
    )
    mlp_out_retain_avg = (
        mlp_post_unablated[0][num_samples:].mean(dim=(0, 1)).cpu().numpy()
    )

    def simple_linear_regression(x, y):
        x_mat = np.stack((np.ones_like(x), x)).T
        return np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ y)

    slope, intercept = simple_linear_regression(
        original_dim_out_avg, expanded_dim_out_avg
    )
    # generate points from line
    x = np.linspace(original_dim_out_avg.min(), original_dim_out_avg.max(), 100)
    y = slope * x + intercept

    fig, ax = plt.subplots()
    ax.scatter(original_dim_out_avg, expanded_dim_out_avg, alpha=0.3)
    ax.plot(x, y, color="black", linestyle="--")
    ax.annotate(
        f"y = {slope:.2f}x + {intercept:.2f}", (0.5, 0.5), xycoords="axes fraction"
    )
    ax.set_title(
        "Avg residual-stream-dimension output from first layer (original, expanded) \n"
        "MLP dims on forget data"
    )
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Original dims")
    ax.set_ylabel("Expanded dims")

    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
    ax[0].set_title("Average MLP activations on forget data")
    ax[1].set_title("Average MLP activations on retain data")
    ax[0].bar(range(len(mlp_out_forget_avg)), mlp_out_forget_avg)
    ax[1].bar(range(len(mlp_out_forget_avg)), mlp_out_retain_avg)
    ax[0].axvline(len(mlp_out_forget_avg) + exp_idx, color="black", linestyle="--")
    ax[1].axvline(len(mlp_out_forget_avg) + exp_idx, color="black", linestyle="--")
