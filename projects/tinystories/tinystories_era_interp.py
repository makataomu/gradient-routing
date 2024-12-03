# %%
import inspect
import os
from copy import deepcopy
from typing import Optional

import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from einops import einsum
from plotly.subplots import make_subplots
from transformer_lens import HookedTransformer

from factored_representations import string_utils
from factored_representations.utils import get_gpu_with_most_memory
from projects.tinystories import shared_settings
from projects.tinystories.retraining_evals import (
    RetrainExperimentConfig,
)
from shared_configs.model_store import load_model


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()  # type: ignore
    var_names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    return " ".join(var_names[0].split("_")).title()


bye = 4


def gettit(hi):
    n = retrieve_name(hi)
    print(n)


gettit(bye)
# preablation_model


# %%
device = get_gpu_with_most_memory()
coherence_finetuning = 60
forget_set_retraining = 20
layers_to_mask = ([0, 1, 2, 3, 4],)
experiment_cfg = shared_settings.SharedExperimentConfig(
    transformer_lens_model_name="roneneldan/TinyStories-8M",
    total_num_stories_to_load=1_000_000,  # delphi-suite/stories has 2_705_118 stories
    batch_size=80,
    grad_accum_steps=1,
    truncate_story_chars_at=1029,  # see below for how this was chosen
    truncate_batch_tokens_at=256,
    learning_rate=5e-4,
    decay_learning_rate=True,
    optimizer_kwargs=dict(betas=(0.9, 0.95), weight_decay=0.1),
    words_to_localize=[
        " tree",
        " trees",
        " forest",
        " forests",
        " woodland",
        " woodlands",
    ],
    unlearning_eval_prompt="Once upon a time, Timmy went to the forest",
    wandb_project_subname="forest",
)
print("PHASE THREE: RETRAINING")
model_save_dir = "tinystories_era_interp"
model_save_name = "era_test"
model_save_path = os.path.join(model_save_dir, model_save_name)
retrain_cfg = RetrainExperimentConfig(
    words_to_localize=experiment_cfg.words_to_localize,
    num_stories_to_retrain=[1, 4, 16, 64],
    num_steps=forget_set_retraining,
    eval_batch_size=experiment_cfg.batch_size,
    max_tokens=experiment_cfg.truncate_batch_tokens_at,
    pure_model_path="bulk_runs_for_paper/e2_pure_seed504363",
    base_model_path="bulk_runs_for_paper/e2_base_seed628920",
    model_save_path=model_save_path + "_post_coherence",
    model_type=experiment_cfg.transformer_lens_model_name,
    prompt=experiment_cfg.unlearning_eval_prompt,
    dry_run=False,
    eval_interval=1,
)


validation_stories = string_utils.load_dataset_with_split(
    "delphi-suite/stories", "validation", None
)
truncated_validation_stories = string_utils.truncate_stories_by_chars(
    validation_stories, experiment_cfg.truncate_story_chars_at
)
forget_validation_stories, retain_validation_stories = (
    string_utils.split_stories_by_concept(
        truncated_validation_stories,
        experiment_cfg.words_to_localize,
    )
)
forget_validation_stories = forget_validation_stories[:1000]
retain_validation_stories = retain_validation_stories[:1000]

# %%
# figs, res_df = run_retrain_evals(
#     forget_validation_stories, retain_validation_stories, retrain_cfg, device
# )

# Display the results
# for fig in figs:
#     fig.show()

# %%

original_model_path = "pretrained/pretrain_tinystories_base_2024-08-04_05-39-38"
original_model = load_model(
    original_model_path,
    experiment_cfg.transformer_lens_model_name,
    device,
)
preablation_config = deepcopy(original_model.cfg)
preablation_config.d_mlp += 64
preablation_model = load_model(
    f"{model_save_path}_pre_ablation", preablation_config, device
)
era_interp_models_dir = "tinystories_era_interp"
era_interp_model_prefix = f"{era_interp_models_dir}/era_test_"
model_config_name = "roneneldan/TinyStories-8M"
post_ablation_model = load_model(
    f"{era_interp_model_prefix}post_ablation", model_config_name, device
)
post_coherence_model = load_model(
    f"{era_interp_model_prefix}post_coherence", model_config_name, device
)

# %%
# Diff the weights between MLP 0 in the original model and the preablation model
original_mlp0_in_weight = original_model.blocks[0].mlp.W_in
# Expand original_mlp0_in_weight to match the preablation model
print("original_mlp0_in_weight.shape:", original_mlp0_in_weight.shape)
preablation_mlp0_in_weight = preablation_model.blocks[0].mlp.W_in
expanded_original_mlp0_in_weight = t.zeros_like(preablation_mlp0_in_weight)
expanded_original_mlp0_in_weight[:, : original_mlp0_in_weight.shape[1]] = (
    original_mlp0_in_weight
)
print("preablation_mlp0_in_weight.shape:", preablation_mlp0_in_weight.shape)
px.imshow(
    expanded_original_mlp0_in_weight.detach().cpu().numpy(),
    title="Expanded original MLP 0 W_in",
).show()
mlp0_in_weight_diff = preablation_mlp0_in_weight - expanded_original_mlp0_in_weight

# Plot a heatmap of the difference
px.imshow(mlp0_in_weight_diff.detach().cpu().numpy(), title="MLP 0 W_in diff").show()

# %%

batch_size = 20
forget_stories_batch = forget_validation_stories[:batch_size]
retain_stories_batch = retain_validation_stories[:batch_size]

forget_input_ids, forget_attention_mask = string_utils.tokenize_batch(
    forget_stories_batch,
    original_model.tokenizer,
    prepend_bos=True,
    truncate_at=experiment_cfg.truncate_batch_tokens_at,
    padding_side="right",
    device=device,
)
retain_input_ids, retain_attention_mask = string_utils.tokenize_batch(
    retain_stories_batch,
    original_model.tokenizer,
    prepend_bos=True,
    truncate_at=experiment_cfg.truncate_batch_tokens_at,
    padding_side="right",
    device=device,
)
with t.inference_mode():
    _, forget_cache = preablation_model.run_with_cache(
        forget_input_ids, attention_mask=forget_attention_mask
    )
    _, retain_cache = preablation_model.run_with_cache(
        retain_input_ids, attention_mask=retain_attention_mask
    )
# Plot the mean activation for each neuron in the MLPs
forget_neuron_acts = {
    f"MLP {i}": forget_cache[f"blocks.{i}.mlp.hook_post"].mean((0, 1)).detach().cpu()
    for i in range(preablation_model.cfg.n_layers)
}
retain_neuron_acts = {
    f"MLP {i}": retain_cache[f"blocks.{i}.mlp.hook_post"].mean((0, 1)).detach().cpu()
    for i in range(preablation_model.cfg.n_layers)
}
gradient_colors = pc.n_colors(
    "rgb(0,0,255)", "rgb(255,0,0)", len(forget_neuron_acts), colortype="rgb"
)
new_dim_start = original_model.cfg.d_mlp
forget_neuron_fig = px.line(
    forget_neuron_acts,
    title="Forget Set - MLP Acts",
    color_discrete_sequence=gradient_colors,
)
forget_neuron_fig.add_vline(x=new_dim_start, line_dash="dash")
forget_neuron_fig.add_annotation(
    x=new_dim_start,
    y=-2,
    text="Expanded Dims -> ",
    showarrow=False,
    xanchor="right",
    yanchor="bottom",
)
forget_neuron_fig.show()

retain_neuron_fig = px.line(
    retain_neuron_acts,
    title="Retain Set - MLP Acts",
    color_discrete_sequence=gradient_colors,
)
retain_neuron_fig.add_vline(x=new_dim_start, line_dash="dash")
retain_neuron_fig.add_annotation(
    x=new_dim_start,
    y=-0.7,
    text="Expanded Dims -> ",
    showarrow=False,
    xanchor="right",
    yanchor="bottom",
)
retain_neuron_fig.show()


# %%
# Plot the mean neuron activations for each MLP in a bar chart
forget_mean_acts = {
    f"MLP {i}": forget_cache[f"blocks.{i}.mlp.hook_post"].mean().detach().cpu()
    for i in range(preablation_model.cfg.n_layers)
}
retain_mean_acts = {
    f"MLP {i}": retain_cache[f"blocks.{i}.mlp.hook_post"].mean().detach().cpu()
    for i in range(preablation_model.cfg.n_layers)
}
gradient_colors = pc.n_colors(
    "rgb(0,0,255)", "rgb(255,0,0)", len(forget_mean_acts), colortype="rgb"
)
new_dim_start = original_model.cfg.d_mlp
fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=("Forget Set - MLP Acts", "Retain Set - MLP Acts"),
)
fig.add_trace(
    go.Bar(
        y=list(forget_mean_acts.values()),
        x=list(forget_mean_acts.keys()),
        marker=dict(color=gradient_colors),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        y=list(retain_mean_acts.values()),
        x=list(retain_mean_acts.keys()),
        marker=dict(color=gradient_colors),
    ),
    row=1,
    col=2,
)
fig.update_layout(showlegend=False)
fig.show()

# %%
# Same as the previous plot, but splitting the MLP into [:new_dim_start] and [new_dim_start:]
forget_main_dim_resids = {
    f"MLP {i}": forget_cache[f"blocks.{i}.mlp.hook_post"][:, :, :new_dim_start]
    .mean()
    .detach()
    .cpu()
    for i in range(preablation_model.cfg.n_layers)
}
forget_expanded_dim_resids = {
    f"MLP {i}": forget_cache[f"blocks.{i}.mlp.hook_post"][:, :, new_dim_start:]
    .mean()
    .detach()
    .cpu()
    for i in range(preablation_model.cfg.n_layers)
}
retain_main_dim_resids = {
    f"MLP {i}": retain_cache[f"blocks.{i}.mlp.hook_post"][:, :, :new_dim_start]
    .mean()
    .detach()
    .cpu()
    for i in range(preablation_model.cfg.n_layers)
}
retain_expanded_dim_resids = {
    f"MLP {i}": retain_cache[f"blocks.{i}.mlp.hook_post"][:, :, new_dim_start:]
    .mean()
    .detach()
    .cpu()
    for i in range(preablation_model.cfg.n_layers)
}
# Plot two bar charts, one for the forget set and one for the retain set
# Each bar chart will have two bars per MLP, one for the main dims and one for the expanded dims
gradient_colors = pc.n_colors(
    "rgb(0,0,255)", "rgb(255,0,0)", len(forget_main_dim_resids), colortype="rgb"
)
fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=("Forget Set - MLP Acts", "Retain Set - MLP Acts"),
)
for i, (main_dim_resids, expanded_dim_resids) in enumerate(
    zip(
        [forget_main_dim_resids, retain_main_dim_resids],
        [forget_expanded_dim_resids, retain_expanded_dim_resids],
    )
):
    for j, (title, dim_resids) in enumerate(
        zip(["Main Dims", "Expanded Dims"], [main_dim_resids, expanded_dim_resids])
    ):
        fig.add_trace(
            go.Bar(
                y=list(dim_resids.values()),
                x=list(dim_resids.keys()),
                name=title,
                marker=dict(
                    color=gradient_colors,
                    pattern_shape="\\" if j == 1 else None,
                    pattern_fillmode="overlay" if j == 1 else None,
                ),
                showlegend=i == 0,
            ),
            row=1,
            col=i + 1,
        )
fig.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
)
fig.show()

# %%


def visualize_neuron_input_output(
    model: HookedTransformer,
    mlp_idx: int,
    neuron_idx: int,
    valid_idxs: Optional[t.Tensor] = None,
):
    if valid_idxs is None:
        valid_idxs = t.arange(model.cfg.d_vocab, device=model.cfg.device)
    in_weights = model.blocks[mlp_idx].mlp.W_in[:, neuron_idx]
    out_weights = model.blocks[mlp_idx].mlp.W_out[neuron_idx]
    words_to_localize = experiment_cfg.words_to_localize
    words_to_localize_idxs = model.to_tokens(
        words_to_localize, prepend_bos=False, padding_side="right"
    )
    pad_token_idx = model.to_tokens(model.tokenizer.pad_token, prepend_bos=False).item()  # type: ignore
    single_tok_word_idxs = (words_to_localize_idxs != pad_token_idx).sum(dim=-1) == 1
    single_tok_word_toks = words_to_localize_idxs[single_tok_word_idxs, 0]
    single_tok_word_strs = model.to_str_tokens(single_tok_word_toks)
    single_tok_word_valid_idxs = (
        valid_idxs[None] == single_tok_word_toks[:, None]
    ).nonzero()[:, -1]

    with t.inference_mode():
        in_weights_embed = einsum(in_weights, model.embed.W_E[valid_idxs], "d,v d->v")
        in_weights_unembed = (
            einsum(in_weights, model.unembed.W_U[:, valid_idxs], "d,d v->v")
            + model.unembed.b_U[valid_idxs]
        )
        # in_weights_unembed = model.unembed(in_weights)
        out_weights_embed = einsum(out_weights, model.embed.W_E[valid_idxs], "d,v d->v")
        out_weights_unembed = (
            einsum(out_weights, model.unembed.W_U[:, valid_idxs], "d,d v->v")
            + model.unembed.b_U[valid_idxs]
        )
        # out_weights_unembed = model.unembed(out_weights)
        print("valid_idxs.shape:", valid_idxs.shape)

    titles = [
        "Input Weights (Embedded)",
        "Input Weights (Unembedded)",
        "Output Weights (Embedded)",
        "Output Weights (Unembedded)",
    ]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )
    for i, logits in enumerate(
        [in_weights_embed, in_weights_unembed, out_weights_embed, out_weights_unembed]
    ):
        row, col = divmod(i, 2)
        fig.add_trace(go.Histogram(x=logits.cpu().numpy()), row=row + 1, col=col + 1)
        for word_str, logit in zip(
            single_tok_word_strs, logits[single_tok_word_valid_idxs]
        ):
            fig.add_annotation(
                x=logit.item(),
                y=0,
                text=word_str,
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                ay=-t.randint(10, 60, (1,)).item(),
                ax=0,
                arrowwidth=2,
                xanchor="center",
                yanchor="bottom",
                row=row + 1,
                col=col + 1,
            )
    fig.update_layout(
        showlegend=False,
        title_text=f"{retrieve_name(model)}: MLP {mlp_idx} Neuron {neuron_idx}",
    )
    fig.update_layout(height=600, width=900)
    fig.show()


mlp_idx = 6
neuron_idx = 532
trained_tok_idxs = (original_model.unembed.W_U.norm(dim=0) > 0.8).nonzero().flatten()
print(trained_tok_idxs.shape)
visualize_neuron_input_output(
    preablation_model, mlp_idx, neuron_idx, valid_idxs=trained_tok_idxs
)
# Check the bias?

# %%
# with t.inference_mode():
#     plot_word_scores(
#         original_model.unembed.W_U.norm(dim=0),
#         original_model,
#         list_len=100,
#         title="Unembed W_U Token Norms (Original Model)",
#     ).show()


# %%
