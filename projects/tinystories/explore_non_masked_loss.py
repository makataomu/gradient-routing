# %%

"""
The purpose of this script is to test the hypothesis that the ERAC model is not
actually broadly localizing forest concepts, but is instead merely assigning very
low probabilities to key words in the forget set, resulting in high loss but
not-actually-deep unlearning.

We test this by measuring the loss on the forget set, ignoring tokens that are routed on,
proportional to how aggressively they are routed.
"""

import torch as t
import torch.utils.data as data

import factored_representations.training as training
import factored_representations.masklib as masklib
import factored_representations.string_utils as string_utils
import shared_configs.model_store as model_store
from projects.tinystories.shared_settings import cfg as experiment_cfg

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

    retain_loader = data.DataLoader(
        string_utils.ListDataset(forget_stories),
        shuffle=False,
        batch_size=experiment_cfg.batch_size,
    )

    forget_loader = data.DataLoader(
        string_utils.ListDataset(forget_stories),
        shuffle=False,
        batch_size=experiment_cfg.batch_size,
    )

    device = t.device("cpu")  # utils.get_gpu_with_most_memory()

    seed = 246062

    model = model_store.load_model(
        f"bulk_runs_for_paper/e7_ERAC_seed{seed}",
        "roneneldan/TinyStories-28M",
        device=device,
    )

    num_stories = 25000
    token_freq_masking_rule, info = masklib.get_token_freq_masking_rule(
        retain_stories=[story for story, _ in retain_stories],
        forget_stories=[story for story, _ in forget_stories],
        num_stories=num_stories,
        truncate_at=None,
        num_synthetic_tokens_retain=20,
        num_synthetic_tokens_forget=20,
        scale=2.0,
        bias=-1.5,
        tokenizer=model.tokenizer,
        device=device,  # type: ignore
    )

    batch = next(iter(forget_loader))
    stories, labels = batch
    input_ids, attention_mask = string_utils.tokenize_batch(
        stories,
        model.tokenizer,
        prepend_bos=True,
        truncate_at=experiment_cfg.truncate_batch_tokens_at,
        padding_side="right",
        device=device,
    )

    masks = token_freq_masking_rule(input_ids)  # 0 is forget, 1 is retain

    with t.inference_mode():
        forget_loss_unreduced, attn_mask_out = (
            training.compute_preds_and_get_ce_loss_unreduced(
                model, input_ids, attention_mask, None
            )
        )

        overall_loss = forget_loss_unreduced.sum() / attn_mask_out.sum()
        unmasked_loss = (forget_loss_unreduced * masks).sum() / (
            attn_mask_out * masks
        ).sum()
        masked_loss = (forget_loss_unreduced * (1 - masks)).sum() / (
            attn_mask_out * (1 - masks)
        ).sum()

    print("Overall forget loss (base model):   1.297")  # from wandb
    print("Overall forget loss (pre-ablation): 1.529")  # from wandb
    print("-- Post-ablation, coherence finetuning --")
    print(f"Overall forget loss:                {overall_loss.item():0.3f}")
    print(f"Forget loss under routing mask:     {masked_loss.item():0.3f}")
    print(f"Forget loss not under routing mask: {unmasked_loss.item():0.3f}")
