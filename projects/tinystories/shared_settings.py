# %%
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

"""
In order for two runs to be comparable, it is necessary but not sufficient for them
to share the same SharedExperimentConfig.
"""


@dataclass
class SharedExperimentConfig:
    # Model loading
    transformer_lens_model_name: str

    # Data
    total_num_stories_to_load: int  # pure model will subset to retain only

    # Training config
    batch_size: int
    grad_accum_steps: int
    truncate_story_chars_at: int  # applied first, before splitting retain/forget
    truncate_batch_tokens_at: int  # applied last, after tokenization
    learning_rate: float
    decay_learning_rate: bool
    optimizer_kwargs: dict

    # Case-specific configs
    words_to_localize: list[str]
    unlearning_eval_prompt: str
    wandb_project_subname: str

    def hash(self) -> str:
        json_string = json.dumps(self.__dict__, sort_keys=True)
        hash_object = hashlib.sha256(json_string.encode())
        hash_hex = hash_object.hexdigest()
        return f"h{hash_hex[:16]}"


@dataclass
class ERAConfig:
    layers_to_mask: list[int]
    to_expand: dict[str, int]
    masking_scheme: Optional[str]
    masking_type: Optional[str]
    expanded_vs_original_dim_learning_rates: dict[str, float]

    include_conditional_bias_term: bool


@dataclass
class RunTypeConfig:
    label: str
    pretrained_model_to_load: Optional[str]
    anneal_gradient_mask_weights: bool
    mask_weight_increase_steps: int

    expand_model: bool  # If False, ignores all ERA parameters
    use_gradient_routing: bool

    # Partial oversight
    forget_data_labeling_percentage: float
    drop_labeled_forget_data: bool
    drop_unlabeled_forget_data: bool
    sort_forget_data_by_label: bool

    # Amount of training
    num_steps_era_training: int
    num_steps_coherence_finetuning: int
    num_steps_forget_set_retraining: int

    l1_coeff: float


cfg = SharedExperimentConfig(
    transformer_lens_model_name="roneneldan/TinyStories-28M",
    total_num_stories_to_load=500_000,  # delphi-suite/stories has 2_705_118 stories
    batch_size=80,
    grad_accum_steps=1,
    truncate_story_chars_at=1029,  # see below for how this was chosen
    truncate_batch_tokens_at=256,
    learning_rate=5e-4,
    decay_learning_rate=True,
    optimizer_kwargs=dict(betas=(0.9, 0.95), weight_decay=0.1),
    words_to_localize=[
        "tree",
        "trees",
        "forest",
        "forests",
        "woodland",
        "woodlands",
    ],
    unlearning_eval_prompt="Once upon a time, Timmy went to the forest",
    wandb_project_subname="forest",
)


def print_info_for_model_tracker(
    model_filename: str, model_dir: str, url: str, hash_str: str
):
    print()
    print("RUN INFO - drag to copy from tmux, then paste into spreadsheet:")
    print(
        "https://docs.google.com/spreadsheets/d/1qcxNo2DAo38kN79Czxj0XQIJ295zwg4aX4SEF3xiejk/"
    )
    print()
    print(model_filename)
    print(model_dir)
    print(datetime.now().date().strftime("%b %-d"))
    print(hash_str)
    print(url)


def get_suggested_character_truncation_length(
    stories: list[str], max_tokens: int, transformer_lens_model_name: str
) -> int:
    model = HookedTransformer.from_pretrained(transformer_lens_model_name, device="cpu")
    char_truncation_len = string_utils.estimate_truncation_length(
        stories, model.tokenizer, max_tokens
    )
    return char_truncation_len


if __name__ == "__main__":
    from copy import deepcopy

    from datasets import load_dataset
    from transformer_lens import HookedTransformer

    import factored_representations.string_utils as string_utils

    # Test hashing
    print(f"{cfg.hash()=}")

    cfg2 = deepcopy(cfg)
    assert cfg.hash() == cfg2.hash()

    cfg2.learning_rate = 1
    assert cfg.hash() != cfg2.hash()

    dataset = load_dataset("delphi-suite/stories", split="train")
    stories: list[str] = dataset["text" if "text" in dataset.column_names else "story"]  # type: ignore

    truncate_story_chars_at = get_suggested_character_truncation_length(
        stories[:25000],
        max_tokens=cfg.truncate_batch_tokens_at,
        transformer_lens_model_name=cfg.transformer_lens_model_name,
    )
    print(f"Recommend {truncate_story_chars_at=}")

    count_stories = False
    if count_stories:
        truncated_stories = string_utils.truncate_stories_by_chars(
            stories, truncate_story_chars_at
        )

        forget_stories, retain_stories = string_utils.split_stories_by_concept(
            truncated_stories, cfg.words_to_localize
        )
        pct = len(forget_stories) / (len(forget_stories) + len(retain_stories))
        print(f"{len(forget_stories)=} ({pct:0.2f} of total)")
        print(f"{len(retain_stories)=} ({1-pct:0.2f} of total)")
