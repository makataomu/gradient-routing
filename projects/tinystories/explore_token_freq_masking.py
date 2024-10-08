# type: ignore
# %%
import uuid

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
from transformers import PreTrainedTokenizer

import shared_configs.model_store as model_store
from factored_representations import masklib
from factored_representations.string_utils import (
    load_dataset_with_split,
    truncate_and_split_concept_data,
)

if __name__ == "__main__":
    SAVE_MODEL = True
    OVERWRITE_MODEL = False
    model_uuid = uuid.uuid4()
    SAVE_FILENAME = f"finetuned/my_test_run{model_uuid}"
    if SAVE_MODEL:
        model_store.ensure_save_path_exists(SAVE_FILENAME, OVERWRITE_MODEL)

    # Load base model
    model_name = "roneneldan/TinyStories-8M"
    device = "cpu"
    original_model_config = get_pretrained_model_config(model_name, device=device)
    old_model = HookedTransformer(original_model_config)
    tokenizer: PreTrainedTokenizer = old_model.tokenizer  # type: ignore

    # ERA settings
    words_to_localize = [
        "tree",
        "trees",
        "forest",
        "forests",
        "woodland",
        "woodlands",
    ]

    truncate_at = 256
    all_stories = load_dataset_with_split(
        dataset_name="delphi-suite/stories", split="train", max_stories=200_000
    )
    concept_stories, other_stories = truncate_and_split_concept_data(
        data=all_stories,
        tokenizer=tokenizer,
        concept_words=words_to_localize,
        max_tokens=truncate_at,
        use_approximate_truncation=True,
    )
    # %%
    num_stories = 25000
    token_freq_masking_rule, info = masklib.get_token_freq_masking_rule(
        retain_stories=other_stories,
        forget_stories=concept_stories,
        num_stories=num_stories,
        truncate_at=None,
        num_synthetic_tokens_retain=20,
        num_synthetic_tokens_forget=1,
        scale=1.0,
        bias=-4.0,
        tokenizer=tokenizer,
        device=device,  # type: ignore
    )

    vocab = tokenizer.batch_decode(
        t.arange(tokenizer.vocab_size, device=device)[:, None]
    )

    df = pd.DataFrame(info, index=vocab)  # type: ignore
    retain_wt = len(other_stories) / (len(other_stories) + len(concept_stories))
    freq = df.retain_freq * retain_wt + df.forget_freq * (1 - retain_wt)
    df.insert(0, "freq", freq)

    def plot_story_weights(story: str, tokenizer, weights, color, ax=None):
        story = story[:100]
        token_indexes = tokenizer(story)["input_ids"]
        weights = [weights[token_idx] for token_idx in token_indexes]
        tokens = tokenizer.tokenize(story)

        space_per_char = 0.1
        total_len = space_per_char * len(story)

        if ax is None:
            _, ax = plt.subplots(figsize=(total_len, 1))

        num_chars_so_far = 0
        x_locs = []
        for tok in tokens:
            x_loc = (num_chars_so_far + len(tok) / 2) * space_per_char
            x_locs.append(x_loc)
            num_chars_so_far += len(tok)

        ax.plot(x_locs, weights, alpha=0.8, color=color, linewidth=2)
        for x_loc, tok in zip(x_locs, tokens):
            if tok[0] == "Ġ":
                tok = tok[1:]
            ax.text(
                x_loc, 0, tok, fontsize=10, ha="center", va="bottom", font="monospace"
            )

        ax.axhline(0, color="black", linestyle=":", alpha=1)
        ax.axhline(1, color="black", linestyle=":", alpha=1)
        ax.set_xlim(0, space_per_char * len(story))
        ax.set_ylim(0, 1)
        ax.axis("off")

    nrows = 10
    fig, axes = plt.subplots(nrows=nrows, figsize=(10, nrows))
    for i in range(nrows):
        plot_story_weights(
            concept_stories[i],
            tokenizer,
            info["mask_weight"],
            ax=axes[i],  # type: ignore
            color="C1",
        )
    fig.suptitle("Token weights in forget stories")
    plt.tight_layout()

    nrows = 10
    fig, axes = plt.subplots(nrows=nrows, figsize=(10, nrows))
    for i in range(nrows):
        plot_story_weights(
            other_stories[i],
            tokenizer,
            info["mask_weight"],
            ax=axes[i],  # type: ignore
            color="C2",
        )
    fig.suptitle("Token weights in retain stories")
    plt.tight_layout()
    # %%
    token_freq = 5
    token_base_rate = 1e4
    nonrare = df[df["freq"] > token_freq / token_base_rate]

    freq_cols = ["freq", "retain_freq", "forget_freq"]
    for col in freq_cols:
        nonrare.insert(0, f"{col}_10k", nonrare[col] * token_base_rate)
        del nonrare[col]

    nonrare = nonrare.sort_values("mask_weight")  # type: ignore
    cols_to_keep = ["forget_freq_10k", "retain_freq_10k", "mask_weight"]

    nonrare_to_paper = pd.concat(
        [nonrare[cols_to_keep].head(10), nonrare[cols_to_keep].tail(10)]
    )
    nonrare_to_paper.columns = ["Forget freq.", "Retain freq.", "Mask weight"]

    # fmt: off
    nonrare_to_paper["Forget freq."] = nonrare_to_paper["Forget freq."].map('{:.1f}'.format)
    nonrare_to_paper["Retain freq."] = nonrare_to_paper["Retain freq."].map('{:.1f}'.format)
    nonrare_to_paper["Mask weight"] = nonrare_to_paper["Mask weight"].map('{:.3f}'.format)
    nonrare_to_paper.index = [f"{tok.replace(" ","\\_")}" for tok in nonrare_to_paper.index]
    latex = nonrare_to_paper.to_latex()
    print(latex)
    # fmt: on

    # Plot distribution of mask weights for non-rare tokens
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(nonrare.mask_weight, bins=30)
    ax.set_xlabel("Mask weight (0=retain, 1=forget)")
    ax.set_ylabel(f"Frequency in {num_stories} stories")
    ax.set_title(
        f"Mask weight distribution for common tokens (>{token_freq} per {token_base_rate:0.0f} tokens)"
    )

    for i, p in enumerate(patches):  # type: ignore
        if i in [1, 5, 15]:
            continue

        bin_center = (bins[i] + bins[i + 1]) / 2
        words_in_bin = nonrare[
            (nonrare.mask_weight >= bins[i]) & (nonrare.mask_weight <= bins[i + 1])
        ]
        top_words = (
            words_in_bin.sort_values("freq_10k", ascending=False).head(5).index.tolist()  # type: ignore
        )
        if top_words:
            annotation = "\n".join(top_words)
            ax.annotate(
                annotation,
                (bin_center, p.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()

# %%
