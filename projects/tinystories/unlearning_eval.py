from typing import List, Optional, Tuple

import pandas as pd
from transformer_lens import HookedTransformer

from factored_representations import masklib, string_utils


def unlearning_eval(
    model: HookedTransformer,
    mask_applier: Optional[masklib.MaskApplier],
    words_to_localize: list[str],
    eval_str: str,
    n_samples: int,
) -> Tuple[pd.DataFrame, str]:
    """
    Measure occurrence of target words in a prompt designed to elicit them; if unlearning
    worked, then the ablated model should not say these words.

    If mask_applier is None, then the model is assumed to be collapsed and only unlearning-baseline is calculated.
    """
    num_in_prompt = string_utils.count_target_words_in_story(
        eval_str, words_to_localize
    )

    # Regular and ablated forward passes
    forward_kwargs = dict(
        max_new_tokens=50, temperature=0.7, prepend_bos=True, verbose=False
    )
    forward_regular: List[str] = [
        model.generate(eval_str, **forward_kwargs)  # type: ignore
        for _ in range(n_samples)
    ]
    if mask_applier is not None:
        with mask_applier.zero_ablate(mask_idx=2):
            forward_ablated: List[str] = [
                model.generate(eval_str, **forward_kwargs)  # type: ignore
                for _ in range(n_samples)
            ]

    results = []
    if mask_applier is None:
        both_completions = {"unlearning-baseline": forward_regular}
    else:
        both_completions = {
            "unlearning-baseline": forward_regular,
            "unlearning-ablated": forward_ablated,  # type: ignore
        }
    for label, completions in both_completions.items():
        for completion in completions:
            num_new = (
                string_utils.count_target_words_in_story(completion, words_to_localize)
                - num_in_prompt
            )
            results.append((label, completion, num_new))

    formatted_baseline = string_utils.format_stories(forward_regular)
    if mask_applier is None:
        all_formatted_stories = formatted_baseline
    else:
        formatted_ablated = string_utils.format_stories(forward_ablated)  # type: ignore
        all_formatted_stories = (
            f"Baseline:\n{formatted_baseline}\n{'+'*20}\nAblated:\n{formatted_ablated}"
        )
    df = pd.DataFrame.from_records(
        results, columns=["label", "completion", "num_extra_localized_words"]
    )
    return df, all_formatted_stories
