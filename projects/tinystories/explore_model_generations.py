# %%


import factored_representations.string_utils as string_utils
import factored_representations.utils as utils
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

    device = utils.get_gpu_with_most_memory()

    seeds = [343771, 409648]
    seed = seeds[0]

    experiment_prefix = "e11-long"

    model_types = ["base", "pure", "ERAC"]
    models = {
        model_type: model_store.load_model(
            f"bulk_runs_for_paper/{experiment_prefix}_{model_type}_seed{seed}",
            "roneneldan/TinyStories-28M",
            device=device,
        )
        for model_type in model_types
    }

    forget_prompt = "Once upon a time, there was a big oak tree."
    retain_prompt = "Once upon a time, there was a kind girl named Lily."

    to_print = []
    for prompt in [forget_prompt, retain_prompt]:
        for model_type, model in models.items():
            out = model.generate(
                prompt, max_new_tokens=200, prepend_bos=True, temperature=0.8
            )
            to_print.append((model_type, out))

    for model_type, out in to_print:
        print(f"Model type: {model_type}")
        print(out)
        print("\n")
