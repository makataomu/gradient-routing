# %%
import random
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
import torch.utils.data as data
import transformers
from matplotlib.lines import Line2D
from tqdm import tqdm

import factored_representations.training as training
import shared_configs.model_store as model_store
from factored_representations import string_utils
from factored_representations.utils import get_gpu_with_most_memory, upload_to_clbin
from projects.tinystories.unlearning_eval import unlearning_eval


@dataclass
class RetrainExperimentConfig:
    words_to_localize: List[str]
    num_stories_to_retrain: List[int]
    num_steps: int
    eval_batch_size: int
    max_tokens: int
    pure_model_path: Optional[str]
    base_model_path: Optional[str]
    model_save_path: str
    model_type: str
    prompt: str
    retrain_all_steps: bool = False
    dry_run: bool = False
    eval_interval: Optional[int] = 1
    num_times_to_retrain: int = 1
    test_retain_stories: int = 1000
    test_forget_stories: int = 1000


@t.inference_mode()
def eval_model(
    model,
    dataset: list[tuple],
    truncate_at: int,
) -> Dict[str, float]:
    total_losses = {}
    dataloader = data.DataLoader(
        string_utils.ListDataset(dataset), batch_size=128, shuffle=False
    )
    torch_device = t.device(model.cfg.device)
    batch_losses = []
    for batch in dataloader:
        stories, labels = batch
        tokens, attention_mask = string_utils.tokenize_batch(
            stories,
            model.tokenizer,
            prepend_bos=True,
            truncate_at=truncate_at,
            padding_side="right",
            device=torch_device,
        )

        loss = training.compute_preds_and_get_ce_loss(
            model, tokens, attention_mask, None
        )
        batch_losses.append(loss.item())

    total_losses["loss"] = sum(batch_losses) / len(batch_losses)
    return total_losses


def eval_words(model, words_to_localize: List[str], prompt: str):
    df, all_formatted_stories = unlearning_eval(
        model,
        None,
        words_to_localize,
        prompt,
        n_samples=4,
    )
    df = (
        df.groupby(["label"])
        .agg({"num_extra_localized_words": "sum"})
        .to_dict()["num_extra_localized_words"]["unlearning-baseline"]
    )
    print("Retraining eval", upload_to_clbin(all_formatted_stories))
    return df


def retrain_model(
    model,
    test_forget_dataset: list[tuple],
    test_retain_dataset: list[tuple],
    train_forget_stories: list[tuple],
    num_stories: int,
    num_steps: int,
    all_steps: bool,
    words_to_localize: list[str],
    prompt: str,
    truncate_at: int,
    eval_interval: Optional[int] = None,
):
    input_ids, attention_mask = string_utils.tokenize_batch(
        [story for story, _ in train_forget_stories[:num_stories]],
        tokenizer=model.tokenizer,
        prepend_bos=True,
        truncate_at=truncate_at,
        padding_side="right",
        device=t.device(model.cfg.device),
    )

    lr = 5e-5
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    eval_losses = {"update_step": [], "forget": [], "retain": []}
    # eval_losses["token_count"].append(eval_words(model, words_to_localize, prompt))

    consecutive_increases = 0
    prev_forget_loss = float("inf")

    for step in (pbar := tqdm(range(num_steps))):
        loss = training.compute_preds_and_get_ce_loss(
            model, input_ids, attention_mask, None
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})

        if eval_interval is not None:
            # Empirically, the curves go linear after the first half of training,
            # so we do't need to evaluate as often
            eval_interval_sped_up = (
                eval_interval if step < num_steps / 2 else eval_interval + 1
            )

            if step % eval_interval_sped_up == 0 or step == num_steps - 1:
                forget_loss = eval_model(model, test_forget_dataset, truncate_at)
                retain_loss = eval_model(model, test_retain_dataset, truncate_at)
                eval_losses["update_step"].append(step + 1)
                eval_losses["forget"].append(forget_loss)
                eval_losses["retain"].append(retain_loss)
                if forget_loss["loss"] > prev_forget_loss:
                    consecutive_increases += 1
                else:
                    consecutive_increases = 0
                prev_forget_loss = forget_loss["loss"]
                # eval_losses["token_count"].append(
                #     eval_words(model, words_to_localize, prompt)
                # )
                if consecutive_increases >= 3 and not all_steps:
                    print(
                        f"Early stopping at step {step + 1} due to increasing forget loss for 3 consecutive evaluations"
                    )
                    break

    return eval_losses


def plot_results(
    res: Dict[int, Dict],
    pure_model,
    base_model,
    test_forget_stories: list[tuple],
    test_retain_stories: list[tuple],
    truncate_at: int,
    loss_type: str,
):
    assert loss_type in ["loss", "loss_under_mask"]
    print("doing pure and base evals")

    pure_model_forget = eval_model(
        pure_model,
        test_forget_stories,
        truncate_at=truncate_at,
    )
    pure_model_retain = eval_model(
        pure_model,
        test_retain_stories,
        truncate_at=truncate_at,
    )
    base_model_forget = eval_model(
        base_model,
        test_forget_stories,
        truncate_at=truncate_at,
    )
    base_model_retain = eval_model(
        base_model,
        test_retain_stories,
        truncate_at=truncate_at,
    )

    fig, ax = plt.subplots()
    for dataset_type in ["forget", "retain"]:
        pure_benchmark = (
            pure_model_forget if dataset_type == "forget" else pure_model_retain
        )[loss_type]
        base_benchmark = (
            base_model_forget if dataset_type == "forget" else base_model_retain
        )[loss_type]
        diff = pure_benchmark - base_benchmark
        ax.axhline(base_benchmark, c="gray")
        ax.axhline(pure_benchmark, c="gray")
        ax.annotate(
            f"pure model ({dataset_type})",
            (0, pure_benchmark - diff * 0.04),
            c="gray",
        )
        ax.annotate(
            f"base model ({dataset_type})",
            (0, base_benchmark + diff * 0.01),
            c="gray",
        )
    for idx, (num_stories, eval_losses) in enumerate(res.items()):
        forget_losses = [losses[loss_type] for losses in eval_losses["forget"]]
        retain_losses = [losses[loss_type] for losses in eval_losses["retain"]]
        ax.plot(forget_losses, label=f"{num_stories=} {loss_type}", c=f"C{idx}")
        ax.plot(retain_losses, ls=":", c=f"C{idx}", alpha=0.3)
    ax.set_title(f"[{loss_type}] Loss when training on small sample of forget stories")
    ax.set_xlabel("Number of updates")
    ax.set_ylabel("Cross-entropy loss")
    solid_line = Line2D([0], [0], color="gray", label="Forget set")
    dotted_line = Line2D([0], [0], color="gray", ls=":", label="Retain set", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([solid_line, dotted_line])
    labels.extend(["Forget set", "Retain set"])
    ax.legend(handles=handles, labels=labels)
    return fig, ax


def results_to_df(res):
    records = []
    for num_stories, res_dict in res.items():
        for update_step, forget, retain in zip(
            res_dict["update_step"], res_dict["forget"], res_dict["retain"]
        ):
            records.append((num_stories, update_step, forget["loss"], retain["loss"]))
    df = pd.DataFrame.from_records(
        records, columns=["num_stories", "update_step", "forget_loss", "retain_loss"]
    )
    return df


def run_retrain_evals(
    forget_stories, retain_stories, cfg: RetrainExperimentConfig, device
):
    max_num_stories = max(cfg.num_stories_to_retrain)

    dfs = []
    for _ in range(cfg.num_times_to_retrain):
        test_retain_stories = random.sample(retain_stories, cfg.test_retain_stories)
        forget_sampled = random.sample(
            forget_stories, cfg.test_forget_stories + max_num_stories
        )
        random.shuffle(forget_stories)
        train_forget_stories = forget_sampled[:max_num_stories]
        test_forget_stories = forget_sampled[max_num_stories:]
        results = get_experiment_results(
            train_forget_stories, test_forget_stories, test_retain_stories, cfg, device
        )
        res_df = results_to_df(results)
        dfs.append(res_df)
    # mean the dfs along the numeric columns
    mean_df = (
        pd.concat(dfs).groupby(["num_stories", "update_step"]).mean().reset_index()
    )

    figs = []

    # Only plot if passed models to eval
    # if cfg.pure_model_path is not None and cfg.base_model_path is not None:
    #     pure_model = model_store.load_model(cfg.pure_model_path, cfg.model_type, device)
    #     base_model = model_store.load_model(cfg.base_model_path, cfg.model_type, device)

    #     fig, ax = plot_results(
    #         res=results,
    #         pure_model=pure_model,
    #         base_model=base_model,
    #         test_forget_stories=test_forget_stories,
    #         test_retain_stories=retain_stories[:max_num_stories],
    #         truncate_at=cfg.max_tokens,
    #         loss_type="loss",
    #     )
    #     figs.append(fig)
    return figs, mean_df


def get_experiment_results(
    train_forget_stories,
    test_forget_stories,
    retain_stories,
    cfg: RetrainExperimentConfig,
    device,
):
    model = model_store.load_model(cfg.model_save_path, cfg.model_type, device)
    models = [model]
    if cfg.base_model_path is not None:
        base_model = model_store.load_model(cfg.base_model_path, cfg.model_type, device)
        models.append(base_model)
    if cfg.pure_model_path is not None:
        pure_model = model_store.load_model(cfg.pure_model_path, cfg.model_type, device)
        models.append(pure_model)

    print("Relearn")
    res = {}
    for num_stories in cfg.num_stories_to_retrain:
        test_forget_dataset = test_forget_stories
        test_retain_dataset = retain_stories
        initial_forget_loss = eval_model(
            model,
            test_forget_dataset,
            cfg.max_tokens,
        )
        initial_retain_loss = eval_model(
            model,
            test_retain_dataset,
            cfg.max_tokens,
        )
        eval_losses = retrain_model(
            model,
            test_forget_dataset=test_forget_dataset,
            test_retain_dataset=test_retain_dataset,
            train_forget_stories=train_forget_stories,
            num_stories=num_stories,
            all_steps=cfg.retrain_all_steps,
            num_steps=cfg.num_steps,
            words_to_localize=cfg.words_to_localize,
            prompt=cfg.prompt,
            truncate_at=cfg.max_tokens,
            eval_interval=cfg.eval_interval,
        )
        eval_losses["update_step"].insert(0, 0)
        eval_losses["forget"].insert(0, initial_forget_loss)
        eval_losses["retain"].insert(0, initial_retain_loss)
        res[num_stories] = eval_losses
        model_store.load_weights(model, cfg.model_save_path)  # Reload the initial model
    return res


if __name__ == "__main__":
    import projects.tinystories.shared_settings as shared_settings

    """
    $(pdm venv activate) && python projects/tinystories/retraining_evals.py

    $(pdm venv activate) && python projects/tinystories/retraining_evals.py dry_run
    """

    device = get_gpu_with_most_memory()
    tokenizer = transformers.AutoTokenizer.from_pretrained("RonenEldan/TinyStories-28M")

    words_to_localize = shared_settings.cfg.words_to_localize

    dry_run = len(sys.argv) > 1 and sys.argv[1] == "dry_run"

    all_stories = string_utils.load_dataset_with_split(
        "delphi-suite/stories", "validation", max_stories=1_000
    )
    truncated_stories = string_utils.truncate_stories_by_chars(
        all_stories, shared_settings.cfg.truncate_story_chars_at
    )
    forget_stories, retain_stories = string_utils.split_and_label_stories_by_concept(
        truncated_stories, shared_settings.cfg.words_to_localize
    )

    experiment_id = 9
    model_store.print_available_models(
        "bulk_runs_for_paper", f"e{experiment_id}_pure_seed"
    )
    seed = 14933
    print(f"Using seed {seed}")
    cfg = RetrainExperimentConfig(
        words_to_localize=words_to_localize,
        num_stories_to_retrain=[1] if dry_run else [64],
        num_steps=1 if dry_run else 20,
        eval_batch_size=80,
        max_tokens=256,
        pure_model_path=f"bulk_runs_for_paper/e{experiment_id}_pure_seed{seed}",
        base_model_path=f"bulk_runs_for_paper/e{experiment_id}_base_seed{seed}",
        model_save_path=f"bulk_runs_for_paper/e{experiment_id}_ERAC_seed{seed}",
        model_type="roneneldan/TinyStories-28M",
        prompt="Once upon a time, Timmy went to the forest",
        dry_run=dry_run,
    )

    figs, results = run_retrain_evals(
        forget_stories=forget_stories,
        retain_stories=retain_stories,
        cfg=cfg,
        device=device,
    )
    for fig in figs:
        fig.show()

# %%
