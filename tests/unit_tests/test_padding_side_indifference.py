# %%
from copy import deepcopy

import torch as t
import torch.utils.data as data

import factored_representations.string_utils as string_utils
import factored_representations.training as training


def params_allclose(model1, model2, atol=1e-3):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not t.allclose(p1, p2, atol=atol):
            return False
    return True


def test_padding_side_indifference(tinystories_8m):
    model1 = deepcopy(tinystories_8m)
    model2 = deepcopy(tinystories_8m)
    assert params_allclose(model1, model2), "Copied models should be the same"

    all_stories = string_utils.load_dataset_with_split(
        "delphi-suite/stories",
        "train",
        max_stories=2,
    )

    truncated_stories = string_utils.truncate_stories_by_chars(
        all_stories, max_character_len=25
    )
    dataloader = data.DataLoader(
        string_utils.ListDataset(truncated_stories),
        batch_size=2,
        shuffle=False,
    )

    for model, pad_side in [(model1, "right"), (model2, "left")]:
        optim = t.optim.AdamW(
            model.parameters(),
            lr=1,
        )
        for batch in dataloader:
            prepend_bos = False
            input_ids, attention_mask = string_utils.tokenize_batch(
                batch,
                model.tokenizer,
                prepend_bos=prepend_bos,
                truncate_at=256,
                padding_side=pad_side,  # type: ignore
                device=model.cfg.device,
            )
            loss = training.compute_preds_and_get_ce_loss(
                model, input_ids, attention_mask, None
            )
            loss.backward()
            optim.step()
            optim.zero_grad()

    assert not params_allclose(model1, tinystories_8m), "Training should change model1"
    assert params_allclose(model1, model2), "Training should not depend on padding side"


# model = tinystories_8m()
# test_padding_side_indifference(model)
