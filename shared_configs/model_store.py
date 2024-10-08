# %%
import os
from pathlib import Path
from typing import Union

import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import get_pretrained_model_config

from factored_representations.files import ensure_shared_dir_exists

"""
This file contains paths to models shared across projects.

See [spreadsheet link]
"""

MODEL_STORE_PATH = ensure_shared_dir_exists("models")


def load_model(
    subpath: str, model_name_or_config: Union[str, HookedTransformerConfig], device
):
    """Load a model from the model store.

    Args:
        subpath: e.g. "pretrained/my_model".
        model_name_or_config: either the name of a pretrained model, or a config.
        device: the device to load the model on.
    """
    assert type(model_name_or_config) in [str, HookedTransformerConfig]
    use_pretrained_config = type(model_name_or_config) is str
    if use_pretrained_config:
        model_name = model_name_or_config
        print(f"Loading weights {subpath}.pt -> {model_name} on {device}...", end=" ")
        config = get_pretrained_model_config(model_name, device=device)
    else:
        print(f"Loading weights {subpath}.pt -> model on {device}...", end=" ")
        config = model_name_or_config

    model = HookedTransformer(config)  # type: ignore

    # Load the state dict; ignores missing layer norm keys resulting from
    # TransformerLens folding layer norm into the weights.
    state_dict = t.load(MODEL_STORE_PATH / f"{subpath}.pt", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("done.")
    return model


def save_model(model, subpath):
    """Save a model to the model store."""
    t.save(model.state_dict(), MODEL_STORE_PATH / f"{subpath}.pt")
    print(f"Model saved to {subpath}.pt.")


def load_weights(model: HookedTransformer, subpath):
    device = next(model.parameters()).device
    loaded_params = t.load(MODEL_STORE_PATH / f"{subpath}.pt", map_location=device)
    model.load_state_dict(loaded_params, strict=False)
    return model


def ensure_save_path_exists(subpath, overwrite: bool) -> Path:
    """Ensure that the save path is valid."""
    msg = "Save path must contain a subdirectory, like 'pretrained' or 'finetuned'."
    assert len(subpath.split(os.sep)) > 1, msg

    assert subpath[-3:] != ".pt", "save_path should not include the file extension."
    path = MODEL_STORE_PATH / f"{subpath}.pt"

    if not overwrite:
        assert not os.path.exists(path), f"Model already exists at path: {path}"

    ensure_shared_dir_exists(path.parent)
    return path


def print_available_models(dir, prefix=""):
    print("Available models:")
    for file in os.listdir(MODEL_STORE_PATH / dir):
        if file.startswith(prefix) and file.endswith(".pt"):
            print(file[:-3])
