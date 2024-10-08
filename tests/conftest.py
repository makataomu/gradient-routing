# %%
import pytest
import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import AutoTokenizer

from factored_representations.utils import (
    get_gpu_with_most_memory,
)
from shared_configs.model_configs import qwen_config_tiny

DEVICE = get_gpu_with_most_memory()


@pytest.fixture(scope="session")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def tinystories_8m() -> HookedTransformer:
    model_name = "roneneldan/TinyStories-8M"
    return HookedTransformer.from_pretrained(model_name, device=DEVICE)


@pytest.fixture(scope="session")
def tiny_transformer_with_batch():
    model_cfg = HookedTransformerConfig(
        d_model=10,
        d_head=4,
        n_heads=5,
        n_layers=3,
        n_ctx=16,
        act_fn="relu",
        d_vocab=20,
        device="cpu",
    )
    model = HookedTransformer(model_cfg)
    model.init_weights()

    batch = 2
    seq = 3
    data = t.arange(batch * seq).reshape((batch, seq))

    return model, data


@pytest.fixture
def tiny_qwen_with_batch(scope="session"):
    model = HookedTransformer(qwen_config_tiny)
    model.init_weights()

    batch = 2
    seq = 3
    data = t.arange(batch * seq).reshape((batch, seq))

    return model, data


@pytest.fixture
def very_small_hookedtransformer():
    model_cfg = HookedTransformerConfig(
        d_model=10,  # Don't make this <=2 or LayerNorm will create strange behavior
        d_head=2,
        n_heads=5,
        n_layers=3,
        n_ctx=16,
        act_fn="relu",
        d_vocab=10,
        device="cpu",
        tokenizer_name="gpt2",
    )
    model = HookedTransformer(model_cfg)
    model.init_weights()
    return model
