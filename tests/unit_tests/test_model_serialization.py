import os

import torch as t

import factored_representations.files as files
import shared_configs.model_store as model_store


def test_save_and_load_weights(tiny_transformer_with_batch):
    files.ensure_shared_dir_exists("test")

    model, data = tiny_transformer_with_batch
    with t.inference_mode():
        out_pre = model(data)

    model_store.save_model(model, "test/test_a")

    model.init_weights()
    with t.inference_mode():
        out_init = model(data)
    assert not t.isclose(out_pre, out_init).all()

    model_store.load_weights(model, "test/test_a")
    with t.inference_mode():
        out_post = model(data)
    assert t.isclose(out_pre, out_post).all()


def test_save_and_load_model(tiny_transformer_with_batch):
    files.ensure_shared_dir_exists("test")

    model, data = tiny_transformer_with_batch

    with t.inference_mode():
        out_pre = model(data)

    model_store.save_model(model, "test/test_a")

    loaded = model_store.load_model("test/test_a", model.cfg, device="cpu")

    with t.inference_mode():
        out_post = loaded(data)

    assert t.isclose(out_pre, out_post).all()


def test_load_and_save_pretrain(tinystories_8m):
    model = tinystories_8m

    # Train on fake data
    batch = 2
    seq = 3
    data = t.arange(batch * seq).reshape((batch, seq))

    model.train()
    optimizer = t.optim.Adam(model.parameters(), lr=1)
    preds = model(data)
    loss = (preds - t.zeros_like(preds)) ** 2
    loss.sum().backward()
    optimizer.step()

    with t.inference_mode():
        out_pre = model(data)
    assert not t.isclose(preds, out_pre).all()

    model_store.save_model(model, "test/test_b")

    loaded = model_store.load_model("test/test_b", model.cfg, device="cpu")
    with t.inference_mode():
        out_loaded = loaded(data)

    assert t.isclose(out_pre, out_loaded).all()


def cleanup():
    test_path = files.ensure_shared_dir_exists("test")
    for f in os.listdir(test_path):
        os.remove(os.path.join(test_path, f))
