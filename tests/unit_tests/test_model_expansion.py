# %%
import torch as t

import factored_representations.masklib as masklib
import factored_representations.model_expansion as model_expansion


def test_expansion_with_zero_weights(tiny_transformer_with_batch):
    model, data = tiny_transformer_with_batch

    new_model = model_expansion.expand_model(
        model,
        {"n_heads": 10, "d_mlp": 2},
        layers_to_initialize=[],
        weight_initialization_coeff=1.0,
    )

    with t.inference_mode():
        out = model(data)
        new_out = new_model(data)

    assert t.allclose(out, new_out, atol=1e-5)


def test_expand_and_ablate(tiny_transformer_with_batch):
    model, data = tiny_transformer_with_batch

    new_model, specs = model_expansion.expand_and_get_mask_specs(
        model,
        {"n_heads": 10, "d_mlp": 20},  # expanding d_head messes with attention patterns
        layers_to_mask=list(range(model.cfg.n_layers)),
        masking_rule=lambda data: t.ones_like(data),
        **model_expansion.DIMENSIONWISE_LR_DEFAULTS,
        weight_initialization_coeff=1.0,
        suppress_warnings=False,
    )

    applier = masklib.MaskApplier(new_model, specs)

    with t.inference_mode():
        out = model(data)
        with applier.zero_ablate(mask_idx=2):
            new_out = new_model(data)

    assert t.allclose(out, new_out, atol=1e-5)


def test_contraction(tiny_transformer_with_batch):
    model, data = tiny_transformer_with_batch

    expanded_model = model_expansion.expand_model(
        model,
        {"n_heads": 10, "d_mlp": 10},
        layers_to_initialize=list(range(model.cfg.n_layers)),
        weight_initialization_coeff=1.0,
    )

    contracted_model = model_expansion.contract_model(expanded_model, model.cfg)

    with t.inference_mode():
        original_out = model(data)
        contracted_out = contracted_model(data)

    assert t.allclose(original_out, contracted_out, atol=1e-5)


def test_qwen_mlp_expansion_with_zero_weights(tiny_qwen_with_batch):
    model, data = tiny_qwen_with_batch

    new_model = model_expansion.expand_model(
        model,
        {"d_mlp": 10},
        layers_to_initialize=[],
        weight_initialization_coeff=1.0,
    )

    with t.inference_mode():
        out = model(data)
        new_out = new_model(data)

    assert t.allclose(out, new_out, atol=1e-5)


def test_qwen_attn_expansion_with_zero_weights(tiny_qwen_with_batch):
    model, data = tiny_qwen_with_batch

    new_model = model_expansion.expand_model(
        model,
        {"n_heads": model.cfg.n_heads * 2},  # triple the number of heads
        layers_to_initialize=[],
        weight_initialization_coeff=1.0,
    )

    with t.inference_mode():
        out = model(data)
        new_out = new_model(data)

    assert t.allclose(out, new_out, atol=1e-5)


def test_qwen_attn_expand_and_ablate(tiny_qwen_with_batch):
    model, data = tiny_qwen_with_batch

    new_model, specs = model_expansion.expand_and_get_mask_specs(
        model,
        {
            "n_heads": model.cfg.n_heads * 2,
            "d_mlp": 20,
        },  # expanding d_head messes with attention patterns
        layers_to_mask=list(range(model.cfg.n_layers)),
        masking_rule=lambda data: t.ones_like(data),
        **model_expansion.DIMENSIONWISE_LR_DEFAULTS,
        weight_initialization_coeff=1.0,
        suppress_warnings=False,
    )

    applier = masklib.MaskApplier(new_model, specs)

    with t.inference_mode():
        out = model(data)
        with applier.zero_ablate(mask_idx=2):
            new_out = new_model(data)

    assert t.allclose(out, new_out, atol=1e-5)


def test_expand_contract_d_model(tiny_qwen_with_batch):
    model, data = tiny_qwen_with_batch

    new_model, specs = model_expansion.expand_and_get_mask_specs(
        model,
        {"d_model": 20},
        layers_to_mask=list(range(model.cfg.n_layers)),
        masking_rule=lambda data: t.ones_like(data),
        suppress_warnings=True,
        **model_expansion.DIMENSIONWISE_LR_DEFAULTS,
        weight_initialization_coeff=1.0,
    )

    contracted_model = model_expansion.contract_model(new_model, model.cfg)

    applier = masklib.MaskApplier(new_model, specs)

    with t.inference_mode():
        out = model(data)
        with applier.zero_ablate(mask_idx=2):
            contracted_out = contracted_model(data)

    assert t.allclose(out, contracted_out, atol=1e-5)


def test_expand_model_off_target_layer_initialization(tiny_transformer_with_batch):
    model, data = tiny_transformer_with_batch
    test_block = 2

    # Setting everything to zero makes it easier to write the test
    for block_idx in range(model.cfg.n_layers):
        for param in model.blocks[block_idx].mlp.parameters():
            param.data.fill_(0)

    new_model = model_expansion.expand_model(
        model,
        {"d_mlp": 10},
        layers_to_initialize=[test_block],
        weight_initialization_coeff=1.0,
    )

    with t.inference_mode():
        out = model(data)
        new_model_out = new_model(data)

    msg = "Expanded model has nonzero initialized weights; output should be different."
    assert not t.allclose(out, new_model_out, atol=1e-5), msg

    # Zero out the initialized weights
    for param in new_model.blocks[test_block].mlp.parameters():
        param.data.fill_(0)

    with t.inference_mode():
        new_model_out = new_model(data)

    msg = "Expanded model has zeros at expanded layer; output should be the same."
    assert t.allclose(out, new_model_out, atol=1e-5)


def test_expanded_weights_copied(tiny_transformer_with_batch):
    model, data = tiny_transformer_with_batch

    new_model = model_expansion.expand_model(
        model,
        {"d_mlp": 10},
        layers_to_initialize=[],
        weight_initialization_coeff=1.0,
    )

    with t.inference_mode():
        out = model(data)
        new_model_out = new_model(data)

    assert t.allclose(out, new_model_out, atol=1e-5)

    for parameter in model.parameters():
        parameter.data[:] = t.ones_like(parameter.data)

    with t.inference_mode():
        out = model(data)
        new_model_out = new_model(data)

    assert not t.allclose(out, new_model_out, atol=1e-5)


def test_layer_extension(tiny_qwen_with_batch):
    original_model, data = tiny_qwen_with_batch

    to_expand = {"n_heads": original_model.cfg.n_heads * 2, "d_mlp": 20}
    expanded_model, specs = model_expansion.expand_and_get_mask_specs(
        original_model,
        to_expand,
        layers_to_mask=[],
        masking_rule=lambda data: t.zeros_like(data),
        **model_expansion.DIMENSIONWISE_LR_DEFAULTS,
        weight_initialization_coeff=1.0,
        suppress_warnings=False,
    )

    applier = masklib.MaskApplier(expanded_model, specs)

    with t.inference_mode():
        out = original_model(data)
        with applier(data):
            expanded_out = expanded_model(data)

    assert t.allclose(
        out, expanded_out, atol=1e-5
    ), "Expanded model should be different"

    layers_to_add = [1, 2]
    extended_model, extended_applier = model_expansion.extend_expanded_layers(
        expanded_model,
        applier,
        layers_to_add=layers_to_add,
        weight_initialization_coeff=1,
    )

    for (name, old_param), new_param in zip(
        expanded_model.named_parameters(), extended_model.parameters()
    ):
        if name.startswith("blocks") and "ln" not in name:
            layer = int(name.split(".")[1])
            is_bias = "b_" in name.split(".")[-1]
            if layer in layers_to_add and not is_bias:
                assert old_param.data is not new_param.data
                assert not t.allclose(
                    old_param,
                    new_param,
                ), f"New layer {layer} should have different weights due to initializatoin"
            else:
                assert t.allclose(
                    old_param,
                    new_param,
                ), f"Old layer {layer} should have the same weights"

    with t.inference_mode():
        extended_out = extended_model(data)
        with extended_applier.zero_ablate(mask_idx=2):
            extended_out_ablated = extended_model(data)

    assert not t.allclose(
        expanded_out, extended_out
    ), "Extended model should be different due to initialization of new layers"
    assert t.allclose(
        expanded_out, extended_out_ablated
    ), "Ablated model should be the same as the expanded model"
