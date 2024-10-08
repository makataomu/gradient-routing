# %%
import torch as t
import transformers
from jaxtyping import Integer

import factored_representations.masklib as masklib


def test_masking_token_freq_weights():
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    text_to_tokenize = "blah Harry Potter a b c Severus Snape"
    token_sequence_to_mask = tokenizer(text_to_tokenize, return_tensors="pt").input_ids

    num_tokens = len(token_sequence_to_mask[0])

    # Masking config
    token_freq_masking_rule, _ = masklib.get_token_freq_masking_rule(
        retain_stories=[100 * " Harry"],
        forget_stories=[100 * " Snape"],
        num_stories=1,
        truncate_at=None,
        num_synthetic_tokens_retain=0.001,
        num_synthetic_tokens_forget=0.001,
        scale=1,
        bias=0,
        tokenizer=tokenizer,
        device=t.device("cpu"),
    )

    mask_weights = token_freq_masking_rule(token_sequence_to_mask)
    assert mask_weights.shape == (1, num_tokens - 1)

    expected = t.tensor(
        [[0.5000, 1.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.0000]]
    )
    assert t.allclose(mask_weights, expected)


def test_masking_resid_stream_concept_level():
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    text_to_tokenize = "blah Harry Potter a b c Severus Snape"
    token_sequence_to_mask = tokenizer(text_to_tokenize, return_tensors="pt").input_ids

    resid_stream_mask_idx_to_dims = {0: [0, 1], 1: [2]}
    d_masked = 10  # made-up value
    num_tokens = len(token_sequence_to_mask[0])

    # Masking config
    num_custom_masks = len(resid_stream_mask_idx_to_dims.values())
    mask_lookup = masklib.get_resid_mask_lookup_from_dict(
        resid_stream_mask_idx_to_dims, d_masked
    )
    mask_lookup_tensor = masklib.precompute_mask(
        mask_lookup, num_custom_masks=num_custom_masks
    )
    concept_masking_rule = masklib.get_concept_masking_rule(
        [([" harry potter"], 0), ([" severus snape"], 1)], tokenizer
    )

    # Apply masks
    masked_indexes = concept_masking_rule(token_sequence_to_mask)
    masked = mask_lookup_tensor[masked_indexes]
    assert masked.shape == (1, num_tokens - 1, d_masked)
    # fmt: off
    assert (
        masked
        == t.tensor(
            [[[True, True, True, True, True, True, True, True, True, True], # -> ah
              [True, True, False, False, False, False, False, False, False, False], # -> Harry
              [True, True, False, False, False, False, False, False, False, False], # -> Potter
              [True, True, True, True, True, True, True, True, True, True], # -> a
              [True, True, True, True, True, True, True, True, True, True], # -> b
              [True, True, True, True, True, True, True, True, True, True], # -> c
              [False, False, True, False, False, False, False, False, False, False], # -> severus
              [False, False, True, False, False, False, False, False, False, False]]] # -> snape
        )
    ).all()
    # fmt: on


def test_masking_resid_stream_full_seq_level():
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    text_to_tokenize = ["blah Harry Potter a b c", "d e f g"]
    token_sequence_to_mask = tokenizer(
        text_to_tokenize, padding=True, return_tensors="pt"
    ).input_ids

    resid_stream_mask_idx_to_dims = {0: [0, 1]}
    d_masked = 7  # made-up value
    num_tokens = len(token_sequence_to_mask[0])

    # Masking config
    num_custom_masks = len(resid_stream_mask_idx_to_dims.values())
    mask_lookup = masklib.get_resid_mask_lookup_from_dict(
        resid_stream_mask_idx_to_dims, d_masked
    )
    mask_lookup_tensor = masklib.precompute_mask(
        mask_lookup, num_custom_masks=num_custom_masks
    )
    seq_level_masking_rule = masklib.get_full_sequence_concept_masking_rule(
        [" Harry Potter"], tokenizer
    )

    # Apply masks
    masked_indexes = seq_level_masking_rule(token_sequence_to_mask)
    masked = mask_lookup_tensor[masked_indexes]
    assert masked.shape == (2, num_tokens - 1, d_masked)
    # fmt: off

    assert (
        masked == t.tensor([
        [[1., 1., 0., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0., 0., 0.]],

        [[1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.]]])).all()
    # fmt: on


def test_masking_resid_stream_token_level():
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    text_to_tokenize = "hello a something b"
    token_sequence_to_mask = tokenizer(text_to_tokenize, return_tensors="pt").input_ids

    resid_stream_mask_idx_to_dims = {0: [0, 1], 1: [2]}
    d_masked = 10  # made-up value
    num_tokens = len(token_sequence_to_mask[0])

    # Masking config
    num_custom_masks = len(resid_stream_mask_idx_to_dims.values())
    mask_lookup = masklib.get_resid_mask_lookup_from_dict(
        resid_stream_mask_idx_to_dims, d_masked
    )
    mask_lookup_tensor = masklib.precompute_mask(mask_lookup, num_custom_masks)
    token_masking_rule = masklib.get_token_masking_rule({" a": 0, " b": 1}, tokenizer)

    # Apply masks
    masked_indexes = token_masking_rule(token_sequence_to_mask)
    masked = mask_lookup_tensor[masked_indexes]

    assert masked.shape == (1, num_tokens - 1, d_masked)
    assert (
        masked
        == t.tensor(
            [
                [
                    [
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [True, True, True, True, True, True, True, True, True, True],
                    [
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                ]
            ]
        )
    ).all()


def test_multiple_mask_specs(very_small_hookedtransformer):
    model = very_small_hookedtransformer
    d_model = model.cfg.d_model
    num_custom_masks = 2
    batch_size = 2
    seq_len = 5

    def send_all_to_mask_0(batch: Integer[t.Tensor, "batch seq"]):
        return t.zeros_like(batch)

    def send_all_to_mask_1(batch: Integer[t.Tensor, "batch seq"]):
        return t.ones_like(batch)

    mask_spec_a = masklib.MaskSpec(
        masking_rule=send_all_to_mask_0,
        layers=[1],
        hook_type="hook_mlp_out",
        mask_lookup=lambda mask_idx: masklib.multi_hot([mask_idx], d_model),
        num_custom_masks=num_custom_masks,
    )

    mask_spec_b = masklib.MaskSpec(
        masking_rule=send_all_to_mask_1,
        layers=[2],
        hook_type="hook_resid_post",
        mask_lookup=lambda mask_idx: masklib.multi_hot([mask_idx], d_model),
        num_custom_masks=num_custom_masks,
    )

    mask_applier = masklib.MaskApplier(model, [mask_spec_a, mask_spec_b])
    batch = t.arange(batch_size * seq_len).reshape((batch_size, seq_len))

    with mask_applier(batch):
        out = model(batch)
        label = t.ones_like(out)
        loss = (out - label).abs().sum()
        loss.backward()

    mlp_final_bias_grad = [
        list(model.blocks[layer].mlp.parameters())[-1].grad
        for layer in range(model.cfg.n_layers)
    ]
    for b in mlp_final_bias_grad:
        print(b)
    assert (
        mlp_final_bias_grad[0] != 0
    ).all(), "Unmasked layer should have no zero grads"
    assert (
        (mlp_final_bias_grad[1] != 0) == mask_spec_a.mask_lookup(0)
    ).all(), "Masked layer zero grads should match mask"
    assert (
        (mlp_final_bias_grad[2] != 0) == mask_spec_b.mask_lookup(1)
    ).all(), "Masked layer zero grads should match mask"


def test_attn_mask(very_small_hookedtransformer):
    model = very_small_hookedtransformer

    num_custom_masks = 1
    batch_size = 2
    seq_len = 3

    mask_lookup = masklib.get_attn_mask_lookup_from_dict(
        {0: [2, 4]}, model.cfg.n_heads, model.cfg.d_head
    )
    mask_spec = masklib.MaskSpec(
        masking_rule=lambda toks: t.zeros_like(toks),  # mask everything >:-)
        layers=[1],
        hook_type="attn.hook_z",
        mask_lookup=mask_lookup,
        num_custom_masks=num_custom_masks,
    )

    mask_applier = masklib.MaskApplier(model, [mask_spec])
    batch = t.arange(batch_size * seq_len).reshape((batch_size, seq_len))

    with mask_applier(batch):
        out = model(batch)
        label = t.ones_like(out)
        loss = (out - label).abs().sum()
        loss.backward()

    head_grads = [
        t.sum(model.blocks[layer].attn.W_V.grad.abs(), dim=(1, 2))
        for layer in range(model.cfg.n_layers)
    ]

    assert (
        head_grads[0] != 0
    ).all(), "Unmasked attention heads should have no zero grads"
    assert (
        (head_grads[1] != 0) == t.tensor([False, False, True, False, True])
    ).all(), "Masked attention heads should have zero grads"
    assert (
        head_grads[2] != 0
    ).all(), "Unmasked attention heads should have no zero grads"


def test_device(very_small_hookedtransformer):
    model = very_small_hookedtransformer.to("cuda")
    num_custom_masks = 1
    batch_size = 2
    seq_len = 3

    mask_lookup = masklib.get_resid_mask_lookup_from_dict(
        {0: [2, 4]}, model.cfg.d_model
    )
    mask_spec = masklib.MaskSpec(
        masking_rule=lambda toks: t.zeros_like(toks),
        layers=[1],
        hook_type="hook_resid_post",
        mask_lookup=mask_lookup,
        num_custom_masks=num_custom_masks,
    )

    mask_applier = masklib.MaskApplier(model, [mask_spec])
    batch = t.arange(batch_size * seq_len).reshape((batch_size, seq_len))
    with mask_applier(batch):
        model(batch)


def test_zero_ablation(very_small_hookedtransformer):
    model = very_small_hookedtransformer

    num_custom_masks = 1
    batch_size = 2
    seq_len = 3

    mask_lookup = masklib.get_attn_mask_lookup_from_dict(
        {0: [2, 4]}, model.cfg.n_heads, model.cfg.d_head
    )
    mask_spec = masklib.MaskSpec(
        masking_rule=lambda toks: t.zeros_like(toks),  # mask everything >:-)
        layers=[1],
        hook_type="attn.hook_z",
        mask_lookup=mask_lookup,
        num_custom_masks=num_custom_masks,
    )

    mask_applier = masklib.MaskApplier(model, [mask_spec])
    batch = t.arange(batch_size * seq_len).reshape((batch_size, seq_len))

    with mask_applier.zero_ablate(mask_idx=0):
        out, cache = model.run_with_cache(batch)
        print(cache)

    head_activations = [
        t.sum(cache[f"blocks.{layer}.attn.hook_z"].abs(), dim=(0, 1, 3))
        for layer in range(model.cfg.n_layers)
    ]

    for layer in range(model.cfg.n_layers):
        if layer in mask_spec.layers:
            assert (
                (head_activations[layer] == 0)
                == t.tensor([False, False, True, False, True])
            ).all()
        else:
            assert not (head_activations[layer] == 0).any()


def test_mlp_masking(very_small_hookedtransformer):
    model = very_small_hookedtransformer
    num_custom_masks = 1
    batch_size = 2
    seq_len = 3

    mlp_masking_dims = [0, 1, 2, 3]
    mlp_not_masking_dims = list(range(4, model.cfg.d_mlp))
    mask_lookup = masklib.get_mlp_mask_lookup_from_dict(
        {0: [0, 1, 2, 3]}, model.cfg.d_mlp
    )
    layers_to_mask_at = [1]
    mask_spec = masklib.MaskSpec(
        masking_rule=lambda toks: t.zeros_like(toks),  # mask everything >:-)
        layers=layers_to_mask_at,
        hook_type="mlp.hook_post",
        mask_lookup=mask_lookup,
        num_custom_masks=num_custom_masks,
    )

    mask_applier = masklib.MaskApplier(model, [mask_spec])
    batch = t.arange(batch_size * seq_len).reshape((batch_size, seq_len))

    with mask_applier(batch):
        out = model(batch)
    label = t.ones_like(out)
    loss = (out - label).abs().sum()
    loss.backward()

    def get_mlp_params_grads(block):
        params = list(block.mlp.parameters())
        W_e, b_e, _, _ = params
        return W_e.grad, b_e.grad

    mlp_weights_grad = [
        get_mlp_params_grads(model.blocks[layer]) for layer in range(model.cfg.n_layers)
    ]
    for layer, grads in enumerate(mlp_weights_grad):
        W_e_grad, b_e_grad = grads
        if layer in layers_to_mask_at:
            assert W_e_grad[:, mlp_not_masking_dims].abs().sum() == 0
            assert W_e_grad[:, mlp_masking_dims].abs().sum() != 0
            assert b_e_grad[mlp_not_masking_dims].abs().sum() == 0
            assert b_e_grad[mlp_masking_dims].abs().sum() != 0
        else:
            assert W_e_grad.abs().sum() != 0
            assert b_e_grad.abs().sum() != 0


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    text_to_tokenize = ["blah Harry Potter a b c", "d e f g"]
    token_sequence_to_mask = tokenizer(
        text_to_tokenize, padding=True, return_tensors="pt"
    ).input_ids

    resid_stream_mask_idx_to_dims = {0: [0, 1]}
    d_masked = 7  # made-up value
    num_tokens = len(token_sequence_to_mask[0])

    # Masking config
    num_custom_masks = len(resid_stream_mask_idx_to_dims.values())
    mask_lookup = masklib.get_resid_mask_lookup_from_dict(
        resid_stream_mask_idx_to_dims, d_masked
    )
    mask_lookup_tensor = masklib.precompute_mask(
        mask_lookup, num_custom_masks=num_custom_masks
    )

    full_seq_mask_rule = masklib.get_full_sequence_concept_masking_rule(
        [" Harry Potter"],
        tokenizer,
    )

    concept_masking_rule = masklib.get_concept_masking_rule(
        [([" Harry Potter"], 0)],
        tokenizer,
    )

    def mask_rule(toks: t.Tensor) -> t.Tensor:
        """
        If data point is in the retain set (1), route fully to retain.
        If data point is in the forget set (0), use the token freq masking rule.
        """
        is_retain = full_seq_mask_rule(toks)
        retain_wt = concept_masking_rule(toks)
        return t.maximum(retain_wt, is_retain)

    mask = mask_rule(token_sequence_to_mask)
    print("Mask:")
    print(mask)
