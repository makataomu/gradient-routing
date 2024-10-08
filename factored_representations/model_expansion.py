# %%
import warnings
from copy import deepcopy
from typing import Callable

import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig

import factored_representations.masklib as masklib

EXPANDABLE_ATTRIBUTES = ["d_head", "n_heads", "d_mlp", "n_key_value_heads", "d_model"]

# These values can be used for setting the learning rates of the expanded/original
# dims in the target layers of an expanded model.
DIMENSIONWISE_LR_DEFAULTS = dict(
    expanded_dim_lr_target=1.0,
    original_dim_lr_target=0.0,
    expanded_dim_lr_off_target=0.0,
    original_dim_lr_off_target=1.0,
)


def expand_model(
    model: HookedTransformer,
    attributes_to_expand: dict[str, int],
    layers_to_initialize: list[int],
    weight_initialization_coeff: float,
    set_n_key_value_heads_automatically: bool = True,
    suppress_warnings: bool = False,
) -> HookedTransformer:
    """
    Expand a model by attention head size, number of attention heads,
    or number of hidden MLP activations.

    Args:
        model: the model to be expanded
        attributes_to_expand: a dict mapping attribute -> num_new_dims
        initialize_new_weights: if False, new model will implement the same
            function as the original.
        set_n_key_value_heads_automatically: if the model
    """
    if "d_head" in attributes_to_expand and not suppress_warnings:
        warning = """
        Expanding d_head changes the attention pattern; in the current implementation,
        this means that the expanded model won't implement the same function as the
        smaller model, even when ablated.
        """
        warnings.warn(warning)

    if "d_model" in attributes_to_expand and not suppress_warnings:
        warning = """
        Expanding d_model changes the residual stream; this means that the expanded
        model won't implement the same function as the smaller model, even when ablated.
        """
        warnings.warn(warning)

    new_cfg = deepcopy(model.cfg)
    for attribute in attributes_to_expand.keys():
        assert attribute in EXPANDABLE_ATTRIBUTES

    for attribute, expansion in attributes_to_expand.items():
        setattr(new_cfg, attribute, getattr(model.cfg, attribute) + expansion)

    # HANDLE GROUPED-QUERY ATTENTION
    if new_cfg.n_key_value_heads is not None:
        if set_n_key_value_heads_automatically:
            assert (
                "n_key_value_heads" not in attributes_to_expand
            ), "Do not pass n_key_value_heads if setting automatically"
            assert model.cfg.n_key_value_heads is not None
            assert (
                (new_cfg.n_heads * model.cfg.n_key_value_heads) % model.cfg.n_heads == 0
            ), "Invalid number of new heads for grouped-query attention."  # type: ignore
            new_cfg.n_key_value_heads = (
                new_cfg.n_heads * model.cfg.n_key_value_heads
            ) // model.cfg.n_heads  # type: ignore

        else:
            error_divide = f"""
            When using grouped-query attention, n_key_value_heads must divide n_heads.
            Your expanded model has n_key_value_heads={new_cfg.n_key_value_heads} and
            n_heads={new_cfg.n_heads}.
            """
            assert new_cfg.n_heads % new_cfg.n_key_value_heads == 0, error_divide

            new_ratio = new_cfg.n_heads / new_cfg.n_key_value_heads
            old_ratio = model.cfg.n_heads / model.cfg.n_key_value_heads  # type: ignore
            warning_ratio = f"""
            When expanding a model with grouped-query attention, you must maintain the
            same ratio of n_heads to n_key_value_heads in order for the larger model to
            implement the same function as the smaller model. Right now you have
            {new_ratio=}, but {old_ratio=}.
            """
            if new_ratio != old_ratio and not suppress_warnings:
                warnings.warn(warning_ratio)

    new_model = HookedTransformer(new_cfg)
    new_model.init_weights()

    for (name, param_old), param_new in zip(
        model.named_parameters(), new_model.parameters()
    ):
        if param_old.shape != param_new.shape:
            if name.startswith("blocks") and "ln" not in name:
                layer = int(name.split(".")[1])
                if layer not in layers_to_initialize:
                    param_new.data = t.zeros_like(param_new)
                else:
                    param_new.data = param_new.data * weight_initialization_coeff

            old_slice = tuple([slice(0, size_old) for size_old in param_old.shape])
            param_new.data[old_slice] = param_old.data
        else:
            param_new.data[:] = param_old.data

    return new_model


def contract_model(
    expanded_model: HookedTransformer,
    original_cfg: HookedTransformerConfig,
) -> HookedTransformer:
    contracted_cfg = deepcopy(original_cfg)
    contracted_model = HookedTransformer(contracted_cfg)

    for (name_e, param_expanded), (name_c, param_contracted) in zip(
        expanded_model.named_parameters(), contracted_model.named_parameters()
    ):
        assert (
            name_e == name_c
        ), f"Layers do not match: {name_e} != {name_c}. Are you using the right config?"
        if param_expanded.shape != param_contracted.shape:
            contracted_slice = tuple(
                [slice(0, size) for size in param_contracted.shape]
            )
            param_contracted.data = param_expanded.data[contracted_slice]
        else:
            param_contracted.data = param_expanded.data
    return contracted_model


def _get_paired_mask_specs_for_expanded_model(
    masking_rule: Callable,
    new_dims: t.Tensor,
    layers_to_mask: list[int],
    num_layers: int,
    hook_type: str,
    expanded_dim_lr_target: float,  # reasonable default: 1.0
    original_dim_lr_target: float,  # reasonable default: 0.0
    expanded_dim_lr_off_target: float,  # reasonable default: 0.0
    original_dim_lr_off_target: float,  # reasonable default: 1.0
) -> list[masklib.MaskSpec]:
    """
    Helper function to get mask specs needed for an expanded model.

    With gradient routing, we normally only localize the "target" data to
    particular dimensions, leaving gradients for "regular" data unmasked.
    However, since we are going to ablate the expanded dimensions, we need
    to make sure regular data stays within the original dimensions.

    Args:
        new_dims: a Tensor of the expanded shape, with 1 indicating a new
            dimension and 0 indicating an original dimension.
    """
    layers_not_masked = [
        layer for layer in range(num_layers) if layer not in layers_to_mask
    ]

    target_data_lrs = (
        new_dims * expanded_dim_lr_target + (~new_dims) * original_dim_lr_target
    )

    off_target_data_lrs = (
        new_dims * expanded_dim_lr_off_target + (~new_dims) * original_dim_lr_off_target
    )

    mask_lookup_target_layers = {
        0: target_data_lrs,  # target data goes to new dims
        1: off_target_data_lrs,  # off-target data goes to original dims
        2: new_dims,  # (for zero-ablation) target data is ablated at new dims
        3: t.ones_like(new_dims),  # (for zero-ablation) let everything through
    }
    target_layers_spec = masklib.MaskSpec(
        masking_rule=masking_rule,
        layers=layers_to_mask,
        hook_type=hook_type,
        mask_lookup=lambda mask_idx: mask_lookup_target_layers[mask_idx],
        num_custom_masks=len(mask_lookup_target_layers),
    )

    mask_lookup_off_target_layers = {
        0: ~new_dims,  # target data goes to original dims
        1: ~new_dims,  # off-target data goes to original dims
        2: new_dims,  # (for zero-ablation) target data is ablated at new dims
        3: t.ones_like(new_dims),  # (for zero-ablation) let everything through
    }

    off_target_layers_spec = masklib.MaskSpec(
        masking_rule=masking_rule,
        layers=layers_not_masked,
        hook_type=hook_type,
        mask_lookup=lambda mask_idx: mask_lookup_off_target_layers[mask_idx],
        num_custom_masks=len(mask_lookup_off_target_layers),
    )

    return [target_layers_spec, off_target_layers_spec]


def expand_and_get_mask_specs(
    model: HookedTransformer,
    attributes_to_expand: dict[str, int],
    layers_to_mask: list[int],
    masking_rule: Callable,
    expanded_dim_lr_target: float,
    original_dim_lr_target: float,
    expanded_dim_lr_off_target: float,
    original_dim_lr_off_target: float,
    weight_initialization_coeff: float,
    suppress_warnings: bool = False,
):
    """
    Expand a model according to attributes_to_expand and return a tuple with the
    expanded model and corresponding MaskSpecs.

    NOTE: expects masking_rule to return 0 on the data being localized (forget),
    and 1 on the data not being localized (retain).

    Args:
        expanded_dim_lr_target - the learning rate for the *target* (forget) data
            in the *expanded* dimensions (previously this was always set to 1.0).
        original_dim_lr_target - the learning rate for the *target* (forget) data in the
            *original* dimensions (previously this was always set to 0.0).
        expanded_dim_lr_off_target - the learning rate for the *off-target* (retain)
            data in the *expanded* dimensions (previously this was always set to 0.0).
        original_dim_lr_off_target - the learning rate for the *off-target* (retain)
            data in the *original* dimensions (previously this was always set to 1.0).
    """
    new_model = expand_model(
        model,
        attributes_to_expand,
        weight_initialization_coeff=weight_initialization_coeff,
        layers_to_initialize=layers_to_mask,
        suppress_warnings=suppress_warnings,
    )

    mask_spec_getter_kwargs = dict(
        expanded_dim_lr_target=expanded_dim_lr_target,
        original_dim_lr_target=original_dim_lr_target,
        expanded_dim_lr_off_target=expanded_dim_lr_off_target,
        original_dim_lr_off_target=original_dim_lr_off_target,
    )

    specs = get_expanded_mask_specs(
        model,
        attributes_to_expand,
        layers_to_mask,
        masking_rule,
        new_model,
        mask_spec_getter_kwargs,
    )

    return new_model, specs


def extend_expanded_layers(
    model,
    mask_applier: masklib.MaskApplier,
    layers_to_add: list[int],
    weight_initialization_coeff: float,
):
    """
    Extend the set of expanded layers of an already-expanded model by (i) initializing
    the expanded parameters of the newly-added layers, and (ii) adding the new layers
    to all mask specs.

    Return:
        (new_model, new_mask_applier)
    """
    assert model is mask_applier.model

    new_model = HookedTransformer(model.cfg)
    new_model.init_weights()

    for (name, param_old), param_new in zip(
        model.named_parameters(), new_model.parameters()
    ):
        if name.startswith("blocks") and "ln" not in name:
            layer = int(name.split(".")[1])
            if layer in layers_to_add:
                is_expanded = t.isclose(param_old, t.tensor(0.0))
                param_new.data[:] = (
                    is_expanded * weight_initialization_coeff * param_new.data[:]
                    + ~is_expanded * param_old.data
                )
            else:
                param_new.data[:] = param_old.data
        else:
            param_new.data[:] = param_old.data

    new_mask_specs = [
        mask_spec._replace(layers=mask_spec.layers + layers_to_add)
        for mask_spec in mask_applier.mask_specs
    ]

    new_mask_applier = masklib.MaskApplier(
        new_model,
        new_mask_specs,
        use_partial_boolean_masks=mask_applier.use_partial_boolean_masks,
    )  # NOTE: if mask_applier._precompute_masks() becomes slow, this will be slow

    return new_model, new_mask_applier


def get_hook_name_from_model_part(model_part: str) -> str:
    if model_part == "n_heads" or model_part == "d_head":
        return "attn.hook_z"
    elif model_part == "d_mlp":
        return "mlp.hook_post"
    elif model_part == "d_model":
        return "hook_resid_post"
    else:
        raise ValueError(f"Unknown model part to get hook name for: {model_part}")


def get_expanded_mask_specs(
    model,
    attributes_to_expand,
    layers_to_mask,
    masking_rule,
    new_model,
    mask_spec_getter_kwargs,
):
    specs = []
    if set(("d_head", "n_heads")).intersection(attributes_to_expand):
        new_attn_dims = t.ones(
            (new_model.cfg.n_heads, new_model.cfg.d_head), dtype=t.bool
        )
        new_attn_dims[: model.cfg.n_heads, : model.cfg.d_head] = 0

        attn_specs = _get_paired_mask_specs_for_expanded_model(
            masking_rule,
            new_attn_dims,
            layers_to_mask,
            num_layers=model.cfg.n_layers,
            hook_type=get_hook_name_from_model_part("n_heads"),
            **mask_spec_getter_kwargs,
        )

        specs = specs + attn_specs

    if "d_mlp" in attributes_to_expand:
        new_mlp_dims = t.ones((new_model.cfg.d_mlp,), dtype=t.bool)
        new_mlp_dims[: model.cfg.d_mlp] = 0

        mlp_specs = _get_paired_mask_specs_for_expanded_model(
            masking_rule,
            new_mlp_dims,
            layers_to_mask,
            num_layers=model.cfg.n_layers,
            hook_type=get_hook_name_from_model_part("d_mlp"),
            **mask_spec_getter_kwargs,
        )
        specs = specs + mlp_specs

    if "d_model" in attributes_to_expand:
        new_res_dims = t.ones((new_model.cfg.d_model,), dtype=t.bool)
        new_res_dims[: model.cfg.d_model] = 0

        residual_specs = _get_paired_mask_specs_for_expanded_model(
            masking_rule,
            new_res_dims,
            layers_to_mask,
            num_layers=model.cfg.n_layers,
            hook_type=get_hook_name_from_model_part("d_model"),
            **mask_spec_getter_kwargs,
        )
        specs = specs + residual_specs
    return specs


if __name__ == "__main__":
    # from shared_configs.model_configs import qwen_config_tiny

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

    batch = 1
    seq = 3
    data = t.arange(batch * seq).reshape((batch, seq))

    with t.inference_mode():
        out = model(data)

    new_model = expand_model(
        model,
        {"n_heads": 12, "d_mlp": 10, "d_model": 3},
        layers_to_initialize=list(range(model.cfg.n_layers)),
        weight_initialization_coeff=1.0,
    )

    contracted_model = contract_model(new_model, model.cfg)  # type: ignore

    with t.inference_mode():
        out = model(data)
        out_c = contracted_model(data)

    assert t.allclose(out, out_c, atol=1e-5)
