# %%
import warnings
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, List, NamedTuple, Tuple, Union

import einops
import torch as t
import torch.nn.functional as F
import tqdm
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint

SUPPORTED_HOOK_TYPES = [
    "hook_resid_post",
    "mlp.hook_post",
    "hook_mlp_out",
    "attn.hook_z",
]
# attn.hook_z is attention
# https://github.com/TransformerLensOrg/TransformerLens/blob/bf64ede92220166471ff259fa6a8193297253dea/transformer_lens/components/abstract_attention.py#L428

"""
# MASKLIB SAMPLE APPLICATION - for data-dependent gradient routing

    rule = get_concept_mask_builder(...)
    spec_resid = MaskSpec(rule, [3, 4, 5], "hook_resid_post",...)
    spec_attn = MaskSpec(rule, [3, 4, 5], "attn.hook_z", ...)
    applier = MaskApplier(model, [spec_resid, spec_attn], device)

    for batch in batches:
        with applier(batch):
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            optimizer.step()
"""


class MaskSpec(NamedTuple):
    """
    Args:
        masking_rule: maps arbitrary data input to (batch, seq) mask indexes
        layers: which layers to mask at
        hook_type: which TransformerLens hook type to use (e.g. 'hook_resid_post')
        mask_lookup: maps mask index to Tensor mask (a Float in full generality), excluding "no mask"
        num_custom_masks: the number of unique mask indexes (arguments to mask_lookup)
    """

    masking_rule: Callable[[Any], Float[t.Tensor, "batch seq"]]
    layers: List[int]
    hook_type: str
    mask_lookup: Callable[[int], Float[t.Tensor, "d_masked"]]
    num_custom_masks: int


def get_hook_name(layer: int, hook_type: str):
    assert (
        hook_type in SUPPORTED_HOOK_TYPES
    ), f"Hook type {hook_type} not in {SUPPORTED_HOOK_TYPES=}."
    return f"blocks.{layer}.{hook_type}"


def precompute_mask(
    mask_lookup: Callable[[int], Float[t.Tensor, "d_masked"]], num_custom_masks: int
):
    d_masked = mask_lookup(0).shape
    mask_lookup_tensor = t.ones((num_custom_masks + 1, *d_masked))
    for mask_idx in range(num_custom_masks):
        try:
            mask_lookup_tensor[mask_idx] = mask_lookup(mask_idx)
        except Exception:
            msg = f"Error when calling mask_lookup({mask_idx}). Are you sure you want {num_custom_masks=}?"
            raise RuntimeError(msg)
    return mask_lookup_tensor


def _broadcast_right(tensor: t.Tensor, other_tensor: t.Tensor):
    ndims, ndims_broadcasted = len(tensor.shape), len(other_tensor.shape)
    assert ndims <= ndims_broadcasted, f"Expected {ndims=} <= {ndims_broadcasted=}"
    assert (
        tensor.shape == other_tensor.shape[:ndims]
    ), f"Expected {tensor.shape=} == {other_tensor.shape[:ndims]=}"
    return tensor.reshape(tensor.shape + (1,) * (ndims_broadcasted - ndims))


class MaskApplier:
    def __init__(
        self,
        model: t.nn.Module,
        mask_specs: List[MaskSpec],
        use_partial_boolean_masks: bool = False,
    ):
        self.model = model
        self.mask_specs = mask_specs
        self.use_partial_boolean_masks = use_partial_boolean_masks
        self.device = next(model.parameters()).device

        self.mask_lookup_tensors = self._precompute_masks(mask_specs, self.device)

    @staticmethod
    def _precompute_masks(mask_specs, device):
        mask_lookup_tensors = []
        for spec in mask_specs:
            mask_lookup_tensor = precompute_mask(
                spec.mask_lookup, spec.num_custom_masks
            )
            mask_lookup_tensors.append(mask_lookup_tensor.to(device))
        return mask_lookup_tensors

    @staticmethod
    def _create_masking_hook(batch_masks, mask_weight: float):
        # Need this so that the value of batch_masks is bound to masking_hook
        # when we create it.
        def masking_hook(value: t.Tensor, hook: HookPoint):
            masked = (
                mask_weight * batch_masks * value
                + (1 - mask_weight * batch_masks) * value.detach()
            )
            return masked

        return masking_hook

    def _get_masking_hooks(self, batch, mask_weight: float):
        # Create all hooks needed for this batch of data.
        hooks = {}
        for spec, mask_lookup_tensor in zip(self.mask_specs, self.mask_lookup_tensors):
            if self.use_partial_boolean_masks:
                mask_weights = spec.masking_rule(batch).to(self.device)
                # Usually, mask_0 : forget set, mask_1 : retain set
                mask_0 = mask_lookup_tensor[t.zeros_like(mask_weights, dtype=t.long)]
                mask_1 = mask_lookup_tensor[t.ones_like(mask_weights, dtype=t.long)]
                mask_weights = _broadcast_right(mask_weights, mask_0)
                batch_masks = mask_0 * (1 - mask_weights) + mask_1 * mask_weights

            else:
                # TODO (optimization): detect identical masking_rules and only call once
                batch_mask_indexes = spec.masking_rule(batch).to(self.device)
                batch_masks = mask_lookup_tensor[batch_mask_indexes]

            for layer_idx in spec.layers:
                hook_name = get_hook_name(layer_idx, spec.hook_type)
                hook = MaskApplier._create_masking_hook(
                    batch_masks, mask_weight=mask_weight
                )
                hooks[hook_name] = hook
        return hooks

    @staticmethod
    def _create_zero_ablation_hook(mask):
        def masking_hook(value: t.Tensor, hook: HookPoint):
            masked = (1 - mask[None, None, :]) * value
            return masked

        return masking_hook

    def _get_zero_ablation_hooks(self, mask_idx):
        hooks = {}
        for spec, mask_lookup_tensor in zip(self.mask_specs, self.mask_lookup_tensors):
            mask = mask_lookup_tensor[mask_idx]
            for layer_idx in spec.layers:
                hook_name = get_hook_name(layer_idx, spec.hook_type)
                hook = MaskApplier._create_zero_ablation_hook(mask)
                hooks[hook_name] = hook
        return hooks

    @staticmethod
    def _create_masking_and_zero_ablation_hook(
        batch_masks, mask_weight: float, zero_ablate_mask: t.Tensor
    ):
        def masking_hook(value: t.Tensor, hook: HookPoint):
            masked = (
                mask_weight * batch_masks * value
                + (1 - mask_weight * batch_masks) * value.detach()
            )
            masked = (1 - zero_ablate_mask) * masked
            return masked

        return masking_hook

    def _get_masking_and_zero_ablation_hooks(
        self,
        batch,
        default_zero_ablate_mask: int,
        zero_ablate_indices: list[int],
        mask_weight: float = 1,
    ):
        hooks = {}
        for spec, mask_lookup_tensor in zip(self.mask_specs, self.mask_lookup_tensors):
            batch_mask_indexes = spec.masking_rule(batch).to(self.device)
            batch_masks = mask_lookup_tensor[batch_mask_indexes]
            zero_ablate_mask_indexes = t.full(
                batch_mask_indexes.shape, default_zero_ablate_mask
            )
            for zero_ablate_idx in zero_ablate_indices:
                zero_ablate_mask_indexes[batch_mask_indexes == zero_ablate_idx] = (
                    zero_ablate_idx
                )
            zero_ablate_mask = (
                ~(mask_lookup_tensor[zero_ablate_mask_indexes]).bool()
            ).int()
            for layer_idx in spec.layers:
                hook_name = get_hook_name(layer_idx, spec.hook_type)
                hook = MaskApplier._create_masking_and_zero_ablation_hook(
                    batch_masks,
                    mask_weight=mask_weight,
                    zero_ablate_mask=zero_ablate_mask,
                )
                hooks[hook_name] = hook
        return hooks

    @contextmanager
    def __call__(self, mask, mask_weight: float = 1):
        # Apply hooks
        try:
            for hook_name, hook in self._get_masking_hooks(mask, mask_weight).items():
                self.model.add_hook(hook_name, hook)
            yield

        finally:
            self.model.reset_hooks()

    @contextmanager
    def zero_ablate(self, mask_idx: int):
        # Apply hooks
        try:
            for hook_name, hook in self._get_zero_ablation_hooks(mask_idx).items():
                self.model.add_hook(hook_name, hook)
            yield

        finally:
            self.model.reset_hooks()

    @contextmanager
    def mask_and_zero_ablate(
        self: "MaskApplier",
        batch: t.Tensor,
        default_zero_ablate_mask: int,
        zero_ablate_indices: list[int],
        mask_weight: float = 1,
    ):
        """
        Masks the batch with the rule but then zero ablates dims that the mask indices in zero_ablate_dims DON'T route through.

        So for example, if we were expanding, we would pass the mask for the non-expanded data in zero_ablate_indices to ablate through the expanded indices on the off-target data.
        """
        try:
            for hook_name, hook in self._get_masking_and_zero_ablation_hooks(
                batch, default_zero_ablate_mask, zero_ablate_indices, mask_weight
            ).items():
                self.model.add_hook(hook_name, hook)
            yield
        finally:
            self.model.reset_hooks()


"""
---------------------------------------------------
mask_lookup - helper functions and implementations
---------------------------------------------------
"""


def multi_hot(list_of_indexes: List[int], d_masked: int, **kwargs):
    mask = t.zeros(d_masked, **kwargs)
    mask[list_of_indexes] = 1
    return mask


def get_resid_mask_lookup_from_dict(
    dims_to_mask: dict[int, List[int]], d_masked: int
) -> Callable[[int], Float[t.Tensor, "d_masked"]]:
    # Easy specification of a mask_lookup via a dictionary of dimensions-to-mask
    # (convenience function for 1-0 masking)
    def mask_lookup(mask_idx):
        mask = multi_hot(dims_to_mask[mask_idx], d_masked, dtype=t.bool)
        return mask

    return mask_lookup


def get_attn_mask_lookup_from_dict(
    heads_to_mask: dict[int, list[int]],
    n_heads: int,
    d_head: int,
) -> Callable[[int], Float[t.Tensor, "d_masked"]]:
    """
    You should apply this mask before the W_O matrix but after you concat all the heads.

    Args:
        heads_to_mask: a dictionary where the keys are the mask indices and the values
        are lists of heads to mask
        n_heads: the number of heads in the model
        d_head: the dimension of each head
    """

    def mask_lookup_attn(mask_idx):
        # 2D because TransformerLens attention hook returns (batch, seq, n_heads, d_head)
        mask = t.zeros((n_heads, d_head), dtype=t.bool)
        for head in heads_to_mask[mask_idx]:
            mask[head, :] = True
        return mask

    return mask_lookup_attn


def get_mlp_mask_lookup_from_dict(
    dims_to_mask: dict[int, List[int]], d_mlp: int
) -> Callable[[int], Float[t.Tensor, "d_masked"]]:
    def mask_lookup(mask_idx):
        mask = multi_hot(dims_to_mask[mask_idx], d_mlp, dtype=t.bool)
        return mask

    return mask_lookup


"""
---------------------------------------------------
masking_rule - helper functions and implementations
---------------------------------------------------
"""


def get_token_masking_rule(toks_to_mask: dict[str, int], tokenizer):
    no_masking_mask_idx = len(set(toks_to_mask.values()))
    token_to_mask = t.full((tokenizer.vocab_size,), no_masking_mask_idx, dtype=t.long)

    for token, mask_idx in toks_to_mask.items():
        encoded = tokenizer.encode(token)
        assert len(encoded) == 1, "All tokens to mask must be a single token"
        encoded = encoded[0]
        token_to_mask[encoded] = mask_idx

    def token_masking_rule(
        toks: Float[t.Tensor, "batch seq"],
    ) -> Float[t.Tensor, "batch seq"]:
        assert tokenizer.eos_token_id is not None

        left_shifted_toks = toks[:, 1:]
        return token_to_mask.to(toks.device)[  # WARNING: this is bad!!!!
            t.clamp(left_shifted_toks, max=tokenizer.vocab_size - 1)
        ]

    return token_masking_rule


def get_concept_masking_rule(
    concepts_to_localize: List[Tuple[List[str], int]],
    tokenizer,
):
    num_custom_masks = len(set([mask_idx for _, mask_idx in concepts_to_localize]))

    tokens_to_localize = []
    for words, mask_idx in concepts_to_localize:
        tokenized_all_words_all_variations = []
        for word in words:
            word_variations = [word.lower(), word.title()]
            for word_variation in word_variations:
                tokenized_word = tokenizer(
                    word_variation, return_tensors="pt"
                ).input_ids[0]
                assert tokenized_word.ndim == 1
                tokenized_all_words_all_variations.append(tokenized_word)
        tokens_to_localize.append((tokenized_all_words_all_variations, mask_idx))

    def concept_masking_rule(
        toks: Float[t.Tensor, "batch seq"],
    ) -> Float[t.Tensor, "batch seq"]:
        """
        TODO: write description of what this does
        """
        assert toks.ndim == 2
        mask = t.full_like(toks, num_custom_masks)
        input_tokens_shifted = toks[:, 1:]
        for toks, mask_idx in tokens_to_localize:
            unique_tok_lens = set(tok.shape[0] for tok in toks)
            for toks_len in unique_tok_lens:
                same_len_toks = t.stack(
                    [tok for tok in toks if tok.shape[0] == toks_len]
                ).to(input_tokens_shifted.device)
                assert same_len_toks.ndim == 2
                expanded_input_tokens = input_tokens_shifted[
                    :, None, :, None
                ].float()  # [batch, seq] -> [batch, channels/1, seq, height/1]
                kernel_size = (toks_len, 1)
                adjacent_tok_sets = F.unfold(
                    expanded_input_tokens, kernel_size
                )  # [batch, toks_len, n_blocks]
                equals_tensor = (
                    adjacent_tok_sets[:, None, :, :] == same_len_toks[None, :, :, None]
                )  # [batch, n_toks, toks_len, n_blocks]
                reduced_equals_tensor = (
                    equals_tensor.all(2)
                    .any(1, keepdim=True)
                    .repeat((1, toks_len, 1))
                    .float()
                )  # [batch, toks_len, n_blocks]
                refolded_matches = F.fold(
                    reduced_equals_tensor,
                    (input_tokens_shifted.shape[1], 1),
                    kernel_size,
                )  # [batch, channels/1, seq, height/1]
                matches = refolded_matches.squeeze((1, 3)).bool()  # [batch, toks]
                mask[:, :-1][matches] = mask_idx
        return mask[:, :-1]

    return concept_masking_rule


def get_full_sequence_concept_masking_rule(
    words_to_localize: List[str],
    tokenizer,
):
    concept_masking_rule = get_concept_masking_rule(
        [(words_to_localize, 0)],
        tokenizer,
    )

    def mask_rule(toks: t.Tensor) -> t.Tensor:
        """Mask the whole sequence if any concept is present"""
        concept_mask = concept_masking_rule(toks)
        seq_mins, _ = concept_mask.min(dim=-1, keepdim=True)
        seq_mask = einops.repeat(seq_mins, "b 1 -> b s", s=concept_mask.shape[1])
        return seq_mask

    return mask_rule


def get_token_frequencies(
    stories: list[str],
    tokenizer,
    synthetic_token_ct: Union[int, float],
    truncate_at: int | None,
):
    counter = Counter()
    for story in tqdm.tqdm(stories):
        if truncate_at:
            tokens = tokenizer(
                story, max_length=truncate_at, add_special_tokens=False, truncation=True
            )["input_ids"]
        else:
            tokens = tokenizer(story, add_special_tokens=False)["input_ids"]
        counter.update(tokens)

    counts = t.Tensor([counter[tok_idx] for tok_idx in range(tokenizer.vocab_size)])
    counts = counts + synthetic_token_ct
    freq = counts / counts.sum()
    return freq


def get_token_freq_masking_rule(
    retain_stories: list[str],
    forget_stories: list[str],
    num_stories: int,
    truncate_at: int | None,
    num_synthetic_tokens_retain: Union[int, float],  # encodes uniform prior over toks
    num_synthetic_tokens_forget: Union[int, float],  # encodes uniform prior over toks
    scale: float,
    bias: float,
    tokenizer,
    device: t.device,
):
    print("Getting token frequencies...")
    retain_freq = get_token_frequencies(
        retain_stories[:num_stories],
        tokenizer,
        num_synthetic_tokens_retain,
        truncate_at,
    )
    retain_counts_posterior = retain_freq
    forget_freq = get_token_frequencies(
        forget_stories[:num_stories],
        tokenizer,
        num_synthetic_tokens_forget,
        truncate_at,
    )
    forget_counts_posterior = forget_freq

    ratio = forget_counts_posterior / retain_counts_posterior
    inverse_ratio = retain_counts_posterior / forget_counts_posterior
    ratio_diff = ratio - inverse_ratio
    mask_weight = 1 - t.nn.functional.sigmoid(scale * ratio_diff + bias).to(device)

    info = dict(
        retain_freq=retain_freq.cpu().numpy(),
        forget_freq=forget_freq.cpu().numpy(),
        ratio=ratio.cpu().numpy(),
        inverse_ratio=inverse_ratio.cpu().numpy(),
        ratio_diff=ratio_diff.cpu().numpy(),
        mask_weight=mask_weight.cpu().numpy(),
    )

    def token_freq_masking_rule(toks):
        left_shifted_toks = toks[:, 1:]
        return mask_weight[t.clamp(left_shifted_toks, max=tokenizer.vocab_size - 1)]

    return token_freq_masking_rule, info


def get_mixture_masking_rule(
    masking_rules: list[Callable],
    mixture_weights: list[float],
):
    if sum(mixture_weights) != 1:
        warnings.warn(f"Mixture weights do not sum to 1: {mixture_weights}")

    def mixture_masking_rule(toks):
        masks = [rule(toks) for rule in masking_rules]
        mixed_mask = sum(mask * weight for mask, weight in zip(masks, mixture_weights))
        return mixed_mask

    return mixture_masking_rule
