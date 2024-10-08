from typing import Tuple, Union

import einops
import torch as t
import torch.nn.functional as F
from jaxtyping import Float


def get_cross_entropy_loss_unreduced(
    logits: t.Tensor, labels: t.Tensor, mask: t.Tensor
) -> t.Tensor:
    logits_flat = einops.rearrange(logits, "b s v -> (b s) v")
    labels_flat = einops.rearrange(labels, "b s -> (b s)")

    unmasked = t.nn.functional.cross_entropy(logits_flat, labels_flat, reduction="none")
    return (
        einops.rearrange(unmasked, "(b s) -> b s", b=logits.shape[0], s=logits.shape[1])
        * mask
    )


def get_cross_entropy_loss(
    logits: t.Tensor, labels: t.Tensor, mask: t.Tensor
) -> t.Tensor:
    loss = get_cross_entropy_loss_unreduced(logits, labels, mask)
    return loss.sum() / mask.sum()


def compute_preds_and_get_ce_loss_unreduced(
    model,
    tokens: t.Tensor,
    attention_mask: t.Tensor,
    other_mask: Union[t.Tensor, None],
) -> Tuple[t.Tensor, t.Tensor]:
    """
    Run a forward pass of the model and compute the cross-entropy loss.

    Args:
        model: The model to run the forward pass on.
        tokens: The full batch of tokens including the final token position that does
            not a have a corresponding ground truth. The input to the model will be
            `tokens[:, :-1]` so that we don't waste computation on the final token.
        attention_mask: The attention mask corresponding to `tokens`. The input to the
            model will be `attention_mask[:, :-1]` because we don't compute the
            prediction for the final token as there is no ground truth for it.
        other_mask: If provided, this mask will be multiplied with the attention mask
            before computing the loss. Should be the same shape as `tokens[:, :-1]`.


    Returns:
        The average cross-entropy loss across all tokens in the batch.
    """
    logits = model(tokens[:, :-1], attention_mask=attention_mask[:, :-1])

    attn_out = t.logical_and(attention_mask[:, :-1], attention_mask[:, 1:])
    loss_mask = attn_out if other_mask is None else attn_out * other_mask
    return get_cross_entropy_loss_unreduced(logits, tokens[:, 1:], loss_mask), loss_mask


def compute_preds_and_get_ce_loss(
    model,
    tokens: t.Tensor,
    attention_mask: t.Tensor,
    other_mask: Union[t.Tensor, None],
):
    loss, loss_mask = compute_preds_and_get_ce_loss_unreduced(
        model, tokens, attention_mask, other_mask
    )
    return loss.sum() / loss_mask.sum()


def compute_slgr_kl_div_loss(
    teacher_model_log_softmax,
    student_model,
    tokens: t.Tensor,
    attention_mask: t.Tensor,
    vocab_mask: Float[t.Tensor, "d_vocab"],
    reverse: bool,
    temp: float,
):
    """
    Assumes the model has already been masked with the right mask.
    """
    student_logits = student_model(tokens, attention_mask=attention_mask)
    attn_out = t.logical_and(attention_mask[:, :-1], attention_mask[:, 1:])
    # append the type of mask that was at the end to attn_out since we want to compare the predictions
    with t.no_grad():
        attn_out = t.cat([attn_out, attn_out[:, [-1]]], dim=1)
    student_log_softmax = F.log_softmax(student_logits / temp, dim=-1)
    # kl_div = t.nn.functional.kl_div(
    #    teacher_model_log_softmax,
    #    student_predictions,
    #    log_target=True,
    #    reduction="none",
    # ) # somehow this oomed but the manual version didn't ... weird
    if reverse:
        kl_div = teacher_model_log_softmax.exp() * (
            teacher_model_log_softmax - student_log_softmax
        )
    else:
        kl_div = student_log_softmax.exp() * (
            student_log_softmax - teacher_model_log_softmax
        )
    kl_div_masked_for_vocab = kl_div * vocab_mask[None, None, :]
    kl_div_summed_over_vocab = kl_div_masked_for_vocab.sum((2))
    kl_div_masked_for_attn = kl_div_summed_over_vocab * attn_out
    return kl_div_masked_for_attn.sum() / attn_out.sum()
