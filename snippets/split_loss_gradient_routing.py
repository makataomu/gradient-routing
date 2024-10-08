import einops
import torch
from torch.nn import Loss

"""
    A mocked-up example of split-loss gradient routing (SLGR).

    SLGR is an alternative to data-dependent backpropagation that is suitable
    for model distillation and which *doesn't* employ data-dependent masks.
    Instead, it splits a loss function acting on vector-valued inputs into
    multiple loss functions acting on subsets of that vector, and back-
    propagates these loss functions through different parts of the networks
    via masking.

    SLGR can be implemented in the forward pass (via stop gradient masks,
    expanding the computational graph), or in the backwards pass (by masking
    gradients across multiple backwards passes). As a consequence, it incurs
    an additional factor of O(NUM_SPLITS) complexity in the worst case.
"""

NUM_SPLITS = 2  # Our example defines two different routing schemes.


def resid_stream_mask_0(x: torch.Tensor):
    # Allow gradients to flow through entire residual stream for corresponding vocab words
    return x


def resid_stream_mask_1(x: torch.Tensor):
    # Force gradients to flow through dim 0 of residual stream for corresponding vocab words
    batch, seq, d_model = x.shape
    mask = torch.zeros(d_model)
    mask[0] = 1
    return mask * x + (1 - mask) * x.detach()


# In this example, we want to force token 47 into dim 0 of the residual stream.
TARGETED_VOCAB_WORD_INDEX = 47


# Define loss masks corresponding to the residual stream gradient masks.
# Each loss will have its gradient routed according its mask.
def logit_loss_mask_0(logits: torch.Tensor):
    return torch.cat(
        [logits[:TARGETED_VOCAB_WORD_INDEX], logits[TARGETED_VOCAB_WORD_INDEX:]]
    )


def logit_loss_mask_1(logits: torch.Tensor):
    return logits[TARGETED_VOCAB_WORD_INDEX]


RESID_STREAM_MASKS = [resid_stream_mask_0, resid_stream_mask_1]
LOGIT_LOSS_MASKS = [logit_loss_mask_0, logit_loss_mask_1]


def masked_distillation_loss(
    logits_student: torch.Tensor, logits_teacher: torch.Tensor
):
    """
    A SLGR loss function: take logits from a student and teacher model that have been
    processed according to different residual stream masks, apply the corresponding loss
    functions, and return the sum for backpropagation.
    """
    logits_student_by_mask = einops.rearrange(
        logits_student, "(m b) s d -> m b s d", m=NUM_SPLITS
    )
    logits_teacher_by_mask = einops.rearrange(
        logits_teacher, "(m b) s d -> m b s d", m=NUM_SPLITS
    )

    loss = 0
    for loss_mask, logits_student, logits_teacher in zip(
        LOGIT_LOSS_MASKS, logits_student_by_mask, logits_teacher_by_mask
    ):
        masked_student = loss_mask(logits_student)
        masked_teacher = loss_mask(logits_teacher)
        loss += Loss(masked_student, masked_teacher)

    return (
        loss  # We can simply backprop on this; gradients will be routed appropriately.
    )


def split_loss_mask_forward_hook(
    module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
):
    """
    A hook for a forward pass with split-loss gradient routing (SLGR).

    The hook applies to the output of a module: for each split of the loss, it
    applies the corresponding mask to the data, and rolls these split-indexed
    batches of data into a single batch dimension.
    """
    # Broadcast if the data doesn't have a "mask/loss" dimension yet.
    do_broadcast = output.shape[0] == module.current_batch_size

    if do_broadcast:
        output_masked = torch.cat([mask_fn(output) for mask_fn in RESID_STREAM_MASKS])
        return output_masked
    else:
        unstacked = einops.rearrange(output, "(m b) s d -> m b s d", m=NUM_SPLITS)
        output_masked = torch.cat(
            [
                mask_fn(unstacked[mask_idx])
                for mask_idx, mask_fn in enumerate(RESID_STREAM_MASKS)
            ]
        )
