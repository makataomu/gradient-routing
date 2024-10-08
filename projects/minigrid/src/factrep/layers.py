from typing import Callable, Optional

import einops
import torch as t
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.nn import functional as F


def negate_gradient(
    x: Float[t.Tensor, "batch ..."], keep_grad: Bool[t.Tensor, "batch n_shards"]
):
    with t.no_grad():
        coef = keep_grad.float().reshape(
            *keep_grad.shape, *([1] * (x.dim() - keep_grad.dim()))
        )
    return coef * x + (1 - coef) * 2 * x.detach() - (1 - coef) * x


def route_gradient(
    x: Float[t.Tensor, "batch ..."], keep_grad: Bool[t.Tensor, "batch n_shards"]
):
    with t.no_grad():
        coef = keep_grad.float().reshape(
            *keep_grad.shape, *([1] * (x.dim() - keep_grad.dim()))
        )

    return coef * x + (1 - coef) * x.detach()


def alpha_loss(
    true_alphas: Int[t.Tensor, "batch n_shards"],
    computed_alphas: Float[t.Tensor, "batch n_shards"],
) -> Float[t.Tensor, "batch"]:
    true_alphas_flat = true_alphas.flatten()
    computed_alphas_flat = computed_alphas.flatten()

    return F.binary_cross_entropy(
        computed_alphas_flat[true_alphas_flat != -1],
        true_alphas_flat[true_alphas_flat != -1],
    )


class GatedShards(nn.Module):
    def __init__(
        self,
        n_shards: int,
        teacher_forcing: bool,
        shifting_alphas: bool,
        gating_enabled: bool,
        routing_enabled: bool,
        negative_gradient: bool,
        make_shard: Callable[[int], nn.Module],
    ):
        super(GatedShards, self).__init__()

        self.n_shards = n_shards
        self.teacher_forcing = teacher_forcing
        self.shifting_alphas = shifting_alphas
        self.gating_enabled = gating_enabled
        self.negative_gradient = negative_gradient
        self.routing_enabled = routing_enabled
        self.gate = nn.LazyLinear(n_shards)

        self.shards = nn.ModuleList([make_shard(i) for i in range(n_shards)])

    def _compute_alphas(
        self,
        x_gate: Float[t.Tensor, "batch ..."],
        true_alphas: Optional[Int[t.Tensor, "batch n_shards"]],
    ) -> tuple[Float[t.Tensor, "batch n_shards"], Float[t.Tensor, "batch n_shards"]]:
        if not self.gating_enabled:
            with t.no_grad():
                constant_1 = t.ones(x_gate.shape[0], self.n_shards).to(x_gate.device)
            return constant_1, constant_1

        computed_alphas = F.sigmoid(self.gate(x_gate))

        if true_alphas is None:
            return computed_alphas, computed_alphas

        with t.no_grad():
            alphas_are_known = true_alphas != -1
        if self.training and not self.teacher_forcing:
            intermediate_alphas = computed_alphas
        else:
            # We're either in training with teacher forcing, or in steered inference
            # Whenever the true shard activation is known, we use it
            intermediate_alphas = t.where(
                alphas_are_known, true_alphas, computed_alphas
            )

        if self.shifting_alphas and not self.teacher_forcing:
            used_alphas = intermediate_alphas
        else:
            if self.negative_gradient:
                used_alphas = negate_gradient(
                    intermediate_alphas, keep_grad=~alphas_are_known
                )
            else:
                used_alphas = route_gradient(
                    intermediate_alphas, keep_grad=~alphas_are_known
                )

        # computed_alphas get gradients from the alpha loss
        # used_alphas get gradients from the the normal RL / value loss
        return computed_alphas, used_alphas

    def _compute_shards(
        self,
        x: Float[t.Tensor, "batch ..."],
        true_alphas: Optional[Int[t.Tensor, "batch n_shards"]],
    ) -> Float[t.Tensor, "batch n_shards ..."]:
        intermediate: Float[t.Tensor, "batch n_shards ..."] = t.stack(
            [shard(x) for shard in self.shards], dim=1
        )

        if true_alphas is None:
            return intermediate

        # We remove gradients from shards we know should not have been used
        if self.negative_gradient:
            output = negate_gradient(intermediate, keep_grad=(true_alphas != 0))
        elif self.routing_enabled:
            output = route_gradient(intermediate, keep_grad=(true_alphas != 0))
        else:
            output = intermediate

        return output

    def forward(
        self,
        x: Float[t.Tensor, "batch ..."],
        x_gate: Float[t.Tensor, "batch ..."],
        true_alphas: Optional[Int[t.Tensor, "batch n_shards"]],
    ) -> dict[str, t.Tensor]:
        x_gate = x if x_gate is None else x_gate
        computed_alphas, used_alphas = self._compute_alphas(x_gate, true_alphas)
        shard_outputs = self._compute_shards(x, true_alphas)
        output = einops.einsum(
            used_alphas,
            shard_outputs,
            "batch n_shards, batch n_shards ... -> batch ...",
        )

        result = {
            "output": output,
            "individual_shard_outputs": shard_outputs,
            "computed_alphas": computed_alphas,
            "used_alphas": used_alphas,
        }
        if true_alphas is not None:
            result["alpha_loss"] = alpha_loss(true_alphas, computed_alphas)

        return result
