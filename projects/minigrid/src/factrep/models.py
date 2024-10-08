from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from factrep.layers import GatedShards
from jaxtyping import Float, Int
from torch import nn
from torch.distributions.categorical import Categorical


def normalize(vectors, epsilon):
    # vectors.shape = (batch, embed)
    return (vectors - vectors.mean(dim=0)) / (vectors.std(dim=0) + epsilon)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


Architecture = Literal["original", "shards_as_heads", "mixture_of_actions", "small"]


@dataclass
class AgentConfig:
    architecture: Architecture
    action_space_dim: int
    teacher_forcing: bool
    shifting_alphas: bool
    negative_gradient: bool
    gating_enabled: bool
    routing_enabled: bool
    n_shards: int
    padding_same: bool = True


def get_pre_sharding_layers(architecture: Architecture, padding="same"):
    match architecture:
        case "original" | "small":
            return nn.Identity()
        case "shards_as_heads":
            return nn.Sequential(
                nn.LazyConv2d(16, kernel_size=2, padding=padding),  # type: ignore
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                layer_init(nn.Conv2d(16, 32, kernel_size=2, padding=padding)),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=2, padding=padding),
                nn.ReLU(),
                nn.Flatten(),
            )
        case "mixture_of_actions":
            return nn.Sequential(
                nn.LazyConv2d(16, kernel_size=2, padding=padding),  # type: ignore
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                layer_init(nn.Conv2d(16, 32, kernel_size=2, padding=padding)),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=2, padding=padding),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(64),
                nn.ReLU(),
            )


def get_post_sharding_layers(
    architecture: Architecture, action_space_dim: int, padding="same"
):
    match architecture:
        case "original":
            return nn.Sequential(
                layer_init(nn.Conv2d(16, 32, kernel_size=2, padding=padding)),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=2, padding=padding),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(64),
                nn.ReLU(),
                layer_init(nn.Linear(64, action_space_dim), std=1.0),
            )
        case "small":
            return nn.Sequential(
                layer_init(nn.Conv2d(16, 32, kernel_size=2, padding=padding)),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(32),
                nn.ReLU(),
                layer_init(nn.Linear(32, action_space_dim), std=1.0),
            )
        case "shards_as_heads":
            return nn.Sequential(
                layer_init(nn.Linear(64, action_space_dim), std=1.0),
            )
        case "mixture_of_actions":
            return nn.Identity()


def get_gating_layers(
    architecture: Architecture,
    gating_enabled: bool = True,
    padding: Literal["same", "valid"] = "same",
):
    if not gating_enabled:
        return lambda x: torch.ones(x.shape[0]).to(x.device)

    match architecture:
        case "original":
            return nn.Sequential(
                nn.LazyConv2d(16, kernel_size=2, padding=padding),  # type: ignore
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                layer_init(nn.Conv2d(16, 32, kernel_size=2, padding=padding)),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=2, padding=padding),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(64),
                nn.ReLU(),
            )
        case "small":
            return nn.Sequential(
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.LazyConv2d(16, kernel_size=2, padding=padding),  # type: ignore
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                layer_init(nn.Conv2d(16, 32, kernel_size=2, padding=padding)),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(32),
                nn.ReLU(),
            )
        case "shards_as_heads":
            return nn.Sequential(
                nn.LazyLinear(64),
                nn.ReLU(),
            )
        case "mixture_of_actions":
            return nn.Sequential(
                nn.LazyLinear(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )


def get_sharding_layers(
    architecture: Architecture,
    teacher_forcing: bool,
    shifting_alphas: bool,
    negative_gradient: bool,
    action_space_dim: int,
    n_shards: int,
    gating_enabled: bool,
    routing_enabled: bool,
    padding: Literal["same", "valid"] = "same",
):
    match architecture:
        case "original":
            make_shard = lambda _: nn.Sequential(
                nn.LazyConv2d(16, kernel_size=2, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
        case "small":
            make_shard = lambda _: nn.LazyConv2d(16, 2, padding="same")
        case "shards_as_heads":
            make_shard = lambda _: nn.Sequential(
                nn.LazyLinear(64),
                nn.ReLU(),
            )
        case "mixture_of_actions":
            make_shard = lambda _: nn.Sequential(
                nn.LazyLinear(16),
                nn.ReLU(),
                nn.Linear(16, action_space_dim),
            )

    return GatedShards(
        n_shards=n_shards,
        teacher_forcing=teacher_forcing,
        shifting_alphas=shifting_alphas,
        gating_enabled=gating_enabled,
        negative_gradient=negative_gradient,
        make_shard=make_shard,
        routing_enabled=routing_enabled,
    )


class Agent(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config

        # critic_conv = nn.Sequential(
        #     nn.LazyConv2d(16, kernel_size=2, padding="same"),  # type: ignore
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     layer_init(nn.Conv2d(16, 32, kernel_size=2)),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=2, padding="same"),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        # self.critic = nn.Sequential(
        #     critic_conv,
        #     nn.LazyLinear(64),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(64, 1), std=1.0),
        # )
        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

        self.pre_sharding = get_pre_sharding_layers(config.architecture)
        self.post_sharding = get_post_sharding_layers(
            config.architecture, config.action_space_dim
        )
        self.gating = get_gating_layers(config.architecture, config.gating_enabled)
        self.sharding = get_sharding_layers(
            architecture=config.architecture,
            teacher_forcing=config.teacher_forcing,
            shifting_alphas=config.shifting_alphas,
            negative_gradient=config.negative_gradient,
            action_space_dim=config.action_space_dim,
            gating_enabled=config.gating_enabled,
            routing_enabled=config.routing_enabled,
            n_shards=config.n_shards,
        )

        self.n_shards = config.n_shards

    def get_value(self, x):
        return self.critic(x)

    def _get_gated_action(
        self,
        state: Float[torch.Tensor, "batch c h w"],
        true_alphas: Optional[Int[torch.Tensor, "batch n_shards"]],
    ) -> tuple[Float[torch.Tensor, "batch n_actions"], dict[str, dict[str, Any]]]:
        state_enc = self.pre_sharding(state)
        shards = self.sharding(state_enc, self.gating(state_enc), true_alphas)
        logits = self.post_sharding(shards["output"])

        if true_alphas is None:
            return logits, {}

        shard_outputs = shards["individual_shard_outputs"].flatten(0, 1)

        metrics = {}

        for shard_idx, true_alpha in enumerate(true_alphas.T):
            computed_alpha = shards["computed_alphas"][:, shard_idx]
            used_alpha = shards["used_alphas"][:, shard_idx]
            for val in [-1, 0, 1]:
                true_val_label = val if val != -1 else "default"
                label = f"computed_alpha_shard={shard_idx}_true={true_val_label}"
                metrics[label] = computed_alpha[true_alpha == val].mean().item()
                used_label = f"used_alpha_shard={shard_idx}_true={true_val_label}"
                metrics[used_label] = used_alpha[true_alpha == val].mean().item()

        losses = {
            "l1": shard_outputs.abs().mean(0).sum(),
            "l2": shard_outputs.pow(2).mean(0).sum().sqrt(),
            "alpha": shards["alpha_loss"],
        }

        return logits, {"metrics": metrics, "loss": losses}

    def get_action_and_value(
        self,
        x: Float[torch.Tensor, "batch c h w"],
        true_alphas: Optional[Int[torch.Tensor, "batch"]] = None,
        action: Optional[Int[torch.Tensor, "batch"]] = None,
        deterministic: bool = False,
    ):
        logits, info = self._get_gated_action(x, true_alphas)
        probs = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=1)
            else:
                action = probs.sample()

        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(x),
            {"loss": {}, "metrics": {}} | info,
        )

    def get_action_distribution(
        self,
        x: Float[torch.Tensor, "batch c h w"],
        true_alphas: Optional[Int[torch.Tensor, "batch"]] = None,
    ) -> Float[torch.Tensor, "batch n_actions"]:
        logits, _ = self._get_gated_action(x, true_alphas)

        return logits.softmax(dim=-1)

    def save(self, path: Path | str, metadata: dict[str, Any] = {}):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": asdict(self.config),
                "metadata": metadata,
            },
            path,
        )

    @staticmethod
    def load(path: Path | str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(path, map_location=device)
        config = AgentConfig(**model["config"])
        agent = Agent(config)
        agent.load_state_dict(model["state_dict"])
        agent.metadata = model["metadata"]
        return agent
