# %%
import torch as t
from torch import nn


def reset_params(model):
    for layer in model.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    def __init__(self, layer_sizes, use_relu_on_output):
        super().__init__()
        assert len(layer_sizes) > 0
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )
        self.use_relu_on_output = use_relu_on_output

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = t.relu(layer(x))
        x = self.layers[-1](x)
        if self.use_relu_on_output:
            x = t.relu(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def get_action_logits(self, obs_stack: t.Tensor):
        raise NotImplementedError("A PolicyNetwork must define the way it gets logits.")

    @t.inference_mode()
    def sample_action(self, obs_stack: t.Tensor):
        logits = self.get_action_logits(obs_stack)
        action = t.distributions.Categorical(logits=logits).sample()
        return action

    def get_action_logprobs(self, obs_stack: t.Tensor, actions: t.Tensor):
        logits = self.get_action_logits(obs_stack)
        dist = t.distributions.Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        info = {"entropy": dist.entropy()}
        return logprobs, info


class PolicyWrapper(PolicyNetwork):
    def __init__(self, module_list):
        super().__init__()
        self.neural_net = nn.Sequential(*module_list)

    def get_action_logits(self, obs_stack: t.Tensor):
        logits = self.neural_net(obs_stack)
        return logits


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNetwork, self).__init__()
        self.neural_net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1, stride=1),
            nn.Flatten(),
            nn.Linear(obs_dim * 4, 256),
            nn.Linear(256, 32),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.neural_net(x)


class ConstantGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_param = nn.Parameter(t.tensor(0.0))

    def forward(self, x):
        batch_size = x.shape[0]
        return t.full((batch_size, 1), fill_value=0.0).to(x.device)


def get_single_expert_policy(obs_dim, num_actions):
    return RoutedPolicyNetwork(
        obs_dim, num_actions, use_gate=False, use_gradient_routing=False
    ).get_diamond_policy()


class RoutedPolicyNetwork(PolicyNetwork):
    """
    Architecture is:
        input -> shared_input -> diamond_expert, ghost_expert -> shared_output
    """

    def __init__(
        self, obs_dim: int, num_actions: int, use_gate: bool, use_gradient_routing: bool
    ):
        super().__init__()
        self.shared_input = nn.Sequential()
        self.ghost_expert = nn.Sequential(
            nn.Flatten(),
            MLP([obs_dim, 256, 256], use_relu_on_output=True),
        )
        self.diamond_expert = nn.Sequential(
            nn.Flatten(),
            MLP([obs_dim, 256, 256], use_relu_on_output=True),
        )
        self.shared_output = nn.Linear(256, num_actions)

        if use_gate:
            self.gating = nn.Sequential(
                nn.Conv2d(4, 4, kernel_size=1, stride=1),
                nn.Flatten(),
                MLP([obs_dim, 256, 256, 1], use_relu_on_output=False),
            )
        else:
            self.gating = ConstantGate()

        self.use_gradient_routing = use_gradient_routing

    def forward(self, x):
        inp_enc = self.shared_input(x)
        diamond_out = self.diamond_expert(inp_enc)
        ghost_out = self.ghost_expert(inp_enc)
        gate_out = self.gating(x)
        return diamond_out, ghost_out, gate_out

    @staticmethod
    def _combine_experts(diamond_out, ghost_out, gate_out):
        gate_prob = t.sigmoid(gate_out)
        return gate_prob * diamond_out + (1 - gate_prob) * ghost_out

    def get_action_logits(self, obs_stack: t.Tensor):
        diamond_out, ghost_out, gate_out = self.forward(obs_stack)
        weighted_out = self._combine_experts(diamond_out, ghost_out, gate_out)
        logits = self.shared_output(weighted_out)
        return logits

    def get_parameters(self):
        expert_params = list(self.ghost_expert.parameters()) + list(
            self.diamond_expert.parameters()
        )
        shared_params = (
            list(self.gating.parameters())
            + list(self.shared_output.parameters())
            + list(self.shared_input.parameters())
        )
        return expert_params, shared_params

    def get_diamond_policy(self):
        return PolicyWrapper(
            [self.shared_input, self.diamond_expert, self.shared_output]
        )

    def get_ghost_policy(self):
        return PolicyWrapper([self.shared_input, self.ghost_expert, self.shared_output])

    @staticmethod
    def _get_logit_stats(logits, actions):
        dist = t.distributions.Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprobs, entropy

    def get_training_info(self, obs_stack: t.Tensor, actions: t.Tensor):
        diamond, ghost, gate_out = self.forward(obs_stack)

        if self.use_gradient_routing:
            gate_d = gate_out.detach()
            outs_routed = {
                "diamond": self._combine_experts(diamond, ghost.detach(), gate_d),
                "ghost": self._combine_experts(diamond.detach(), ghost, gate_d),
                "both": self._combine_experts(diamond, ghost, gate_out),
            }
        else:
            out = self._combine_experts(diamond, ghost, gate_out)
            outs_routed = {"both": out, "diamond": out, "ghost": out}

        logits_routed = {
            key: self.shared_output(out) for key, out in outs_routed.items()
        }

        logprobs_routed = {}
        entropy_routed = {}

        for key, logits in logits_routed.items():
            logprobs_routed[key], entropy_routed[key] = self._get_logit_stats(
                logits, actions
            )

        return logprobs_routed, entropy_routed, gate_out


def get_reinforce_loss(processed_batch, policy, value_fn, coefs: dict):
    obs = processed_batch["obs"]
    actions = processed_batch["actions"]
    returns = processed_batch["returns"]

    logprobs, policy_info = policy.get_action_logprobs(obs, actions)  # type: ignore

    entropy_bonus = coefs["entropy_bonus"] * t.mean(policy_info["entropy"])

    value_pred = value_fn(obs).squeeze()
    policy_loss = -t.mean(logprobs * (returns - value_pred.detach()))
    value_loss = coefs["value_loss"] * t.mean((value_pred - returns) ** 2)
    loss = policy_loss + value_loss - entropy_bonus

    batch_metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy_bonus": entropy_bonus.item(),
        "loss": loss.item(),
        "avg_action_prob": t.mean(t.exp(logprobs)).item(),
        "avg_return": t.mean(returns).item(),
    }
    return loss, batch_metrics


def _propagate_indicators_backward(arr):
    """
    Helper function to label entire episode with end-of-episode label

    Args:
        arr - a (timestep, env)-shaped tensor with values in (-1, 0, 1),
            with "-1" indicating a timestep in the middle of an episode,
            and 0 or 1 occuring at the end of an episode, indicating
            whether some condition is true (e.g. "ghost reached.")
    """
    assert len(arr.shape) == 2
    bool_arr = t.zeros(arr.shape, dtype=t.bool, device=arr.device)
    bool_arr[-1] = arr[-1] == 1
    for timestep in reversed(range(len(arr) - 1)):
        is_missing = arr[timestep] == -1
        is_true_now = arr[timestep] == 1
        is_true_at_next = bool_arr[timestep + 1] == 1

        bool_arr[timestep] = t.logical_or(
            is_true_now, t.logical_and(is_missing, is_true_at_next)
        )
    return bool_arr


def _process_info_arr(arr):
    return _propagate_indicators_backward(arr).flatten().unsqueeze(-1)


def get_filtered_reinforce_loss(processed_batch, policy, value_fn, coefs: dict):
    device = next(policy.parameters()).device

    info = processed_batch["infos"]

    ep_reached_oversight = _propagate_indicators_backward(info["oversight"])
    ep_reached_ghost = _propagate_indicators_backward(info["reached_ghost"])
    ep_reached_diamond = _propagate_indicators_backward(info["reached_diamond"])
    reached_nothing = t.logical_and(~ep_reached_diamond, ~ep_reached_ghost)
    assert not t.logical_and(ep_reached_diamond, ep_reached_ghost).any()
    assert (ep_reached_ghost.float() + ep_reached_diamond + reached_nothing == 1).all()

    has_oversight = (
        t.logical_or(ep_reached_oversight, reached_nothing).flatten().to(device)
    )
    kept_pct = has_oversight.sum() / len(has_oversight)

    filtered_batch = {
        key: processed_batch[key][has_oversight]
        for key in ["obs", "actions", "returns"]
    }

    loss, batch_metrics = get_reinforce_loss(filtered_batch, policy, value_fn, coefs)

    # Correct for smaller batch sizes to prevent late-run instability
    scaled_loss = kept_pct * loss
    batch_metrics["kept_pct"] = kept_pct.item()
    batch_metrics["loss"] = scaled_loss.item()

    return scaled_loss, batch_metrics


def get_routed_reinforce_loss(processed_batch, policy, value_fn, coefs: dict):
    device = next(policy.parameters()).device

    obs = processed_batch["obs"]
    actions = processed_batch["actions"]
    returns = processed_batch["returns"]
    info = processed_batch["infos"]

    ep_has_oversight = _process_info_arr(info["oversight"])
    ep_reached_ghost = _process_info_arr(info["reached_ghost"])
    ep_reached_diamond = _process_info_arr(info["reached_diamond"])
    reached_nothing = ~ep_reached_diamond * ~ep_reached_ghost
    assert not t.logical_and(ep_reached_diamond, ep_reached_ghost).any()
    assert (ep_reached_ghost.float() + ep_reached_diamond + reached_nothing == 1).all()

    route_indicators = {
        "both": t.logical_or(~ep_has_oversight, reached_nothing),
        "diamond": t.logical_and(ep_has_oversight, ep_reached_diamond),
        "ghost": t.logical_and(ep_has_oversight, ep_reached_ghost),
    }

    value_pred = value_fn(obs).squeeze()
    value_pred_baseline = value_pred.detach()

    logprobs_routed, entropy_routed, gate_logit = policy.get_training_info(
        obs,
        actions,  # type: ignore
    )

    policy_loss = t.tensor(0.0).to(device)
    entropy_bonus = t.tensor(0.0).to(device)
    gate_loss = t.tensor(0.0).to(device)

    batch_metrics = {}

    end_of_ep_labels = 0.5 * (
        1 + route_indicators["diamond"].float() - route_indicators["ghost"].float()
    )

    understood_terminal_state = t.logical_or(
        route_indicators["diamond"], route_indicators["ghost"]
    )
    gate_loss = nn.BCEWithLogitsLoss()(
        gate_logit[understood_terminal_state],
        end_of_ep_labels[understood_terminal_state],
    )

    gate_prob = nn.functional.sigmoid(gate_logit)
    gate_no_oversight = gate_prob[~understood_terminal_state]
    gate_loss += coefs["gate_loss_no_oversight"] * (
        t.minimum(gate_no_oversight, 1 - gate_no_oversight).mean()
    )

    # Compute policy gradient loss for each route type
    for route_type in ["both", "diamond", "ghost"]:
        to_train = route_indicators[route_type].squeeze(-1)
        logprobs = logprobs_routed[route_type][to_train]
        entropy = entropy_routed[route_type][to_train]
        returns_subset = returns[to_train]
        gate_subset = gate_prob[to_train]

        loss_wt = to_train.sum() / t.numel(to_train)
        value_baseline = value_pred_baseline[to_train]
        policy_loss += -loss_wt * t.mean(logprobs * (returns_subset - value_baseline))

        entropy_bonus += -loss_wt * t.mean(entropy)

        if route_type == "diamond":
            batch_metrics["avg_seen_diamond_gate"] = t.mean(gate_subset).item()
        elif route_type == "ghost":
            batch_metrics["avg_seen_ghost_gate"] = t.mean(gate_subset).item()
        else:
            batch_metrics["avg_unseen_or_unfinished_gate"] = t.mean(gate_subset).item()

    value_loss = coefs["value_loss"] * t.mean((value_pred - returns) ** 2)
    entropy_bonus *= coefs["entropy_bonus"]
    gate_loss *= coefs["gate_loss"]

    loss = policy_loss + value_loss + entropy_bonus + gate_loss

    batch_metrics |= {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "gate_loss": gate_loss.item(),
        "entropy_bonus": entropy_bonus.item(),
        "loss": loss.item(),
        "avg_action_prob": t.mean(t.exp(logprobs_routed["both"])).item(),
        "avg_return": t.mean(returns).item(),
        "avg_gate_openness": t.mean(t.minimum(1 - gate_prob, gate_prob)).item(),
    }

    return loss, batch_metrics


if __name__ == "__main__":
    import projects.minigrid_repro.grid as grid

    env_kwargs = dict(
        n_envs=17,
        nrows=5,
        ncols=5,
        max_step=32,
        oversight_prob=0.5,
        spurious_oversight_prob=0.0,
    )

    device = "cuda"
    env = grid.ContinuingEnv(device=device, **env_kwargs)  # type: ignore
    obs = env.get_obs()

    policy = RoutedPolicyNetwork(
        env.obs_size, 4, use_gate=False, use_gradient_routing=True
    ).to(device)

    out = policy.forward(obs)
    for item in out:
        print(item.shape)

    actions = policy.sample_action(obs)

    expert_policy = policy.get_diamond_policy()
    expert_policy.sample_action(obs)

    def test_propagate_indicators_backward():
        input_1 = t.tensor([-1, 0])
        expected_1 = t.tensor([0, 0], dtype=t.bool)

        input_2 = t.tensor([-1.0, -1.0, -1.0, 1.0, 0.0, -1.0])
        expected_2 = t.tensor([1, 1, 1, 1, 0, 0], dtype=t.bool)

        for inp, exp in [(input_1, expected_1), (input_2, expected_2)]:
            out = _propagate_indicators_backward(inp.unsqueeze(-1)).squeeze()
            assert out.allclose(exp), print("Expected:", exp, "\nGot:     ", out)

    test_propagate_indicators_backward()
