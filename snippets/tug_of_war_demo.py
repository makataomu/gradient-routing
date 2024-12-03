# %%

import torch as t
import matplotlib.pyplot as plt
import numpy as np

"""
This script gives a minimal example of the tug of war effect that
results from using gradient routing to "point" a collection of
parameters at two different targets simultaneously.

The script also shows how this tug of war can be broken by including
a "residual coherence" term. This term essentially induces different
learning targets for different parameter subsets, removing the tug of
war effect.
"""

num_runs = 3
num_optimizer_steps = 1000

forget_wt = 1
residual_coherence_wt = 0.2
l1_penalty = 0


t.manual_seed(3)

thetas_list = []
theta_grads_list = []
for _ in range(num_runs):
    theta = t.empty(2, requires_grad=True)
    theta.data[:] = t.randn(2) * 4
    optimizer = t.optim.SGD([theta], lr=0.01)

    x_retain = t.tensor([2])
    x_forget = t.tensor([1])

    thetas = t.zeros((num_optimizer_steps, 2))
    theta_grads = t.zeros((num_optimizer_steps, 2))
    for step in range(num_optimizer_steps):
        optimizer.zero_grad()

        retain_loss = (theta.sum() - x_retain) ** 2
        retain_coherence_loss = (theta[0] - x_retain) ** 2
        forget_loss = (theta[0].detach() + theta[1] - x_forget) ** 2
        l1_norm = t.abs(theta).sum()

        loss = (
            retain_loss
            + forget_wt * forget_loss
            + residual_coherence_wt * retain_coherence_loss
            + l1_penalty * l1_norm
        )

        loss.backward()
        optimizer.step()

        thetas[step] = theta.detach().clone()
        theta_grads[step] = theta.grad.clone()  # type: ignore
    thetas_list.append(thetas)
    theta_grads_list.append(theta_grads)

fig, ax = plt.subplots()
description = ""
ax.set_title(f"Parameter trajectories over multiple runs\n{description}")
ax.set_xlabel("theta[0] (updates on retain data only)")
ax.set_ylabel("theta[1] (updates on all data)")
for param_idx, thetas in enumerate(thetas_list):
    ax.scatter(
        thetas[:, 0],
        thetas[:, 1],
        np.geomspace(5, 0.01, num_optimizer_steps),
        alpha=0.5,
    )
    ax.scatter(
        thetas[-1, 0],
        thetas[-1, 1],
        c="gold",
        marker="*",
        edgecolors="black",
        s=100,
        label="Final value" if param_idx == 0 else None,
    )
ax.legend()

fig, ax = plt.subplots()
ax.set_title("Sample gradient trajectory")
ax.set_xlabel("Update step")

ax.axhline(0, c="black", ls=":", alpha=0.3)
ax.plot(
    range(num_optimizer_steps),
    thetas_list[0][:, 0] + thetas_list[0][:, 1],
    label="theta[0] + theta[1]",
)
ax.plot(
    range(num_optimizer_steps),
    theta_grads_list[0][:, 0],
    label="theta[0].grad",
    ls="--",
    c="C1",
)
ax.plot(
    range(num_optimizer_steps),
    theta_grads_list[0][:, 1],
    label="theta[1].grad",
    ls="--",
    c="C1",
)
ax.legend()
