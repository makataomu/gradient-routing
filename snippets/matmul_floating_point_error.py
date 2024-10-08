# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def sample_errors(d_model, n_samples, device):
    errors = []
    for _ in range(n_samples):
        d_mlp = 4 * d_model

        W_1 = t.rand((d_model, d_mlp), device=device) / np.sqrt(d_model)
        W_2 = t.rand((d_mlp, d_model), device=device) / np.sqrt(d_model)

        out = W_1 @ W_2

        permutation = t.randperm(d_mlp, device=device)
        out_perm = W_1[:, permutation] @ W_2[permutation, :]

        error = (out - out_perm).abs().max().item()
        errors.append(error)

    return errors


device = "cuda:3"

d_models = t.linspace(1, 4000, 100).long()
res = []
q_5s = []
q_95s = []
for d_model in tqdm.tqdm(d_models):
    errors = sample_errors(d_model, 50, device)
    res.append(np.mean(errors))
    q_5, q_95 = np.quantile(errors, [0.05, 0.95])
    q_5s.append(q_5)
    q_95s.append(q_95)

fig, ax = plt.subplots()
ax.set_xlabel("d_model")
ax.set_ylabel("(average) max of absolute errors")
title = "Error of permuted matmul vs. equivalent non-permuted"
subtitle = "(d_model,4*d_model) times (4*d_model,d_model)"
ax.set_title(title + "\n" + subtitle)

ax.plot(d_models, res)
ax.fill_between(d_models, q_5s, q_95s, alpha=0.5)
