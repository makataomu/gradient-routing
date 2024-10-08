# %%
from load_model import load_model_for_inference
import torch
import torch.nn.functional as F
import os
import re
import matplotlib.pyplot as plt
import numpy as np


start = "<|endoftext|>"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_normal_samples = 5
num_steered_samples = 5
max_new_tokens = 150  # number of tokens generated in each sample
temperature = (
    1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)


def get_latest_checkpoint(folder_path="out/"):
    # Get all files in the specified folder
    files = os.listdir(folder_path)

    # Filter for checkpoint files and extract their numbers
    checkpoints = []
    for file in files:
        match = re.match(r"ckpt_(\d+)\.pt", file)
        if match:
            checkpoints.append((int(match.group(1)), file))

    # If no checkpoints found, return None
    if not checkpoints:
        return "No path fits"

    # Sort checkpoints by number (descending) and return the filename of the latest
    latest_checkpoint = max(checkpoints, key=lambda x: x[0])[1]
    return latest_checkpoint if latest_checkpoint is not None else "No path fits"


path = get_latest_checkpoint()
print("loading checkpoint", path)
# don't run configurator if running in vscode jupyter
if "get_ipython" not in globals():
    exec(open("configurator.py").read())  # overrides from command line or config file
model, mask_config, encode, decode_most, ctx, device = load_model_for_inference(
    path=path, compile=False
)


def decode(li):
    try:
        return decode_most(li)  # the tokenizer can't handle end of text for some reason
    except Exception:
        print("error decoding", li)
        return "???"


# encode the beginning of the prompt
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start, allowed_special={"<|endoftext|>"})
# start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
print("regular generations")
with torch.no_grad():
    with ctx:
        for k in range(num_normal_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print("---------------")

# this is a coin flip if it's pos or neg
steer_with = -30

# %%
k = 20
# print(
#    f"unembed for 0th residual stream direction (top {k}):",
#    [decode([ti]) for ti in model.lm_head.weight[:, 0].topk(k).indices.cpu().numpy()],
# )
# print(
#    f"unembed for -0th residual stream direction (top {k}):",
#    [
#        decode([ti])
#        for ti in (-model.lm_head.weight[:, 0]).topk(k).indices.cpu().numpy()
#    ],
# )
cos_sim_pos = (F.normalize(model.lm_head.weight, dim=1))[:, 0].detach().cpu().topk(k)
print(
    f"unembed for 0th residual stream direction highest cos sim(top {k}):",
    [decode([ti]) for ti in cos_sim_pos.indices.numpy()],
)
cos_sim_neg = (F.normalize(-model.lm_head.weight, dim=1)[:, 0]).detach().cpu().topk(k)
print(
    f"unembed for -0th residual stream direction highest cos sim(top {k}):",
    [decode([ti]) for ti in cos_sim_neg.indices.numpy()],
)
cos_sim = cos_sim_pos if steer_with > 0 else cos_sim_neg
top_k_toks_and_cos_sim = [
    (decode([ti]), ci)
    for ti, ci in zip(cos_sim.indices.numpy(), cos_sim.values.numpy())
]
labels, values = zip(*top_k_toks_and_cos_sim)
print(", ".join(list(map(lambda s: f"\\texttt{{{s.replace(" ", "\\_")}}}", labels))))
print(top_k_toks_and_cos_sim)


# Create the plot
fig, ax = plt.subplots(figsize=(3, 4))  # Adjusted figure size for better visibility

# Create horizontal bar plot
y_pos = np.arange(len(labels))
ax.barh(y_pos, values)

# Customize the plot
ax.set_yticks(y_pos)
labels_space_to_underscore = [label.replace(" ", "_") for label in labels]
ax.set_yticklabels(labels_space_to_underscore)
ax.invert_yaxis()  # Labels read top-to-bottom
ax.set_xlabel("Cos sim with [1, 0, ..., 0]")
# x label font size less
ax.xaxis.label.set_size(8)
# center the title
# ax.set_title(f"Top {k} tokens localized to the 0th residual stream dimension")
# log scale x axis

# Set x-axis limit between 0 and 1
# ax.set_xlim(0, 1)

# Adjust layout and display the plot
plt.tight_layout()
plt.margins(0, 0)
plt.savefig(
    f"top_{k}_tokens_localized_to_0th_residual_stream_dimension_{path}.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
)
plt.show()
# %%

# print("\n\nAblation\n\n")
# with torch.no_grad():
#    with ctx:
#        for k in range(num_samples):
#            y = model.generate(
#                x,
#                max_new_tokens,
#                temperature=temperature,
#                top_k=top_k,
#                ablate_idx=torch.tensor(0),
#            )
#            print(decode(y[0].tolist()))
#            print("---------------")

print("\n\nSteering\n\n")


def hook(module, input, output):
    output[:, :, 0] += steer_with
    return output


# model.transformer.h[min(mask_config.blocks_to_mask)].register_forward_hook(hook)
model.transformer.h[10].register_forward_hook(hook)

with torch.no_grad():
    with ctx:
        for k in range(num_steered_samples):
            y = model.generate(
                x,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            print(decode(y[0].tolist()))
            print("---------------")

# %%
