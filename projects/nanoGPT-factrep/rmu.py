import torch
import transformers
from model import GPTConfig, GPT, MaskConfig
from dist_dataloader import DistributedDataLoader
import wandb  # Import wandb

import os
from contextlib import nullcontext
import tiktoken
import tqdm

# Initialize wandb
wandb.init(project="gpt2-rmu", name="gpt2-rmu-1")

path = "model_to_rmu.pt"

# -----------------------------------------------------------------------------
init_from = (
    "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
)
out_dir = "out"  # ignored if init_from is not 'resume'
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = False  # don't want to compile since we're doing funky stuff
# -----------------------------------------------------------------------------

# Log hyperparameters
wandb.config.update(
    {
        "init_from": init_from,
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "compile": compile,
    }
)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

ckpt_path = os.path.join(out_dir, path)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint["model_args"])
gptconf.l1_coeff = 0.0  # don't want l1 loss here; only crossentropy
print(gptconf)
mask_config = MaskConfig(**checkpoint["mask_config"])
model = GPT(gptconf, mask_config)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)
n_embd = model.config.n_embd

weight_decay = 0.1
learning_rate = 6e-4 / 10  # divide by 10 b/c finetuning
beta1 = 0.9
beta2 = 0.95
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
model.to(device)
if compile:
    print("compiling model...")
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# Log model configuration
wandb.config.update(
    {
        "n_embd": n_embd,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
    }
)

# clone the model to be used as the frozen model
frozen_model = GPT(gptconf, mask_config)
frozen_model.load_state_dict(state_dict, strict=False)
frozen_model.to(device)
# freeze the frozen model
for param in frozen_model.parameters():
    param.requires_grad = False

# look for the meta pickle in case it is available in the dataset folder
if gptconf.vocab_size == 151936:
    print("using qwen tokenizer")
    enc = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
elif gptconf.vocab_size == 50304:
    print("using gpt2 tokenizer")
    enc = tiktoken.get_encoding("gpt2")
else:
    raise ValueError(f"no tokenizer found for {vocab_size=}")
encode = enc.encode
decode = enc.decode
tokenizer = enc

T = 1024
batch_size = 4
gradient_accumulation_steps = 16
forget_set_train_dataloader = DistributedDataLoader(
    "data/fineweb-edu/wmdp_to_eval_on.bin", batch_size, T, 0, 1
)

coherence_set_dataloader = DistributedDataLoader(
    "data/fineweb-edu/train.bin", batch_size, T, 0, 1
)

# Log dataloader configuration
wandb.config.update(
    {
        "T": T,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
)


def get_batch(split):
    if split == "coherence":
        dataloader = coherence_set_dataloader
    elif split == "forget":
        dataloader = forget_set_train_dataloader
    else:
        raise ValueError(f"split must be 'coherence' or 'forget', got {split}")

    x, y = dataloader.next_batch()

    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x, y = (
        x.pin_memory().to(device, non_blocking=True),
        y.pin_memory().to(device, non_blocking=True),
    )
    return x, y


mask_ids = torch.full((batch_size, T), 1).to(device)  # don't do anything crazy

control_vector = torch.rand(1, 1, n_embd, device=device) * 40

layers_to_rmu_on = [14, 15, 16]
layer_to_stop_on = max(layers_to_rmu_on)
alpha = 150  # just from RMU code

# Log RMU configuration
wandb.config.update(
    {
        "layers_to_rmu_on": layers_to_rmu_on,
        "layer_to_stop_on": layer_to_stop_on,
        "alpha": alpha,
    }
)

# set all params frozen
for param in model.parameters():
    param.requires_grad = False
# set the params of the layers we want to rmu on to be trainable
for layer in layers_to_rmu_on:
    for param in model.transformer.h[layer].mlp.c_proj.parameters():
        param.requires_grad = True

for i in (pbar := tqdm.tqdm(range(1000))):
    for grad_accum_step in range(gradient_accumulation_steps):
        x_f, _ = get_batch("forget")
        x_r, _ = get_batch("coherence")
        with ctx:
            # forget
            forget_layer_to_stop_on_acts = model(
                x_f, mask_ids=mask_ids, stop_at_layer=layer_to_stop_on
            )
            unlearn_loss = torch.nn.functional.mse_loss(
                forget_layer_to_stop_on_acts, control_vector
            )

            # retain
            retain_layer_to_stop_on_acts = model(
                x_r, mask_ids=mask_ids, stop_at_layer=layer_to_stop_on
            )
            with torch.no_grad():
                retain_layer_to_stop_on_good_acts = frozen_model(
                    x_r, mask_ids=mask_ids, stop_at_layer=layer_to_stop_on
                )
            retain_loss = torch.nn.functional.mse_loss(
                retain_layer_to_stop_on_acts, retain_layer_to_stop_on_good_acts
            )

            loss = unlearn_loss + retain_loss * alpha
            loss = loss / gradient_accumulation_steps

        loss.backward()
        losses_dict = {
            "loss": loss.item(),
            "unlearn_loss": unlearn_loss.item(),
            "retain_loss": retain_loss.item(),
        }
        print(losses_dict)
        pbar.set_postfix(losses_dict)

        # Log losses to wandb
        wandb.log(losses_dict)

    optimizer.step()
    optimizer.zero_grad()

# Log final model to wandb
# wandb.save(os.path.join(out_dir, path))

# Finish the wandb run
wandb.finish()

# save the model
final_name = "model_to_rmu_rmued.pt"
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "model_args": checkpoint["model_args"],
    "mask_config": mask_config.__dict__,
    "config": GPTConfig(**checkpoint["model_args"]),
}
torch.save(checkpoint, os.path.join(out_dir, final_name))
