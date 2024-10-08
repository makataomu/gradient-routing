import os
from contextlib import nullcontext

import load_special_text
import tiktoken
import torch
import tqdm
import transformers
from dist_dataloader import DistributedDataLoader
from model import GPT, GPTConfig, MaskConfig

# Hyperparameters and configuration
out_dir = "out"
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True  # use PyTorch 2.0 to compile the model to be faster
batch_size = 1
T = 1024
weight_decay = 0.1
learning_rate = 6e-4 / 10  # divide by 10 b/c finetuning
beta1 = 0.9
beta2 = 0.95


def evaluate_and_retrain_model(model_path, model_is_rmu, device):
    # Set up device and context
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
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

    # Load model
    ckpt_path = os.path.join(out_dir, model_path)
    print(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    gptconf.l1_coeff = 0.0
    print(gptconf)
    mask_config = MaskConfig(**checkpoint["mask_config"])
    model = GPT(gptconf, mask_config)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)

    # Configure model
    n_embd = model.config.n_embd
    max_dim_to_mask = (
        max(model.mask_config.mlp_dims_to_mask[0])
        if isinstance(model.mask_config.mlp_dims_to_mask[0], list)
        else max(model.mask_config.mlp_dims_to_mask[0][0])
    )
    new_mlp_dims_to_mask = [
        list(range(0, max_dim_to_mask)),
        list(range(0, 4 * n_embd)),
        list(range(max_dim_to_mask, 4 * n_embd)),
    ]
    model.mask_config.mlp_dims_to_mask = new_mlp_dims_to_mask
    model._compute_and_register_masks()

    # Setup optimizer
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    model.to(device)
    if compile:
        print("Compiling the model...")
        model = torch.compile(model)

    # Setup tokenizer
    if gptconf.vocab_size == 151936:
        print("Using qwen tokenizer")
        enc = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    elif gptconf.vocab_size == 50304:
        print("Using gpt2 tokenizer")
        enc = tiktoken.get_encoding("gpt2")
    else:
        raise ValueError(f"No tokenizer found for vocab_size={gptconf.vocab_size}")

    # Setup data loaders
    forget_set_train_dataloader = DistributedDataLoader(
        "data/fineweb-edu/wmdp_to_train_on.bin", batch_size, T, 0, 1
    )
    coherence_set_dataloader = DistributedDataLoader(
        "data/fineweb-edu/train.bin", batch_size, T, 0, 1
    )

    # Evaluation function
    def eval_model():
        print("Evaluating model...")
        val_loss = load_special_text.get_val_loss(
            model, enc, "data/fineweb-edu/val.bin", device, ablate_idx=None
        )
        print(f"Val loss: {val_loss}")
        virology_loss = load_special_text.get_val_loss(
            model, enc, "data/fineweb-edu/wmdp_to_eval_on.bin", device, ablate_idx=None
        )
        print(f"Virology loss: {virology_loss}")
        virology_loss_without_masked_tokens = load_special_text.get_val_loss(
            model,
            enc,
            "data/fineweb-edu/wmdp_to_eval_on.bin",
            device,
            ablate_idx=None,
            include_masked_tokens=False,
        )
        print(
            f"Virology loss without masked tokens: {virology_loss_without_masked_tokens}"
        )
        return {
            "val_loss": val_loss,
            "virology_loss": virology_loss,
            "virology_loss_without_masked_tokens": virology_loss_without_masked_tokens,
        }

    # Initial evaluation
    print("Performing initial evaluation...")
    initial_losses = eval_model()

    # Ablation if needed
    if not model_is_rmu:
        print("ABLATING!")
        for layer in mask_config.blocks_to_mask:
            model.transformer.h[layer].mlp.c_fc.weight.data[:max_dim_to_mask, 0] = 0
            model.transformer.h[layer].mlp.c_fc.bias.data[:max_dim_to_mask] = 0
            model.transformer.h[layer].mlp.c_proj.weight.data[:, :max_dim_to_mask] = 0

        print("Performing post-ablation evaluation...")
        post_ablation_losses = eval_model()
    else:
        post_ablation_losses = None

    # Coherence finetuning (commented out)
    print("Starting coherence finetuning...")
    gradient_accumulation_steps = 128
    for i in range(32):
        for j in tqdm.tqdm(range(gradient_accumulation_steps)):
            x, y = coherence_set_dataloader.next_batch()
            x, y = x.to(device), y.to(device)
            mask_ids = torch.full((batch_size, T), 2 if not model_is_rmu else 1).to(
                device
            )
            with ctx:
                _, loss = model(x, y, mask_ids=mask_ids)
            loss = loss / gradient_accumulation_steps
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    gradient_accumulation_steps = 8 // batch_size
    # Retraining
    print("Starting retraining...")
    lowest_losses = eval_model()
    for i in tqdm.tqdm(range(20)):
        for _ in range(gradient_accumulation_steps):
            x, y = forget_set_train_dataloader.next_batch()
            x, y = x.to(device), y.to(device)
            mask_ids = torch.full((batch_size, T), 2 if not model_is_rmu else 1).to(
                device
            )

            with ctx:
                _, loss = model(x, y, mask_ids=mask_ids)
            loss = loss / gradient_accumulation_steps
            loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (i + 1) % 5 == 0:
            print(f"\nEvaluating after {i+1} retraining iterations...")
            current_losses = eval_model()
            for key in lowest_losses:
                lowest_losses[key] = min(lowest_losses[key], current_losses[key])

    print("Evaluation and retraining completed.")
    return {
        "initial_losses": initial_losses,
        "post_ablation_losses": post_ablation_losses,
        "lowest_retrain_losses": lowest_losses,
    }


if __name__ == "__main__":
    model_is_rmu = False
    path = (
        "ckpt_last-try-b4-iclr-7_10000.pt"
        if not model_is_rmu
        else "model_to_rmu_rmued.pt"
    )

    results = evaluate_and_retrain_model(path, model_is_rmu, device)

    print("\nFinal Results:")
    print("Initial Losses:", results["initial_losses"])
    if results["post_ablation_losses"]:
        print("Post-Ablation Losses:", results["post_ablation_losses"])
    print("Lowest Retrain Losses:", results["lowest_retrain_losses"])

    print("\nLowest losses achieved:")
    print("Lowest val loss:", results["lowest_retrain_losses"]["val_loss"])
    print("Lowest virology loss:", results["lowest_retrain_losses"]["virology_loss"])
    print(
        "Lowest virology loss without masked tokens:",
        results["lowest_retrain_losses"]["virology_loss_without_masked_tokens"],
    )
