"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import datetime
import math
import os
import pickle
import time
from contextlib import nullcontext

import coherence_ft_and_retrain_evals
import load_special_text
import torch
import transformers
from dist_dataloader import DistributedDataLoader
from eval import eval_nano_tasks
from jaxtyping import Float
from model import GPT, GPTConfig, MaskConfig
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on fineweb-edu
# I/O
out_dir = "out"
eval_interval = 1000
log_interval = 50
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
do_mc_evals = False  # if True, do multiple choice evals
do_retrain_evals = True
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True
wandb_project = "pretrain-gpt2"
wandb_run_name = "test for retrain evals in the run"
custom_path = "data/virology.txt"
# data
dataset = "fineweb-edu"
gradient_accumulation_steps = 20 * 8  # used to simulate larger batch sizes
# gradient_accumulation_steps = 8 # TODO change
batch_size = 8  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 20
n_head = 12
n_key_value_head = 2
n_embd = 1536
l1_coeff = 1e-7
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
tie_weights = True
use_pos_embd = False
use_rotary_embeddings = True
bias = True  # do we use bias inside LayerNorm and Linear (not including attn) layers?
attn_bias = True
use_conditional_bias = False
# AdamW optimizer
learning_rate = 3.0 * 6e-4  # max learning rate
max_iters = 60_000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
use_masking = True

distill = False

sprinkle_wmdp_in_nx = 5  # this determines what data file we use
#sprinkle_wmdp_in_nx = 0  # this determines what data file we use

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")


# mask index 0 is for forget, mask index 1 is for retain
def get_token_masking_rule(toks_to_mask: dict[str, int], tokenizer, device):
    no_masking_mask_idx = len(set(toks_to_mask.values()))
    token_to_mask = torch.full(
        (tokenizer.eos_token_id + 1,),
        no_masking_mask_idx,
        dtype=torch.long,
        device=device,
    )

    for token, mask_idx in toks_to_mask.items():
        encoded = tokenizer.encode(token)
        assert (
            len(encoded) == 1
        ), f"All tokens to mask must be a single token, but {token} is not."
        encoded = encoded[0]
        token_to_mask[encoded] = mask_idx

    def token_masking_rule(
        correct_toks: Float[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch seq"]:
        return token_to_mask[correct_toks]

    return token_masking_rule


# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(
        backend=backend, timeout=datetime.timedelta(minutes=120)
    )  # the master process will be doing evals that take a while so we give a lot of time
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_local_rank = 0


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if master_process:
        print(*args, **kwargs)


# tokens_to_mask = {
#    " virus": 0,
#    " viruses": 0,
#    "COVID": 0,
#    " hepatitis": 0,
#    " COVID": 0,
#    " cell": 0,
#    " cells": 0,
#    " disease": 0,
#    " diseases": 0,
#    " infect": 0,
# }
# manually added COVIDs to the below and reomved a bunch that could be ambigious
tokens_to_mask = {
    "COVID": 0,
    " COVID": 0,
    "RNA": 0,
    " infections": 0,
    "DNA": 0,
    " genome": 0,
    " virus": 0,
    " gene": 0,
    " viruses": 0,
    " mutations": 0,
    " antibodies": 0,
    " influenza": 0,
    " bacteria": 0,
    "PCR": 0,
    " cell": 0,
    " herpes": 0,
    " bacterial": 0,
    " pathogens": 0,
    " tumor": 0,
    " vaccine": 0,
}
mask_rule = get_token_masking_rule(
    tokens_to_mask,
    tokenizer,
    torch.device(device),
)
mask_config = MaskConfig(
    blocks_to_mask=[
        #0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        #3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
        0, 1, 2, 3, 4, 5, 6, 7
    ],
    mlp_dims_to_mask=[
        (list(range(0, 80)), 1.0, -5e-8),
        (list(range(0, 4 * n_embd)), 1.0, 0.0),
    ],  # dimensions, target_lr, off_target_lr
    # mlp_dims_to_mask=[list(range(0, 256)), list(range(0, 4 * n_embd))],
    attn_dims_to_mask=[],
    resid_dims_to_mask=[],
    tokens_to_mask=tokens_to_mask,
)
print(
    f"{ddp_world_size=}, {gradient_accumulation_steps=}, {batch_size=}, {block_size=}"
)
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
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

if distill:
    import transformers

    teacher_model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
    teacher_model.to(device)


# poor man's data loader
data_dir = os.path.join("data", dataset)
file_name_str = lambda split: (
    f"{split}.bin"
    if sprinkle_wmdp_in_nx < 1
    else f"{split}_wmdp_{sprinkle_wmdp_in_nx}_big.bin"
)
train_split_file = os.path.join(data_dir, file_name_str("train"))
val_split_file = os.path.join(data_dir, file_name_str("val"))
if master_process:
    print(f"LOADING DATA FROM {train_split_file} and {val_split_file}")


train_dataloader = DistributedDataLoader(
    train_split_file, batch_size, block_size, ddp_local_rank, ddp_world_size
)
val_dataloader = DistributedDataLoader(
    val_split_file, batch_size, block_size, ddp_local_rank, ddp_world_size
)


def get_batch(split):
    if split == "train":
        dataloader = train_dataloader
    else:
        dataloader = val_dataloader

    x, y = dataloader.next_batch()

    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_key_value_head=n_key_value_head,
    n_embd=n_embd,
    l1_coeff=l1_coeff,
    tie_weights=True,
    block_size=block_size,
    bias=bias,
    attn_bias=attn_bias,
    vocab_size=None,
    dropout=dropout,
    use_pos_emb=use_pos_embd,
    use_rope=use_rotary_embeddings,
    use_conditional_bias=use_conditional_bias,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of Qwen-2 to 151936 (151643 rounded up for efficiency)"
        )
    model_args["vocab_size"] = (
        meta_vocab_size if meta_vocab_size is not None else 151936
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, mask_config)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap with DDP before compilation for speedup
# https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.inference_mode()
def eval_on_mcqs(model):
    evals = [
        "wmdp_bio",
        "wmdp_bio_continuation",
        "mmlu",
        "mmlu_virology",
        "mmlu_computer_security",
        "mmlu_college_computer_science",
        "mmlu_college_biology",
        "mmlu_continuation_virology",
        "mmlu_continuation_computer_security",
        "mmlu_continuation_college_computer_science",
        "mmlu_continuation_college_biology",
    ]
    results = eval_nano_tasks(
        model, device, tokenizer.encode, tokenizer.decode, evals, ablate=None
    )
    results_ablated = eval_nano_tasks(
        model, device, tokenizer.encode, tokenizer.decode, evals, ablate=0
    )

    res = {}
    for eval in results.keys():
        res[eval] = results[eval]["acc,none"]
    for eval in results_ablated.keys():
        res["ablated_" + eval] = results_ablated[eval]["acc,none"]
    return res


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(also_do_ablate: bool):
    out = {}
    out_ablated = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        losses_ablated = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y, mask_rule(Y))
                losses[k] = loss.item()
                if also_do_ablate:
                    ablated_logits, ablated_loss = raw_model.forward_ablated(
                        X, Y, ablate_idx=torch.tensor(0)
                    )
                    losses_ablated[k] = ablated_loss.item()
        out[split] = losses.mean()
        if also_do_ablate:
            out_ablated[split] = losses_ablated.mean()
    model.train()
    if also_do_ablate:
        return out, out_ablated
    else:
        return out


@torch.inference_mode()
def estimate_custom_loss(custom_path: str, ablate_idx=None):
    model.eval()
    with ctx:
        loss = load_special_text.eval_on_special_batch(
            raw_model,
            custom_path,
            batch_size,
            block_size,
            device,
            include_masked_tokens=True,
            ablate_idx=ablate_idx,
        )
    model.train()
    return loss


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    project_dir = os.path.dirname(os.path.abspath(__file__))
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=config,
        settings=wandb.Settings(code_dir=project_dir),
        dir=project_dir,
    )
    wandb.run.log_code(  # type: ignore
        project_dir,
    )

# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        torch.cuda.empty_cache()
        losses, ablated_losses = estimate_loss(also_do_ablate=True)
        custom_loss_unablated = estimate_custom_loss(custom_path)
        custom_loss_ablated = estimate_custom_loss(
            custom_path, ablate_idx=torch.tensor(0)
        )
        if do_mc_evals:
            mcq_results = eval_on_mcqs(raw_model)
            if wandb_log:
                wandb.log(mcq_results)
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, custom loss {custom_loss_unablated:.4f}"
        )
        print(
            f"step {iter_num}: train loss ablated {ablated_losses['train']:.4f}, val loss ablated {ablated_losses['val']:.4f}, custom loss ablated {custom_loss_ablated:.4f}"
        )

        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "concept/loss": custom_loss_unablated,
                    "train/loss_ablate": ablated_losses["train"],
                    "val/loss_ablate": ablated_losses["val"],
                    "concept/loss_ablate": custom_loss_ablated,
                    "lr": lr,
                },
                step=iter_num,
            )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "mask_config": mask_config.__dict__,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            print(f"saving checkpoint to {out_dir}")
            ckpt = f"ckpt_{wandb_run_name}_{iter_num}.pt"
            path_to_save = os.path.join(out_dir, ckpt)
            torch.save(checkpoint, path_to_save)
            if do_retrain_evals:
                torch.cuda.empty_cache()
                retrain_results = (
                    coherence_ft_and_retrain_evals.evaluate_and_retrain_model(
                        ckpt, model_is_rmu=False, device=device
                    )
                )
                # add prefix before logging
                if retrain_results["post_ablation_losses"]:
                    if wandb_log:
                        wandb.log(
                            {
                                f"post_ablation/{k}": v
                                for k, v in retrain_results[
                                    "post_ablation_losses"
                                ].items()
                            },
                            step=iter_num,
                        )
                        # do for 'lowest_retrain_losses' as well
                        wandb.log(
                            {
                                f"lowest_retrain/{k}": v
                                for k, v in retrain_results[
                                    "lowest_retrain_losses"
                                ].items()
                            },
                            step=iter_num,
                        )
                        # 'initial_losses' as well
                        wandb.log(
                            {
                                f"initial/{k}": v
                                for k, v in retrain_results["initial_losses"].items()
                            },
                            step=iter_num,
                        )
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    los = 0
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        if use_masking:
            mask_idxs = mask_rule(Y)
        else:
            mask_idxs = None
        with ctx:
            if distill:
                with torch.inference_mode():
                    teacher_logits = teacher_model(
                        X, output_hidden_states=False, output_attentions=False
                    ).logits
                    log_teacher_logits = torch.log_softmax(teacher_logits, dim=-1)
                logits, _ = model(X, mask_ids=mask_idxs)
                log_logits = torch.log_softmax(logits, dim=-1)
                # https://arxiv.org/pdf/2407.14679 says forward KLD is best
                loss = (
                    log_teacher_logits.exp() * (log_teacher_logits - log_logits)
                ).sum()
                loss = (
                    loss / gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            else:
                logits, loss = model(X, Y, mask_idxs)
                loss = (
                    loss / gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
        los += loss.item()
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    if master_process and wandb_log:
        wandb.log({"loss_on_master_process_train": los}, step=iter_num)
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    del loss, mask_idxs, logits  # hopefully free memory maybe
    torch.cuda.empty_cache()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        k = 50
        # print the top k unembeds of the conditional bias
        if use_conditional_bias:
            print(
                f"top {k} unembeds of conditional bias:",
                [
                    tokenizer.decode(t)
                    for t in raw_model.lm_head(model.conditional_bias).topk(k).indices
                ],
            )
        # print the L1 norm of the MLP weights in all the blocks_to_mask
        mlp_l1_norm = 0.0
        for block in mask_config.blocks_to_mask:
            mlp = raw_model.transformer.h[block].mlp
            for param in mlp.parameters():
                mlp_l1_norm += param.abs().sum().item()
        print(
            f"iter {iter_num} done in {dt:.1f} sec: loss {los:.4f}, mlp_l1 {mlp_l1_norm:.4f}"
        )
        wandb.log({"mlp_l1_norm": mlp_l1_norm}, step=iter_num)

        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        # lossf = loss.item() * gradient_accumulation_steps
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
