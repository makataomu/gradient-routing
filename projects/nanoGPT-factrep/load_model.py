from model import GPTConfig, GPT, MaskConfig
from typing import Tuple, Callable
import torch


def load_model_for_inference(
    compile: bool = True, path: str = "ckpt.pt"
) -> Tuple[GPT, MaskConfig, Callable, Callable, Callable, torch.device]:
    import os
    import pickle
    from contextlib import nullcontext
    import torch
    import transformers
    import tiktoken

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
    )  # 'float32' or 'bfloat16' or 'float16'
    compile = compile  # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
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

    # model
    if init_from == "resume":
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        # TODO save the mask config in the checkpoint
        gptconf = GPTConfig(**checkpoint["model_args"])
        print(gptconf)
        mask_config = MaskConfig(**checkpoint["mask_config"])
        model = GPT(gptconf, mask_config)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith("gpt2"):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        print("compiling model...")
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if (
        init_from == "resume"
        and "config" in checkpoint
        and "dataset" in checkpoint["config"]
    ):  # older checkpoints might not have these...
        meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda li: "".join([itos[i] for i in li])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
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
    model.config.l1_coeff = 0.0 # we don't want any l1 loss when doing evals
    return model, mask_config, encode, decode, ctx, device
