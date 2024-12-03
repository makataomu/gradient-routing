# %%
import json
import random
import time
from typing import List, Optional

import numpy as np
import nvidia_smi
import requests
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def set_random_seeds():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)


def get_gpu_with_most_memory(
    gpus_to_limit_to: Optional[list[int]] = None,
) -> torch.device:
    if not torch.cuda.is_available():
        if torch.backends.mps.is_available():  # mac native gpu
            return torch.device("mps")
        return torch.device("cpu")
    nvidia_smi.nvmlInit()
    device_count = nvidia_smi.nvmlDeviceGetCount()
    max_free_memory = 0
    chosen_device = 0

    gpu_ids = (
        range(device_count)
        if gpus_to_limit_to is None
        else [id for id in gpus_to_limit_to if id < device_count]
    )
    for i in gpu_ids:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        if info.free > max_free_memory:
            max_free_memory = info.free
            chosen_device = i

    nvidia_smi.nvmlShutdown()
    return torch.device(f"cuda:{chosen_device}")


class Timer:
    def __init__(self, num_tasks):
        self.start_time = time.time()
        self.num_tasks = num_tasks
        self.current_task = 0

    def increment(self):
        self.current_task += 1
        elapsed_hours = (time.time() - self.start_time) / (60 * 60)
        time_per_task = elapsed_hours / self.current_task
        time_remaining = time_per_task * (self.num_tasks - self.current_task)
        print("-" * 80)
        print(
            f"Finished task {self.current_task}/{self.num_tasks} - "
            f"Elapsed: {elapsed_hours:.2f} hours - "
            f"Remaining: {time_remaining:.2f} hours"
        )
        print("-" * 80, flush=True)


def upload_to_clbin(text: str) -> Optional[str]:
    try:
        r = requests.post("https://clbin.com", data={"clbin": text}, timeout=5)
        return r.content.decode("utf-8").strip()
    except Exception:
        try:
            # try paste.rs
            r = requests.post("https://paste.rs", data=text)
            return r.content.decode("utf-8").strip()
        except Exception:
            return None


def get_cross_entropy(logits, other_logits):
    return (-F.softmax(logits, dim=-1) * F.log_softmax(other_logits, dim=-1)).sum(
        dim=-1
    )


def count_token_in_text(model: HookedTransformer, token: str, text: str) -> int:
    tl_model = model if isinstance(model, HookedTransformer) else model.transformer
    tokenized = tl_model.tokenizer.tokenize(text)  # type: ignore
    return sum([1 for t in tokenized if t == token])


def count_str_in_text(string: str, text: str) -> int:
    return text.lower().count(string.lower())


def word_variations(word: List[str]) -> List[str]:
    variations = []
    for w in word:
        variations.append(w.lower())
        variations.append(w.capitalize())
        variations.append(" " + w.lower())
        variations.append(" " + w.capitalize())
    return variations


def get_first_row_and_delete(filename: str) -> dict | None:
    with open(filename, "r") as f:
        lines = f.readlines()
        if not lines:
            return None
        first_row = lines[0].strip()
        if not first_row:
            return None
    with open(filename, "w") as f:
        f.writelines(lines[1:])
    return json.loads(first_row)
