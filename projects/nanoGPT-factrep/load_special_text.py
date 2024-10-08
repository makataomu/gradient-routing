import torch
from typing import Tuple
import transformers
from dist_dataloader import DistributedDataLoader
import tqdm


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
        correct_toks,
    ):
        return token_to_mask[correct_toks]

    return token_masking_rule


def load_special_batch(
    text_path: str, block_len: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    torch_tokens = torch.tensor(tokens, dtype=torch.long)
    x = torch.stack(
        [
            torch_tokens[i : i + block_len]
            for i in range(0, len(torch_tokens) - block_len, block_len)
        ]
    )
    y = torch.stack(
        [
            torch_tokens[i + 1 : i + 1 + block_len]
            for i in range(0, len(torch_tokens) - block_len, block_len)
        ]
    )
    x, y = (
        x.pin_memory().to(device, non_blocking=True),
        y.pin_memory().to(device, non_blocking=True),
    )
    return x, y


def eval_on_special_batch(
    model,
    text_path: str,
    batch_size: int,
    block_len: int,
    device,
    include_masked_tokens,
    ablate_idx=None,
) -> float:
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    rule = get_token_masking_rule(model.mask_config.tokens_to_mask, tokenizer, device)
    x, y = load_special_batch(text_path, block_len, device)
    # print("evaluating on", text_path, "with ablate_idx", ablate_idx)
    total_loss = 0
    num_batches = 0
    with torch.inference_mode():
        for i in tqdm.tqdm(range(0, len(x), batch_size)):
            batch_x = x[i : i + batch_size]
            batch_y = y[i : i + batch_size]
            loss_mask = rule(batch_y)
            if ablate_idx is not None:
                logits, loss = model.forward_ablated(
                    batch_x, batch_y, ablate_idx=ablate_idx, reduce_loss=False
                )
            else:
                logits, loss = model(
                    batch_x, batch_y, torch.ones_like(batch_y), reduce_loss=False
                )
            if not include_masked_tokens:
                loss = loss * loss_mask.flatten(0, 1)
                loss = loss.mean()
            else:
                loss = loss.mean()
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches


@torch.no_grad()
def get_val_loss(model, tokenizer, val_path, device, ablate_idx=None, include_masked_tokens = True):
    token_masking_rule = get_token_masking_rule(
        model.mask_config.tokens_to_mask, tokenizer, device
    )
    val_dataloader = DistributedDataLoader(val_path, 2, 1024, 0, 1)
    losses = []
    for i in tqdm.tqdm(range(300)):
        x, y = val_dataloader.next_batch()
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
        if ablate_idx is not None:
            logits, loss = model.forward_ablated(
                x, y, ablate_idx=ablate_idx, reduce_loss=False
            )
        else:
            logits, loss = model(x, y, torch.ones_like(y), reduce_loss=False)
        if not include_masked_tokens:
            loss = loss * token_masking_rule(y).flatten(0, 1)
            loss = loss.mean()
        else:
            loss = loss.mean()
        losses.append(loss.item())
    return sum(losses) / len(losses)


if __name__ == "__main__":
    import load_model

    model, mask_config, encode, decode, ctx, device = (
        load_model.load_model_for_inference(
            compile=True, path="best_virology_so_far.pt"
        )
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    print("using device ", device)
    # text = "data/virology.txt"
    text = "full_text_of_virology_and_children.txt"
    with ctx:
        print(
            "validation unablated: ",
            get_val_loss(model, tokenizer, "data/fineweb-edu/val.bin", device, ablate_idx=None),
        )                                                   
        print(                                              
            "validation ablated: ",                         
            get_val_loss(model, tokenizer, "data/fineweb-edu/val.bin", device, ablate_idx=0),
        )
        print(
            "virology unablated: ",
            eval_on_special_batch(
                model,
                text,
                2,
                1024,
                device,
                ablate_idx=None,
                include_masked_tokens=True,
            ),
        )
        print(
            "virology ablated: ",
            eval_on_special_batch(
                model,
                text,
                2,
                1024,
                device,
                ablate_idx=0,
                include_masked_tokens=True,
            ),
        )
        print(
            "virology unablated (not including masked tokens): ",
            eval_on_special_batch(
                model,
                text,
                2,
                1024,
                device,
                ablate_idx=None,
                include_masked_tokens=False,
            ),
        )
        print(
            "virology ablated (not including masked tokens): ",
            eval_on_special_batch(
                model,
                text,
                2,
                1024,
                device,
                ablate_idx=0,
                include_masked_tokens=False,
            ),
        )
