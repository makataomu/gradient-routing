"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from rope import RotaryEmbedding
from torch.nn import functional as F


class CausalGroupedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_rope = config.use_rope
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_key_value_head = config.n_key_value_head
        self.d_head = config.n_embd // config.n_head

        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_key_value_head == 0

        self.dropout = config.dropout

        self.c_attn_q = nn.Linear(
            config.n_embd, 1 * config.n_embd, bias=config.attn_bias
        )
        self.c_attn_kv = nn.Linear(
            config.n_embd,
            2 * config.n_key_value_head * self.d_head,
            bias=config.attn_bias,
        )  # * 2 b/c we want key and value projs
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)

    def forward(self, x, rotary=None, mask=None):
        assert mask is None, "Attention masking not yet supported"
        B, T, C = x.size()

        q = self.c_attn_q(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k, v = self.c_attn_kv(x).split(self.n_key_value_head * self.d_head, dim=2)
        k = k.view(B, T, self.n_key_value_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_key_value_head, self.d_head).transpose(1, 2)
        if rotary is not None:
            q = rotary.rotate_queries_or_keys(q)
            k = rotary.rotate_queries_or_keys(k)
        # from https://pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
        # we are doing this instead of using the builtin grouped query attn support because of a torch bug that means we can't use nightly on H100s
        # k = k.repeat_interleave(q.size(-3) // k.size(-3), -3)
        # v = v.repeat_interleave(q.size(-3) // v.size(-3), -3)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
            enable_gqa=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.attn_bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x, rotary=None, mask=None):
        assert mask is None, "Attention masking not yet supported"
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        if rotary is not None:
            q = rotary.rotate_queries_or_keys(q)
            k = rotary.rotate_queries_or_keys(k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        x = self.c_fc(x)
        x = F.gelu(x)
        if mask is not None:
            x = x * mask + (1.0 - mask) * x.detach()
            l1_penalty = x.abs().sum()  # only penalize when we're masking
        else:
            l1_penalty = torch.tensor(0.0).to(x.device)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x, l1_penalty

    def forward_ablated(self, x, mask):
        x = self.c_fc(x)
        x = F.gelu(x)
        if mask is not None:
            x = x * (1.0 - mask)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if eps is None:
            self.eps = (
                1e-6  # https://huggingface.co/Qwen/Qwen2-1.5B/blob/main/config.json#
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        x = x / scale
        return x * self.weight


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rmsnorm_1 = RMSNorm(config.n_embd)
        if hasattr(config, "n_key_value_head") and config.n_key_value_head is not None:
            self.attn = CausalGroupedSelfAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        self.rmsnorm_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, masks_by_type, rotary):
        resid_mask = masks_by_type.get("resid")
        x = x + self.attn(
            self.rmsnorm_1(x), mask=masks_by_type.get("attn"), rotary=rotary
        )
        mlp_out, l1_penalty = self.mlp(self.rmsnorm_2(x), masks_by_type.get("mlp"))
        x = x + mlp_out
        if resid_mask is not None:
            x = resid_mask * x + (1 - resid_mask) * x.detach()
        return x, l1_penalty

    def forward_ablated(self, x, masks_by_type, rotary):
        resid_mask = masks_by_type.get("resid")
        x = x + self.attn(
            self.rmsnorm_1(x), mask=masks_by_type.get("attn"), rotary=rotary
        )
        x = x + self.mlp.forward_ablated(self.rmsnorm_2(x), masks_by_type.get("mlp"))
        if resid_mask is not None:
            x = (1 - resid_mask) * x
        return x


"""
Pipeline:
    masking_rule: (batch data) -> mask_ids
    mask_lookups: mask_ids -> masks_by_type :

"""


@dataclass
class MaskConfig:
    blocks_to_mask: list[int]

    mlp_dims_to_mask: list[list[int]]
    attn_dims_to_mask: list[list[int]]
    resid_dims_to_mask: list[list[int]]

    # only one of these should be present, other should be None
    tokens_to_mask: dict[str, int]
    seqs_to_mask: list[tuple[list[str], int]] = None


def get_multi_hot(dims: list[int], length: int):
    mask = torch.zeros(length)
    mask[dims] = 1
    return mask


def get_multi_hot_different_lrs(
    dims: list[int], length: int, target_lr: float, off_target_lr: float
):
    mask = torch.full((length,), off_target_lr)
    mask[dims] = target_lr
    return mask


def _precompute_mask_lookups(config, mask_config):
    """
    Return:
        all_masks - a dict keyed by "mlp", "resid", or "attn", which
        contains a tensor of shape (n_possible_masks, mask_shape) for each mask type.
    """
    assert len(mask_config.attn_dims_to_mask) == 0, "Attn not yet supported"

    dims_to_mask = {
        "mlp": mask_config.mlp_dims_to_mask,
        "resid": mask_config.resid_dims_to_mask,
        "attn": mask_config.attn_dims_to_mask,
    }
    dim_sizes = {"mlp": 4 * config.n_embd, "resid": config.n_embd, "attn": None}

    all_masks = {}
    for mask_type, mask_idx_to_dims in dims_to_mask.items():
        if mask_idx_to_dims is not None and len(mask_idx_to_dims) > 0:
            masks_for_dims = []
            for dims in mask_idx_to_dims:
                if isinstance(dims, list):
                    masks_for_dims.append(get_multi_hot(dims, dim_sizes[mask_type]))
                elif isinstance(dims, tuple):
                    # we case this differently so that we can load old mask configs with models
                    # before we implemented the change
                    dimensions, target_lr, off_target_lr = dims
                    multi_hot = get_multi_hot_different_lrs(
                        dimensions, dim_sizes[mask_type], target_lr, off_target_lr
                    )
                    masks_for_dims.append(multi_hot)
                else:
                    raise ValueError("the dims either need to be a tuple or a list")

            all_masks[mask_type] = torch.stack(masks_for_dims)

    return all_masks


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_key_value_head: int = None
    n_embd: int = 768
    dropout: float = 0.0
    l1_coeff: float = 0.0  # How much should we penalize MLP acts?
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attn_bias: bool = True
    tie_weights: bool = False
    use_pos_emb: bool = True
    use_rope: bool = False
    use_pos_emb: bool = True
    use_rope: bool = False
    use_conditional_bias: bool = False


class GPT(nn.Module):
    def __init__(self, config, mask_config: MaskConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.mask_config = mask_config
        self._compute_and_register_masks()

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
            | (
                dict(
                    wpe=nn.Embedding(config.block_size, config.n_embd),
                )
                if self.config.use_pos_emb
                else {}
            )
        )
        self.rotary = (
            RotaryEmbedding((config.n_embd // config.n_head) // 2)
            if self.config.use_rope
            else None
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:  # for backwards compatability
            self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )
        if config.use_conditional_bias:
            self.conditional_bias = nn.Parameter(
                torch.zeros(config.n_embd), requires_grad=True
            )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _compute_and_register_masks(self):
        """Register so will be moved to GPU"""
        all_masks = _precompute_mask_lookups(self.config, self.mask_config)
        for mask_type, mask_stack in all_masks.items():
            self.register_buffer(f"{mask_type}_mask", mask_stack)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if self.config.use_pos_emb:
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx, targets=None, mask_ids=None, reduce_loss=True, stop_at_layer=None
    ):
        device = idx.device
        b, t = idx.size()
        l1_penalty = torch.tensor(0.0).to(device)
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        if mask_ids is not None:
            all_masks = {
                name.replace("_mask", ""): buf
                for name, buf in self.named_buffers()
                if name.endswith("_mask")
            }
            masks_by_type = {
                mask_type: mask_stack[mask_ids]
                for mask_type, mask_stack in all_masks.items()
            }
        else:
            masks_by_type = {}
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.config.use_pos_emb:
            pos_emb = self.transformer.wpe(
                pos
            )  # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        for block_idx, block in enumerate(self.transformer.h):
            block_masks = (
                masks_by_type if block_idx in self.mask_config.blocks_to_mask else {}
            )
            x, l1_penalty_for_block = block(x, block_masks, rotary=self.rotary)
            if block_idx == stop_at_layer:
                return x
            if self.config.l1_coeff != 0.0:
                l1_penalty += l1_penalty_for_block

        x = self.transformer.ln_f(x)
        if self.config.use_conditional_bias:
            token_positions_masking = mask_ids == 0
            x += self.conditional_bias * token_positions_masking.float().unsqueeze(-1)
        logits = self.lm_head(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction="mean" if reduce_loss else "none",
            )
            if self.config.l1_coeff > 0.0:
                tokens_per_batch = x.shape[0] * x.shape[1]
                loss += l1_penalty * self.config.l1_coeff / tokens_per_batch
        else:
            loss = None

        return logits, loss

    def forward_ablated(self, idx, targets=None, ablate_idx: int = 0, reduce_loss=True):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        all_masks = {
            name.replace("_mask", ""): buf
            for name, buf in self.named_buffers()
            if name.endswith("_mask")
        }

        masks_by_type = {
            mask_type: mask_stack[
                torch.full((b, t), ablate_idx, dtype=torch.long, device=device)
            ]
            for mask_type, mask_stack in all_masks.items()
        }
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.config.use_pos_emb:
            pos_emb = self.transformer.wpe(
                pos
            )  # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        for block_idx, block in enumerate(self.transformer.h):
            block_masks = (
                masks_by_type if block_idx in self.mask_config.blocks_to_mask else {}
            )
            x = block.forward_ablated(x, block_masks, rotary=self.rotary)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction="mean" if reduce_loss else "none",
            )
        else:
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, top_k=None, ablate_idx=None
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            if ablate_idx is not None:
                logits, _ = self.forward_ablated(idx_cond, ablate_idx=ablate_idx)
            else:
                logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
