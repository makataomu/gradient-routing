# %%
import torch as t

import factored_representations.training as training


def test_get_cross_entropy_loss():
    batch, seq, d_vocab = 2, 7, 11
    logits = t.arange(batch * seq * d_vocab).reshape(batch, seq, d_vocab).float()
    logits[1, 0, :] = 0  # Try a different value
    labels = t.arange(batch * seq).reshape(batch, seq) % d_vocab
    cross_entropy = t.zeros((batch, seq))
    for batch_idx in range(batch):
        for seq_idx in range(seq):
            logit = logits[batch_idx, seq_idx]
            label = labels[batch_idx, seq_idx]
            probs = t.softmax(logit, dim=0)[label]
            cross_entropy[batch_idx, seq_idx] = -t.log(probs)

    # Test unmasked
    loss = training.get_cross_entropy_loss(logits, labels, mask=t.ones((batch, seq)))
    assert t.allclose(loss, cross_entropy.mean())

    # Test with padding
    padding = {
        "left": [0, 1],
        "right": [seq - 2, seq - 1],
        "both": [0, 1, seq - 2, seq - 1],
        "haphazard": t.arange(seq) % 3 == 0,
    }
    for label, pad_indices in padding.items():
        mask = t.ones((batch, seq), dtype=t.bool)
        mask[:, pad_indices] = 0
        loss = training.get_cross_entropy_loss(logits, labels, mask=mask)
        assert t.allclose(
            loss, cross_entropy[mask].mean()
        ), f"Mismatch for padding={label}"

    # Test with different padding per batch item
    mask = t.ones((batch, seq), dtype=t.bool)
    mask[0, padding["left"]] = 0
    mask[1, padding["right"]] = 0
    loss = training.get_cross_entropy_loss(logits, labels, mask=mask)
    assert t.allclose(
        loss, cross_entropy[mask].mean()
    ), "Mismatch for different mask patterns by batch element"


def test_compute_preds_and_get_cross_entropy_loss(tokenizer):
    for side in ["left", "right"]:
        tokenizer.padding_side = side

        out = tokenizer(
            ["Four gigachads", "hello"],
            padding=True,
            return_tensors="pt",
        )
        tokens: t.Tensor = out["input_ids"]  # type: ignore
        attn_mask: t.Tensor = out["attention_mask"]  # type: ignore
        print(f"{attn_mask}")

        batch, seq = tokens.shape

        perfect_logits_when_masked = t.full(
            (batch, seq - 1, tokenizer.vocab_size), -1e5
        )
        for batch_idx in range(batch):
            for seq_input_idx in range(seq - 1):
                next_token = tokens[batch_idx, seq_input_idx + 1]
                mask = attn_mask[batch_idx, seq_input_idx]
                if mask:
                    perfect_logits_when_masked[batch_idx, seq_input_idx, next_token] = 0
                else:
                    # Add logit for incorrect token.
                    # If masking is done correctly, this should be ignored.
                    perfect_logits_when_masked[batch_idx, seq_input_idx, 47] = 5

        def mocked_model(x, attention_mask):
            return perfect_logits_when_masked

        loss = training.compute_preds_and_get_ce_loss(
            model=mocked_model, tokens=tokens, attention_mask=attn_mask, other_mask=None
        )
        assert loss == 0, f"Expected loss=0 when {side} masking correctly; got {loss=}"

        loss = training.compute_preds_and_get_ce_loss(
            model=mocked_model,
            tokens=tokens,
            attention_mask=t.ones_like(attn_mask),
            other_mask=None,
        )
        assert (
            loss != 0
        ), f"Expected loss != 0 when {side} masking correctly; got {loss=}"
