"""Exact response-token loss without materializing full-sequence logits."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def supervised_loss_sum(model, inputs, chunk_size=128, fixed_shapes=False):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    hidden = model.model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_cache=False,
    ).last_hidden_state
    labels = inputs["labels"][:, 1:]
    if fixed_shapes:
        # Keep prompt/pad positions as ignored targets, avoiding variable-size
        # boolean gathers. Pad the last head chunk to the same shape as all others.
        selected = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        targets = labels.reshape(-1)
        extra = (-targets.numel()) % chunk_size
        selected = F.pad(selected, (0, 0, 0, extra))
        targets = F.pad(targets, (0, extra), value=-100)
    else:
        valid = labels.ne(-100)
        # Ignored prompt/padding tokens contribute zero loss and zero head gradient.
        selected = hidden[:, :-1][valid]
        targets = labels[valid]
    if targets.numel() == 0:
        return hidden.sum() * 0.0

    def head_loss(values, target):
        logits = model.lm_head(values)
        return F.cross_entropy(logits.float(), target, reduction="sum")

    losses = []
    for start in range(0, targets.numel(), chunk_size):
        stop = start + chunk_size
        # Recompute the linear head and FP32 CE on backward. Only small hidden
        # states survive forward, rather than sequence x vocabulary tensors.
        losses.append(
            checkpoint(
                head_loss,
                selected[start:stop],
                targets[start:stop],
                use_reentrant=False,
                preserve_rng_state=False,
            )
        )
    return torch.stack(losses).sum()
