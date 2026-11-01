import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from common import token_losses
from common.adapters import inject
from task1_finetuning.loss import supervised_loss_sum


@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("fixed_shapes", [False, True])
def test_head_chunks_preserve_two_update_training(adapter, tied, fixed_shapes):
    torch.manual_seed(6304)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=41,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            tie_word_embeddings=tied,
            attention_dropout=0.0,
            use_cache=False,
        )
    )
    if adapter:
        inject(model, rank=4, alpha=8)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    reference = copy.deepcopy(model).train()
    model.train()
    optimizers = [
        torch.optim.AdamW(m.parameters(), lr=2e-5) for m in (reference, model)
    ]
    head_sizes = []
    hook = model.lm_head.register_forward_hook(
        lambda module, args, output: head_sizes.append(output.shape[0])
    )
    for _ in range(2):
        inputs = {
            "input_ids": torch.randint(0, 41, (2, 9)),
            "attention_mask": torch.ones(2, 9, dtype=torch.long),
            "labels": torch.randint(0, 41, (2, 9)),
        }
        inputs["labels"][:, :4] = -100
        inputs["labels"][1, -2:] = -100
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        expected, mask = token_losses(
            reference(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).logits,
            inputs["labels"],
        )
        expected = expected.sum() / mask.sum()
        if fixed_shapes:
            inputs = {
                key: torch.nn.functional.pad(value, (0, 4), value=pad)
                for (key, value), pad in zip(inputs.items(), (0, 0, -100))
            }
        actual = (
            supervised_loss_sum(model, inputs, chunk_size=3, fixed_shapes=fixed_shapes)
            / mask.sum()
        )
        torch.testing.assert_close(actual, expected)
        expected.backward()
        actual.backward()
        for (name, p), (_, q) in zip(
            reference.named_parameters(), model.named_parameters()
        ):
            if p.grad is not None:
                torch.testing.assert_close(
                    p.grad, q.grad, rtol=2e-4, atol=2e-6, msg=name
                )
        for opt in optimizers:
            opt.step()
    hook.remove()
    assert head_sizes and max(head_sizes) <= 3
    for p, q in zip(reference.parameters(), model.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-4, atol=2e-6)
