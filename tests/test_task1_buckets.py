import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from common import pad_text
from task1_finetuning.batching import pad_to_bucket
from task1_finetuning.loss import supervised_loss_sum


@pytest.mark.parametrize(
    "length,target", [(511, 512), (512, 512), (513, 1024), (1025, 2048), (2048, 2048)]
)
def test_bucket_keeps_real_tokens_and_masks_padding(length, target):
    row = {"input_ids": list(range(length)), "labels": [-100] + list(range(1, length))}
    original = pad_text([row], 0, "cpu")
    padded = pad_to_bucket(original, 0)
    for key in original:
        torch.testing.assert_close(padded[key][:, :length], original[key])
    assert padded["input_ids"].shape == (1, target)
    assert padded["attention_mask"][:, length:].eq(0).all()
    assert padded["labels"][:, length:].eq(-100).all()


def test_bucket_rejects_overlength_instead_of_truncating():
    inputs = pad_text([{"input_ids": [1] * 2049}], 0, "cpu")
    with pytest.raises(ValueError, match="never truncate"):
        pad_to_bucket(inputs, 0)


def test_real_bucket_padding_preserves_supervised_loss_and_parameter_gradients():
    torch.manual_seed(6304)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=19,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_dropout=0.0,
            use_cache=False,
        )
    ).train()
    reference = copy.deepcopy(model)
    inputs = pad_text(
        [
            {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]},
            {"input_ids": [6, 7, 8], "labels": [-100, -100, 8]},
        ],
        0,
        "cpu",
    )
    expected = supervised_loss_sum(reference, inputs)
    actual = supervised_loss_sum(model, pad_to_bucket(inputs, 0))
    torch.testing.assert_close(actual, expected)
    expected.backward()
    actual.backward()
    for p, q in zip(reference.parameters(), model.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6)
