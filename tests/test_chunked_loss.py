import pytest
import torch

from common import token_losses


@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_chunked_loss_preserves_mask_values_and_gradients(chunk_size):
    torch.manual_seed(6304)
    original = torch.randn(2, 9, 17, requires_grad=True)
    chunked = original.detach().clone().requires_grad_()
    labels = torch.randint(0, 17, (2, 9))
    labels[:, :4] = -100  # Includes chunks containing only ignored tokens.
    labels[1, -2:] = -100
    expected, mask = token_losses(original, labels)
    actual, actual_mask = token_losses(chunked, labels, chunk_size=chunk_size)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_mask, mask)
    (expected.sum() / mask.sum()).backward()
    (actual.sum() / mask.sum()).backward()
    torch.testing.assert_close(chunked.grad, original.grad)


def test_chunked_loss_does_not_save_fp32_vocabulary_activations():
    logits = torch.randn(2, 9, 17, dtype=torch.float16, requires_grad=True)
    labels = torch.randint(0, 17, (2, 9))
    saved = []

    def pack(tensor):
        saved.append((tensor.shape, tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss, _ = token_losses(logits, labels, chunk_size=3)
    assert not any(len(shape) == 3 and dtype == torch.float32 for shape, dtype in saved)
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
