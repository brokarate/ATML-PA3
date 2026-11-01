from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import smoke


class TinyOverflowModel(nn.Module):
    def __init__(self, overflow_for):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=32)
        self.embed = nn.Embedding(32, 4)
        self.head = nn.Linear(4, 32)
        self.calls = 0
        self.overflow_for = overflow_for
        self.head.bias.register_hook(self.poison_initial_gradients)

    def poison_initial_gradients(self, grad):
        return (
            torch.full_like(grad, float("inf"))
            if self.calls <= self.overflow_for
            else grad
        )

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def forward(self, x, **kwargs):
        self.calls += 1
        return SimpleNamespace(logits=self.head(self.embed(x)))


def setup_synthetic_model(monkeypatch, overflow_for):
    import transformers

    torch.manual_seed(6304)
    model = TinyOverflowModel(overflow_for)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model
    )
    monkeypatch.setattr(smoke, "reset_peak_memory", lambda device: None)
    monkeypatch.setattr(smoke, "synchronize", lambda device: None)
    monkeypatch.setattr(
        smoke,
        "memory_stats",
        lambda device: {
            "peak_gpu_bytes": 0,
            "memory_measurement": "synthetic CPU fixture",
        },
    )


def test_real_model_check_allows_calibration_but_requires_real_updates(monkeypatch):
    setup_synthetic_model(monkeypatch, overflow_for=2)
    result = smoke.check_real_model(
        torch.device("cpu"), length=8, steps=2, max_attempts=8
    )
    assert result["successful_steps"] == 2
    assert result["attempted_steps"] >= 4
    assert all(r["skipped_update"] for r in result["attempts"][:2])
    assert result["attempts"][0]["scale_after"] < result["initial_scale"]
    for attempt in result["attempts"]:
        if not attempt["skipped_update"]:
            assert attempt["finite_grad_norm"]
            assert attempt["probe_finite"] and attempt["probe_max_abs_change"] > 0


def test_real_model_check_preserves_diagnostics_and_fails_persistent_overflow(
    monkeypatch,
):
    setup_synthetic_model(monkeypatch, overflow_for=100)
    with pytest.raises(smoke.SmokeFailure) as caught:
        smoke.check_real_model(torch.device("cpu"), length=8, steps=2, max_attempts=3)
    assert caught.value.details["successful_steps"] == 0
    assert len(caught.value.details["attempts"]) == 3
    assert caught.value.details["final_scale"] < caught.value.details["initial_scale"]
