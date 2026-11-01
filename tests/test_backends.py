import pytest
import torch
from common import seed_all
from common import device as backend
from scripts.smoke import check_amp, check_tiny_qwen


def test_backend_selection_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setenv("PA3_DEVICE", "mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Requested mps GPU is unavailable"):
        backend.require_accelerator()


def test_auto_selects_apple_gpu_when_cuda_absent(monkeypatch):
    monkeypatch.setenv("PA3_DEVICE", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_active", None)
    assert backend.require_accelerator().type == "mps"
    assert backend.model_backend_kwargs("mps") == {"attn_implementation": "eager"}


def test_amp_finite_and_overflow_behavior_on_cpu():
    # CPU validates logic, not physical MPS support. Target-device tests are below.
    result = check_amp(torch.device("cpu"))
    assert (
        result["finite_update"] and result["overflow_skip"] and result["scale_backoff"]
    )
    assert result["rng_resume"] and result["scaler_resume"]


def test_exact_svd_returns_leading_subspaces():
    torch.manual_seed(6304)
    matrix = torch.randn(12, 8)
    u, v = backend.leading_bases(matrix, 3)
    assert u.shape == (12, 3) and v.shape == (8, 3)
    torch.testing.assert_close(u.T @ u, torch.eye(3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(v.T @ v, torch.eye(3), atol=1e-5, rtol=1e-5)
    _, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    torch.testing.assert_close((u.T @ matrix @ v).abs().diag(), singular[:3])


def test_mps_memory_monitor_is_labeled_as_sampled(monkeypatch):
    readings = iter([100, 500, 200, 200])
    monkeypatch.setattr(
        torch.mps, "current_allocated_memory", lambda: next(readings, 200)
    )
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 1024)
    monitor = backend.MPSMemoryMonitor(interval=10)
    try:
        monitor.sample()
        stats = monitor.stats()
        assert stats["peak_gpu_bytes"] == 500
        assert stats["peak_gpu_driver_bytes"] == 1024
        assert "lower bounds" in stats["memory_measurement"]
    finally:
        monitor.close()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Physical MPS device unavailable"
)
def test_physical_mps_amp_scaling_and_rng():
    check_amp(torch.device("mps"))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Physical MPS device unavailable"
)
def test_physical_mps_qwen_training_projection_and_generation():
    seed_all()
    check_tiny_qwen(torch.device("mps"))
