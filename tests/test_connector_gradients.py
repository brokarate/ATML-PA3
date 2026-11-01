import torch
from task2_vlm.train import connector_gradient_status, merge_memory_peaks


def test_memory_peaks_preserve_resume_prefix_and_separate_stage_values():
    stage_one = {"peak_gpu_bytes": 900, "peak_gpu_driver_bytes": 1500}
    resumed_segment = {"peak_gpu_bytes": 800, "peak_gpu_driver_bytes": 1600}
    assert merge_memory_peaks(stage_one, resumed_segment) == {
        "peak_gpu_bytes": 900,
        "peak_gpu_driver_bytes": 1600,
    }
    stage_two = {"peak_gpu_bytes": 700, "peak_gpu_driver_bytes": 1400}
    assert merge_memory_peaks(stage_two) == stage_two
    assert merge_memory_peaks(stage_one, stage_two) == stage_one


def test_missing_zero_and_nonfinite_gradients_are_distinct():
    connector = torch.nn.Linear(2, 2)
    assert connector_gradient_status(connector) == "missing"
    for p in connector.parameters():
        p.grad = torch.zeros_like(p)
    assert connector_gradient_status(connector) == "zero"
    connector.weight.grad[0, 0] = 1
    assert connector_gradient_status(connector) == "finite_nonzero"
    connector.weight.grad[0, 0] = float("nan")
    assert connector_gradient_status(connector) == "nonfinite"


def test_overflow_reaches_scaler_and_skips_weight_update():
    connector = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(connector.parameters())
    scaler = torch.amp.GradScaler("cpu", init_scale=65536)
    before = [p.detach().clone() for p in connector.parameters()]
    scaler.scale(connector(torch.ones(1, 2)).sum()).backward()
    connector.weight.grad.fill_(float("nan"))
    assert connector_gradient_status(connector) == "nonfinite"
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    assert scaler.get_scale() == 32768
    for p, original in zip(connector.parameters(), before):
        torch.testing.assert_close(p, original)
