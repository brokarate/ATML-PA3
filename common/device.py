"""Explicit GPU selection, FP16 AMP and backend-specific measurement.

No quantization, batch-size reduction, or automatic FP32 training fallback.
"""

import atexit
import os
import platform
import threading

import torch

_active = None
_monitor = None


def require_accelerator():
    global _active
    requested = os.environ.get("PA3_DEVICE", "auto").lower()
    if requested not in {"auto", "mps", "cuda"}:
        raise ValueError("PA3_DEVICE must be auto, mps, or cuda")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps"
    available = (
        torch.cuda.is_available()
        if requested == "cuda"
        else torch.backends.mps.is_available()
    )
    if not available:
        raise RuntimeError(
            f"Requested {requested} GPU is unavailable to this Python process. "
            "On Apple Silicon, run `python -m scripts.smoke --device mps` in your own Terminal. "
            "No CPU, BF16, or unscaled-training fallback is applied."
        )
    _active = torch.device(requested)
    if requested == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return _active


def current_device(device=None):
    return (
        torch.device(device)
        if device is not None
        else (_active or require_accelerator())
    )


def autocast_for(device):
    return torch.autocast(torch.device(device).type, dtype=torch.float16)


def make_scaler(device, **kwargs):
    scaler = torch.amp.GradScaler(torch.device(device).type, **kwargs)
    if not scaler.is_enabled():
        raise RuntimeError("Dynamic gradient scaling must remain enabled")
    return scaler


def synchronize(device=None):
    device = current_device(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def empty_cache(device=None):
    device = current_device(device)
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def model_backend_kwargs(device):
    # Eager attention avoids CUDA-only/unsupported optimized attention kernels.
    # This changes kernel choice, not the architecture, objective or token budget.
    return (
        {"attn_implementation": "eager"} if torch.device(device).type == "mps" else {}
    )


def leading_bases(weight, rank):
    """Exact FP32 SVD; MPS preprocessing runs explicitly on CPU, once per weight."""
    target = weight.device
    matrix = weight.detach().float()
    if target.type == "mps":
        matrix = matrix.cpu()
    with torch.no_grad(), torch.autocast(matrix.device.type, enabled=False):
        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u[:, :rank].contiguous().to(target), vh[:rank].T.contiguous().to(target)


class MPSMemoryMonitor:
    """Sampled peaks are lower bounds, never presented as CUDA allocator peaks."""

    def __init__(self, interval=0.01):
        self.interval = interval
        self.allocated = self.driver = 0
        self.error = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def sample(self):
        try:
            allocated = torch.mps.current_allocated_memory()
            driver = torch.mps.driver_allocated_memory()
            with self.lock:
                self.allocated = max(self.allocated, allocated)
                self.driver = max(self.driver, driver)
        except Exception as exc:
            self.error = repr(exc)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def close(self):
        self.stop_event.set()
        self.thread.join()

    def stats(self):
        self.sample()
        if self.error:
            raise RuntimeError(f"MPS memory sampling failed: {self.error}")
        with self.lock:
            return {
                "peak_gpu_bytes": self.allocated,
                "peak_gpu_driver_bytes": self.driver,
                "memory_measurement": "sampled MPS high-water marks; lower bounds, not exact peaks",
                "memory_sampling_interval_seconds": self.interval,
            }


def stop_memory_tracking():
    global _monitor
    if _monitor is not None:
        _monitor.close()
        _monitor = None


atexit.register(stop_memory_tracking)


def reset_peak_memory(device=None):
    global _monitor
    device = current_device(device)
    stop_memory_tracking()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        _monitor = MPSMemoryMonitor()


def memory_stats(device=None):
    device = current_device(device)
    if device.type == "cuda":
        return {
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
            "memory_measurement": "CUDA allocator peak allocated bytes",
        }
    if device.type == "mps":
        if _monitor is None:
            raise RuntimeError(
                "Call reset_peak_memory before collecting MPS memory statistics"
            )
        return _monitor.stats()
    raise ValueError("GPU memory measurement requires mps or cuda")


def backend_info():
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    name = (
        torch.cuda.get_device_name()
        if available["cuda"]
        else ("Apple GPU (MPS)" if available["mps"] else None)
    )
    return {
        "gpu": name,
        "device": str(_active) if _active else None,
        "available_backends": available,
        "macos": platform.mac_ver()[0] or None,
        "machine": platform.machine(),
        "mps_cpu_svd": True,
        "mps_fallback_environment": os.environ.get(
            "PYTORCH_ENABLE_MPS_FALLBACK", "unset"
        ),
        "mps_attention": "eager",
        "mps_prefer_metal": os.environ.get("PYTORCH_MPS_PREFER_METAL", "unset"),
        "precision_policy": "FP16 autocast; FP32 trainable parameters and optimizer states; dynamic gradient scaling",
    }
