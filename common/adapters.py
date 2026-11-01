import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from .device import leading_bases

ATTENTION = {"q_proj", "k_proj", "v_proj", "o_proj"}
ALL_TARGETS = ATTENTION | {"gate_proj", "up_proj", "down_proj"}


class Adapter(nn.Module):
    def __init__(self, base, rank=16, alpha=32, dropout=0.05, protection=0):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        # Initialize A before doing any SVD: LoRA-16/OPLoRA have identical RNG consumption.
        self.A = nn.Parameter(
            torch.empty(
                rank, base.in_features, device=base.weight.device, dtype=torch.float32
            )
        )
        self.B = nn.Parameter(
            torch.zeros(
                base.out_features, rank, device=base.weight.device, dtype=torch.float32
            )
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        if protection:
            if protection > min(base.weight.shape):
                raise ValueError("Protection rank exceeds weight dimensions")
            with torch.no_grad():
                u, v = leading_bases(base.weight, protection)
            self.register_buffer("U", u)
            self.register_buffer("V", v)
        else:
            self.register_buffer("U", None)
            self.register_buffer("V", None)

    def factors(self):
        if self.U is None:
            return self.B, self.A
        # Projection is deliberately FP32 even inside an autocast training forward.
        with torch.autocast(self.A.device.type, enabled=False):
            return (
                self.B - self.U @ (self.U.T @ self.B),
                self.A - (self.A @ self.V) @ self.V.T,
            )

    def delta(self):
        b, a = self.factors()
        return self.scale * (b @ a)

    def forward(self, x):
        b, a = self.factors()
        return self.base(x) + F.linear(F.linear(self.dropout(x), a), b) * self.scale


def inject(model, rank=16, alpha=32, targets=ALL_TARGETS, protection=0):
    model.requires_grad_(False)
    start = time.perf_counter()
    names = [
        n
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.split(".")[-1] in targets
    ]
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(
            parent,
            child,
            Adapter(getattr(parent, child), rank, alpha, 0.05, protection),
        )
    return {
        "modules": names,
        "preprocessing_seconds": time.perf_counter() - start,
        "basis_bytes": sum(
            (m.U.numel() + m.V.numel()) * 4
            for m in model.modules()
            if isinstance(m, Adapter) and m.U is not None
        ),
    }


def adapter_state(model):
    return {
        n: p.detach().cpu()
        for n, p in model.state_dict().items()
        if n.endswith((".A", ".B", ".U", ".V"))
    }


def export_factors(model):
    return {
        n: {
            "A": m.factors()[1].detach().cpu(),
            "B": m.factors()[0].detach().cpu(),
            "scale": m.scale,
        }
        for n, m in model.named_modules()
        if isinstance(m, Adapter)
    }


@torch.no_grad()
def merge_into(fresh, factors):
    for name, f in factors.items():
        weight = fresh.get_submodule(name).weight
        weight.copy_(
            (
                weight.float()
                + f["scale"]
                * f["B"].to(weight.device).float()
                @ f["A"].to(weight.device).float()
            ).to(weight.dtype)
        )
    return fresh
