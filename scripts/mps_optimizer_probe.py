"""Bounded optimizer-only MPS diagnostic; no model, data, or training outputs."""

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch


def worker(implementation, steps):
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Run this check in your Mac Terminal; MPS is unavailable here"
        )
    torch.manual_seed(6304)
    parameter = torch.nn.Parameter(torch.zeros(16384, 1024, device="mps"))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=2e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        fused=implementation == "fused",
        foreach=False,
    )
    parameter.grad = torch.full_like(parameter, 0.001)
    records = []
    for step in range(steps):
        optimizer.param_groups[0]["lr"] = 2e-5 * (step + 1) / 38
        optimizer.step()
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
        record = {
            "step": step + 1,
            "live_bytes": torch.mps.current_allocated_memory(),
            "driver_bytes": torch.mps.driver_allocated_memory(),
        }
        records.append(record)
        print(
            f"{implementation} {step + 1}/{steps}: "
            f"live={record['live_bytes'] / 2**30:.3f}GiB "
            f"driver={record['driver_bytes'] / 2**30:.3f}GiB",
            file=sys.stderr,
            flush=True,
        )
        # Bound this small diagnostic, rather than approaching the system limit.
        if record["driver_bytes"] > 4 * 2**30:
            break
    return {
        "implementation": implementation,
        "parameter_shape": list(parameter.shape),
        "records": records,
        "driver_growth_bytes": records[-1]["driver_bytes"] - records[0]["driver_bytes"],
        "completed_steps": len(records),
        "finite_parameters": bool(torch.isfinite(parameter).all().item()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["standard", "fused"])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", default="results/mps-optimizer-probe.json")
    args = parser.parse_args()
    if not 1 <= args.steps <= 20:
        parser.error("steps must be between 1 and 20")
    if args.worker:
        print(json.dumps(worker(args.worker, args.steps)))
        return
    result = {"torch": torch.__version__, "checks": [], "status": "completed"}
    for implementation in ("standard", "fused"):
        # Separate processes release all Metal/framework caches between checks.
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.mps_optimizer_probe",
                "--worker",
                implementation,
                "--steps",
                str(args.steps),
            ],
            text=True,
            stdout=subprocess.PIPE,
        )
        if completed.returncode:
            result["status"] = "failed"
            result["checks"].append(
                {"implementation": implementation, "returncode": completed.returncode}
            )
            break
        result["checks"].append(json.loads(completed.stdout))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['status']}: {destination}", flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
