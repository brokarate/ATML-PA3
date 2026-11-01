"""Run bounded, isolated MPS checks and preserve results for inspection."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("Run this command in your Mac Terminal with MPS available")
    root = Path(__file__).resolve().parents[1]
    destination = root / "results/mps-diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "torch": torch.__version__,
        "python": sys.executable,
        "cases": [],
        "note": "Diagnostic runs only; no assignment checkpoints are created.",
    }
    jobs = [
        ("optimizer", "scripts.mps_optimizer_probe", []),
        ("fixed", "scripts.mps_shape_probe", ["--shape", "fixed"]),
        ("fixed-input", "scripts.mps_shape_probe", ["--shape", "fixed-input"]),
        ("dynamic", "scripts.mps_shape_probe", ["--shape", "dynamic"]),
    ]
    # Keep matmul selection identical between these comparisons. Do not alter
    # allocator safety limits or enable automatic CPU fallback.
    environment = os.environ.copy()
    environment.pop("PYTORCH_MPS_PREFER_METAL", None)
    summary = destination / "summary.json"

    def save():
        summary.write_text(json.dumps(result, indent=2) + "\n")

    save()
    for name, module, arguments in jobs:
        print(f"\nStarting diagnostic: {name}", flush=True)
        log = destination / f"{name}.log"
        start = time.perf_counter()
        entry = {"case": name, "log": str(log)}
        try:
            with log.open("w") as output:
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", module, *arguments],
                    cwd=root,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                try:
                    # Check progress without holding the terminal silent for a
                    # whole model run. Kill a stalled child before the next case.
                    while True:
                        try:
                            entry["returncode"] = process.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            print(
                                f"{name}: still checking; output in {log.name}",
                                flush=True,
                            )
                            if time.perf_counter() - start > 600:
                                process.kill()
                                process.wait()
                                entry["status"] = "timeout"
                                break
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
            entry.setdefault(
                "status", "completed" if entry.get("returncode") == 0 else "stopped"
            )
        except KeyboardInterrupt:
            entry["status"] = "interrupted"
            result["status"] = "interrupted"
            entry["seconds"] = time.perf_counter() - start
            result["cases"].append(entry)
            save()
            raise
        except Exception as error:
            entry.update(status="error", error=str(error))
        entry["seconds"] = time.perf_counter() - start
        result["cases"].append(entry)
        save()
        print(f"{name}: {entry['status']} ({entry['seconds']:.0f}s)", flush=True)
    result["status"] = "finished"
    save()
    print(
        f"\nDiagnostics finished: {summary}\nTell me the checks are done.", flush=True
    )


if __name__ == "__main__":
    main()
