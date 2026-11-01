import importlib.metadata
import torch
from common import environment


def main():
    environment("results/environment-check.json")
    required = {
        "torch": "2.9.1",
        "transformers": "4.57.1",
        "datasets": "3.6.0",
        "lm-eval": "0.4.9.1",
        "math-verify": "0.8.0",
    }
    errors = []
    for name, wanted in required.items():
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed = "missing"
        print(f"{name}: {observed} (implementation pin: {wanted})")
        if observed.split("+")[0] != wanted:
            errors.append(name)
    print("CUDA:", torch.cuda.is_available())
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())
    if torch.backends.mps.is_available():
        print(
            "Apple GPU detected. Validate FP16/scaling with: python -m scripts.smoke --device mps"
        )
    elif not torch.cuda.is_available():
        print(
            "No GPU is visible to this process. Offline analysis/tests can run; training fails explicitly."
        )
    if errors:
        raise SystemExit(
            "Install requirements in an isolated environment; differing/missing packages: "
            + ", ".join(errors)
        )


if __name__ == "__main__":
    main()
