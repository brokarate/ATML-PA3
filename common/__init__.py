import hashlib
import importlib.metadata
import json
import math
import random
import re
import subprocess
import unicodedata
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

SEED = 6304
QWEN = "Qwen/Qwen3-0.6B"


def model_revision(model=QWEN):
    from transformers import AutoConfig

    revision = getattr(AutoConfig.from_pretrained(model), "_commit_hash", None)
    if not revision:
        raise RuntimeError("Could not resolve immutable model revision")
    return revision


def load_partial(model, state):
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    expected |= {
        n for n, _ in model.named_buffers() if n.endswith((".U", ".V")) or n == "scale"
    }
    if set(state) != expected:
        raise ValueError(
            f"Trainable checkpoint keys mismatch: missing={expected - set(state)}, extra={set(state) - expected}"
        )
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(result.unexpected_keys)


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path, rows, append=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str, allow_nan=False) + "\n")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rewind_log(path, update):
    path = Path(path)
    if path.exists():
        rows = [r for r in load_jsonl(path) if r["update"] <= update]
        jsonl(path, rows)


def normalize_text(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).lower()).strip()


def environment(path, extra=None):
    from .device import backend_info

    packages = {}
    for name in (
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "pandas",
        "lm-eval",
        "math-verify",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    protocol = Path(__file__).resolve().parent.parent / "configs/protocol.json"
    root = Path(__file__).resolve().parent.parent
    source_hashes = {
        str(p.relative_to(root)): fingerprint(p)
        for folder in (
            "common",
            "pa3",
            "task1_finetuning",
            "task2_vlm",
            "task3_inference",
        )
        for p in (root / folder).glob("*.py")
    }
    dump(
        path,
        {
            "packages": packages,
            "git": rev.stdout.strip(),
            "seed": SEED,
            "source_sha256": source_hashes,
            "protocol": read(protocol) if protocol.exists() else None,
            "protocol_sha256": fingerprint(protocol) if protocol.exists() else None,
            **backend_info(),
            "cuda": torch.version.cuda,
            **(extra or {}),
        },
    )


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state_all(s["cuda"])
    if s.get("mps") is not None:
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Checkpoint has MPS RNG state but MPS is unavailable; resume on the same backend"
            )
        torch.mps.set_rng_state(s["mps"].cpu())


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chat_ids(tokenizer, messages, generation=True):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation, enable_thinking=False
    )


def supervised(tokenizer, question, answer):
    prompt = chat_ids(tokenizer, [{"role": "user", "content": question}])
    # Appending directly to the non-thinking generation prefix avoids supervising
    # Qwen's empty thinking block. Include exactly one end-of-turn token.
    target = tokenizer.encode(answer, add_special_tokens=False) + [
        tokenizer.convert_tokens_to_ids("<|im_end|>")
    ]
    return {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target}


def pad_text(rows, pad_id, device):
    width = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        ids[i, :n] = torch.tensor(r["input_ids"], device=device)
        labels[i, :n] = torch.tensor(r.get("labels", [-100] * n), device=device)
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def token_losses(logits, labels, chunk_size=None):
    labels = labels[:, 1:]
    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        from torch.utils.checkpoint import checkpoint

        def cross_entropy(values, targets):
            return F.cross_entropy(
                values.float().transpose(1, 2),
                targets,
                reduction="none",
                ignore_index=-100,
            )

        # Recompute only the deterministic loss chunks during backward, avoiding
        # a full sequence-by-vocabulary FP32 softmax retained for backward.
        chunks = []
        for start in range(0, labels.shape[1], chunk_size):
            stop = min(start + chunk_size, labels.shape[1])
            values, targets = logits[:, start:stop], labels[:, start:stop]
            if torch.is_grad_enabled() and values.requires_grad:
                chunk = checkpoint(
                    cross_entropy,
                    values,
                    targets,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                chunk = cross_entropy(values, targets)
            chunks.append(chunk)
        return torch.cat(chunks, dim=1), labels.ne(-100)
    loss = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2),
        labels,
        reduction="none",
        ignore_index=-100,
    )
    return loss, labels.ne(-100)


def optimizer_groups(model, weight_decay=0.01, lr=None):
    groups = [[], []]
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups[int(p.ndim < 2 or name.endswith("bias"))].append(p)
    return [
        {
            "params": p,
            "weight_decay": 0 if i else weight_decay,
            **({"lr": lr} if lr is not None else {}),
        }
        for i, p in enumerate(groups)
        if p
    ]


def lr_factor(step, total, warmup, floor=0.0):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def bootstrap(values, count, seed=SEED):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("Bootstrap requires one value per independent question/pair")
    rng = np.random.default_rng(seed)
    means = [
        float(values[rng.integers(len(values), size=len(values))].mean())
        for _ in range(count)
    ]
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
    }
