"""Backend checks with synthetic inputs. These are not assignment experiments."""

import argparse
import gc
import os
import time
import traceback
from pathlib import Path

import torch
from torch import nn
from common import (
    QWEN,
    dump,
    environment,
    rng_state,
    restore_rng,
    seed_all,
    token_losses,
)
from common.adapters import inject, export_factors, merge_into
from common.device import (
    autocast_for,
    empty_cache,
    make_scaler,
    memory_stats,
    model_backend_kwargs,
    require_accelerator,
    reset_peak_memory,
    stop_memory_tracking,
    synchronize,
)


def check_amp(device):
    """Validate native scaling, a real update, overflow skip and scaler checkpoint."""
    device = torch.device(device)
    seed_all()
    model = nn.Linear(8, 4).float().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = make_scaler(device, init_scale=128.0, growth_interval=2)
    x = torch.randn(4, 8, device=device)
    target = torch.zeros(4, 4, device=device)
    before = model.weight.detach().clone()
    with autocast_for(device):
        output = model(x)
        loss = (output.float() - target).square().mean()
    if output.dtype != torch.float16:
        raise AssertionError(f"Autocast did not produce FP16 output: {output.dtype}")
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    if torch.equal(before, model.weight):
        raise AssertionError("Finite FP16 training step did not update FP32 parameters")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise AssertionError("Trainable master parameters must stay FP32")
    for state in opt.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if state[key].dtype != torch.float32:
                raise AssertionError("AdamW moment states must stay FP32")
    opt.zero_grad(set_to_none=True)
    old_scale = scaler.get_scale()
    before_overflow = model.weight.detach().clone()
    with autocast_for(device):
        loss = model(x).float().square().mean()
    scaler.scale(loss).backward()
    next(model.parameters()).grad.fill_(float("inf"))
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    if (
        not torch.equal(before_overflow, model.weight)
        or not scaler.get_scale() < old_scale
    ):
        raise AssertionError(
            "Nonfinite gradients must skip optimizer step and reduce scale"
        )
    restored = make_scaler(device)
    restored.load_state_dict(scaler.state_dict())
    if restored.state_dict() != scaler.state_dict():
        raise AssertionError("Scaler checkpoint roundtrip failed")
    saved_rng = rng_state()
    first = torch.rand(8, device=device)
    restore_rng(saved_rng)
    if not torch.equal(first, torch.rand(8, device=device)):
        raise AssertionError("RNG checkpoint roundtrip failed")
    return {
        "autocast_dtype": str(output.dtype),
        "finite_update": True,
        "overflow_skip": True,
        "scale_backoff": True,
        "scaler_resume": True,
        "rng_resume": True,
        "optimizer_moments": "float32",
    }


def check_tiny_qwen(device):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        pad_token_id=0,
    )
    config._attn_implementation = model_backend_kwargs(device).get(
        "attn_implementation", "eager"
    )
    model = Qwen3ForCausalLM(config).float().to(device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    x = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = make_scaler(device, init_scale=128.0)
    model.train()
    with autocast_for(device):
        loss = model(x, labels=x).loss
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    del opt
    # Frozen FP16 backbone plus FP32 orthogonally projected trainables.
    model = model.half()
    inject(model, rank=4, alpha=8, protection=4)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".B"):
                p.normal_(std=0.001)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    opt.zero_grad(set_to_none=True)
    with autocast_for(device):
        loss = model(x, labels=x).loss
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0
    )
    scaler.step(opt)
    scaler.update()
    model.eval()
    with torch.no_grad(), autocast_for(device):
        unmerged = model(x).logits.float().cpu()
    factors = export_factors(model)
    # Recover base linear modules before applying the exported update.
    from common.adapters import Adapter

    for name in list(factors):
        adapter = model.get_submodule(name)
        if not isinstance(adapter, Adapter):
            raise AssertionError("Expected adapter module")
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, adapter.base)
    merge_into(model, factors)
    with torch.no_grad():
        merged = model(x).logits.float().cpu()
        generated = model.generate(
            x[:1],
            attention_mask=torch.ones_like(x[:1]),
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
    rms = (merged - unmerged).square().mean().sqrt().item()
    if rms > 0.02:
        raise AssertionError(f"Tiny merged/unmerged discrepancy: {rms}")
    return {
        "checkpointed_backward": True,
        "fp16_backbone_fp32_adapters": True,
        "exact_svd_projection": True,
        "merge_rms": rms,
        "generation_tokens": generated.shape[1] - x.shape[1],
    }


class SmokeFailure(AssertionError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def check_real_model(device, length=2048, steps=2, max_attempts=12):
    """Actual model and prescribed microbatch; synthetic tokens, no result claims."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        QWEN, torch_dtype=torch.float32, **model_backend_kwargs(device)
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8)
    scaler = make_scaler(device)
    # Start small enough to avoid unrelated invalid token IDs; shape is the requested length.
    x = torch.randint(
        10, min(1000, model.config.vocab_size), (2, length), device=device
    )
    labels = x.clone()
    labels[:, : length // 2] = -100
    # Native GradScaler may skip initial iterations while its default scale
    # calibrates. Require actual successful updates, with a bounded retry budget.
    # Do not reduce precision, disable scaling, or change the model/input shape.
    if max_attempts < steps:
        raise ValueError("max_attempts must be at least the requested successful steps")
    probe_name, probe = next(
        (n, p) for n, p in model.named_parameters() if p.requires_grad and p.ndim == 1
    )
    attempts = []
    successful_steps = 0
    initial_scale = scaler.get_scale()
    reset_peak_memory(device)
    synchronize(device)
    start = time.perf_counter()
    for attempt in range(max_attempts):
        opt.zero_grad(set_to_none=True)
        with autocast_for(device):
            logits = model(x, attention_mask=torch.ones_like(x)).logits
            values, valid = token_losses(logits, labels)
            loss = values.sum() / valid.sum()
        if not torch.isfinite(loss):
            raise SmokeFailure(
                "Nonfinite real-model smoke loss",
                {"attempts": attempts, "failed_attempt": attempt + 1},
            )
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old = scaler.get_scale()
        before = probe.detach().cpu().clone()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < old
        after = probe.detach().cpu()
        change = (after - before).abs().max().item()
        finite_grad_norm = bool(torch.isfinite(grad_norm))
        parameter_finite = bool(torch.isfinite(after).all())
        row = {
            "attempt": attempt + 1,
            "loss": loss.item(),
            "scale_before": old,
            "scale_after": scaler.get_scale(),
            "skipped_update": skipped,
            "finite_grad_norm": finite_grad_norm,
            "unscaled_grad_norm": grad_norm.item() if finite_grad_norm else None,
            "probe_parameter": probe_name,
            "probe_finite": parameter_finite,
            "probe_max_abs_change": change if parameter_finite else None,
        }
        attempts.append(row)
        print(
            f"real-model attempt {attempt + 1}/{max_attempts}: loss={row['loss']:.4f}, "
            f"scale={old:g}->{scaler.get_scale():g}, "
            f"{'SKIPPED (overflow)' if skipped else 'UPDATED'}, finite_grad_norm={finite_grad_norm}",
            flush=True,
        )
        if skipped and (not parameter_finite or change != 0):
            raise SmokeFailure(
                "An overflow-skipped step changed the parameter probe",
                {"attempts": attempts},
            )
        if not skipped:
            if not finite_grad_norm or not parameter_finite or change == 0:
                raise SmokeFailure(
                    "Optimizer did not produce a finite, nonzero verified update",
                    {"attempts": attempts},
                )
            successful_steps += 1
        del logits, values, valid, loss
        if successful_steps >= steps:
            break
    synchronize(device)
    stats = {
        "synthetic_inputs": True,
        "model": QWEN,
        "microbatch": 2,
        "sequence_length": length,
        "required_successful_steps": steps,
        "successful_steps": successful_steps,
        "attempted_steps": len(attempts),
        "max_attempts": max_attempts,
        "initial_scale": initial_scale,
        "final_scale": scaler.get_scale(),
        "losses": [r["loss"] for r in attempts],
        "skipped_updates": [r["skipped_update"] for r in attempts],
        "attempts": attempts,
        "seconds": time.perf_counter() - start,
        **memory_stats(device),
    }
    if successful_steps < steps:
        raise SmokeFailure(
            f"Only {successful_steps}/{steps} verified updates after {max_attempts} attempts; "
            "inspect saved loss/gradient/scale history before full training",
            stats,
        )
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--real-model", action="store_true")
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=12,
        help="Bound on real-model attempts, including initial scaler-calibration skips",
    )
    parser.add_argument("--output", default="results/backend-check.json")
    args = parser.parse_args()
    if not 2 <= args.length <= 2048 or args.steps < 1 or args.max_attempts < args.steps:
        parser.error("length must be 2..2048; steps >= 1; max-attempts >= steps")
    result = {"requested_device": args.device, "assignment_experiment": False}
    try:
        os.environ["PA3_DEVICE"] = args.device
        device = torch.device("cpu") if args.device == "cpu" else require_accelerator()
        seed_all()
        result["amp"] = check_amp(device)
        result["tiny_qwen"] = check_tiny_qwen(device)
        if args.real_model:
            if device.type == "cpu":
                raise ValueError("Real-model memory qualification requires a GPU")
            gc.collect()
            empty_cache(device)
            result["real_model"] = check_real_model(
                device, args.length, args.steps, args.max_attempts
            )
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=traceback.format_exc())
        if isinstance(exc, SmokeFailure):
            result["real_model"] = exc.details
    finally:
        stop_memory_tracking()
        dump(args.output, result)
        environment(str(Path(args.output).with_suffix(".environment.json")))
    print(result["status"] + ": " + args.output)
    if result["status"] != "passed":
        print(result["error"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
