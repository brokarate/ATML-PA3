from common.device import (
    autocast_for,
    empty_cache,
    make_scaler,
    memory_stats,
    require_accelerator,
    reset_peak_memory,
    synchronize,
)
import time
import gc
from pathlib import Path
import torch
from common.adapters import adapter_state, export_factors, inject
from common import (
    QWEN,
    dump,
    environment,
    jsonl,
    load_jsonl,
    lr_factor,
    optimizer_groups,
    pad_text,
    read,
    restore_rng,
    rng_state,
    seed_all,
    load_partial,
)
from .config import METHODS, SAVE_STEPS
from .loss import supervised_loss_sum
from .batching import MPS_LENGTH_BUCKETS, pad_to_bucket
from common import fingerprint, rewind_log
from common.device import model_backend_kwargs


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = require_accelerator()
    seed_all()
    rows = load_jsonl(Path(args.data) / "train.jsonl")
    data_hash = fingerprint(Path(args.data) / "train.jsonl")
    if len(rows) != 40000:
        raise ValueError("Task 1 must have exactly 40,000 prepared examples")
    peak, rank = METHODS[args.method]
    out = Path(args.output) / args.method
    out.mkdir(parents=True, exist_ok=True)
    revision = read(Path(args.data) / "selection.json")["model_revision"]
    tok = AutoTokenizer.from_pretrained(QWEN, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN,
        revision=revision,
        torch_dtype=torch.float16 if rank else torch.float32,
        **model_backend_kwargs(device),
    ).to(device)
    overhead = (
        inject(
            model, rank, rank * 2, protection=128 if args.method == "oplora16" else 0
        )
        if rank
        else {}
    )
    reference_start = time.perf_counter()
    reference = (
        {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}
        if args.method == "l2sp"
        else None
    )
    overhead.update(
        reference_storage_bytes=(
            sum(p.numel() * 4 for p in reference.values()) if reference else 0
        ),
        reference_copy_seconds=time.perf_counter() - reference_start,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    opt = torch.optim.AdamW(
        optimizer_groups(model), lr=peak, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, 1250, 38))
    scaler = make_scaler(device)
    start_step = 0
    elapsed_before = 0.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("device", "cuda") != device.type:
            raise ValueError(
                "Resume training on the same backend to preserve optimizer and RNG behavior"
            )
        if state["method"] != args.method:
            raise ValueError("Resume method mismatch")
        if state["revision"] != revision:
            raise ValueError("Resume model revision differs from prepared data")
        if state["data_sha256"] != data_hash:
            raise ValueError("Resume training data differs from checkpoint")
        (
            load_partial(model, state["model"])
            if rank
            else model.load_state_dict(state["model"])
        )
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"])
        start_step, elapsed_before = state["step"], state["elapsed"]
        rewind_log(out / "train.jsonl", start_step)
    elif (out / "train.jsonl").exists():
        raise FileExistsError(
            "Use --resume or a fresh output directory; existing run logs found"
        )
    environment(
        out / "environment.json",
        {
            "method": args.method,
            "overhead": overhead,
            "padding_buckets": (
                list(MPS_LENGTH_BUCKETS) if device.type == "mps" else None
            ),
            "model_revision": getattr(model.config, "_commit_hash", None),
        },
    )
    reset_peak_memory(device)
    synchronize(device)
    start = time.perf_counter()
    transfer_time = 0.0
    for step in range(start_step, 1250):
        update_start = time.perf_counter()
        batch = rows[step * 32 : (step + 1) * 32]
        denominator = sum(sum(x != -100 for x in r["labels"][1:]) for r in batch)
        opt.zero_grad(set_to_none=True)
        sft_sum = 0.0
        for offset in range(0, 32, 2):
            inputs = pad_text(batch[offset : offset + 2], tok.pad_token_id, device)
            if device.type == "mps":
                inputs = pad_to_bucket(inputs, tok.pad_token_id)
            with autocast_for(device):
                loss = supervised_loss_sum(model, inputs) / denominator
            sft_sum += loss.detach().item()
            scaler.scale(loss).backward()
            # Release this microbatch before allocating the next forward pass.
            del inputs, loss
            if device.type == "mps":
                # Checkpoint/autograd cycles can delay tensor destruction; Metal
                # also caches allocations for varying sequence lengths. Release
                # dead objects and unused pool blocks before the next microbatch.
                gc.collect()
                synchronize(device)
                empty_cache(device)
        omega = 0.0
        if reference is not None:
            # Parameter-wise backward avoids retaining a full FP32 reference on the GPU.
            for name, p in model.named_parameters():
                tick = time.perf_counter()
                ref = reference[name].to(device)
                synchronize(device)
                transfer_time += time.perf_counter() - tick
                penalty = 0.5 * (p.float() - ref).square().sum()
                omega += penalty.detach().item()
                scaler.scale(1e-3 * penalty).backward()
                del ref, penalty
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        old_scale = scaler.get_scale()
        lr = opt.param_groups[0]["lr"]
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        scheduler.step()
        synchronize(device)
        mps_memory = {}
        memory_text = ""
        if device.type == "mps":
            empty_cache(device)
            mps_memory = {
                "mps_live_bytes": torch.mps.current_allocated_memory(),
                "mps_driver_bytes": torch.mps.driver_allocated_memory(),
            }
            memory_text = (
                f" | MPS live={mps_memory['mps_live_bytes'] / 2**30:.1f}GiB"
                f" driver={mps_memory['mps_driver_bytes'] / 2**30:.1f}GiB"
            )
        update_seconds = time.perf_counter() - update_start
        session_seconds = time.perf_counter() - start
        average_seconds = session_seconds / (step + 1 - start_step)
        eta_seconds = average_seconds * (1250 - step - 1)
        jsonl(
            out / "train.jsonl",
            [
                {
                    "update": step + 1,
                    "examples": (step + 1) * 32,
                    "sft_loss": sft_sum,
                    "omega": omega,
                    "lambda_omega": 0.001 * omega,
                    "total_loss": sft_sum + 0.001 * omega,
                    "lr": lr,
                    "grad_norm": (
                        float(grad_norm) if torch.isfinite(grad_norm) else None
                    ),
                    "skipped_update": skipped,
                    "scaler": scaler.get_scale(),
                    "update_seconds": update_seconds,
                    "elapsed_seconds": elapsed_before + session_seconds,
                    "eta_seconds": eta_seconds,
                    **mps_memory,
                }
            ],
            append=True,
        )
        print(
            f"{args.method} update {step + 1}/1250 | "
            f"loss={sft_sum + 0.001 * omega:.4f} | "
            f"{update_seconds:.1f}s/update | "
            f"ETA {eta_seconds / 3600:.1f}h | "
            f"{'SKIPPED (overflow)' if skipped else 'UPDATED'}{memory_text}",
            flush=True,
        )
        if step + 1 in SAVE_STEPS:
            synchronize(device)
            elapsed = elapsed_before + time.perf_counter() - start
            state = {
                "method": args.method,
                "device": device.type,
                "revision": revision,
                "data_sha256": data_hash,
                "step": step + 1,
                "data_position": (step + 1) * 32,
                "model": (
                    adapter_state(model)
                    if rank
                    else {n: p.cpu() for n, p in model.state_dict().items()}
                ),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng_state(),
                "elapsed": elapsed,
            }
            dest = out / f"update_{step + 1}.pt"
            torch.save(state, dest)
            del state
            if rank:
                tick = time.perf_counter()
                torch.save(export_factors(model), out / f"factors_{step + 1}.pt")
                overhead["export_seconds"] = time.perf_counter() - tick
            dump(
                out / f"resources_{step + 1}.json",
                {
                    "training_seconds": elapsed,
                    **memory_stats(device),
                    "checkpoint_bytes": dest.stat().st_size,
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "reference_transfer_seconds": transfer_time,
                    **overhead,
                },
            )
    tok.save_pretrained(out / "tokenizer")
    # Compact logs/resources belong in Git; checkpoints and downloaded data do not.
    import shutil

    tracked = Path("results/task1/training") / args.method
    tracked.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.json*"):
        shutil.copy2(p, tracked / p.name)
