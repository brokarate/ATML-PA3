"""Bounded real-model diagnostic, separate from assignment training/checkpoints."""

import argparse
import gc
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import (
    QWEN,
    dump,
    environment,
    load_jsonl,
    lr_factor,
    optimizer_groups,
    pad_text,
    read,
    seed_all,
)
from common.device import autocast_for, make_scaler, require_accelerator
from task1_finetuning.loss import supervised_loss_sum
from task1_finetuning.batching import pad_to_bucket


def main():
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        choices=["fixed", "fixed-input", "bucketed", "dynamic"],
        default="fixed",
    )
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--data", default="data/task1")
    args = parser.parse_args()
    if not 1 <= args.updates <= 10:
        parser.error("updates must be between 1 and 10")
    os.environ["PA3_DEVICE"] = "mps"
    device = require_accelerator()
    seed_all()
    destination = Path(f"results/mps-shape-probe-{args.shape}.json")
    result = {
        "status": "running",
        "shape": args.shape,
        "measurements": [],
        "updates": [],
    }
    environment(destination.with_suffix(".environment.json"))

    def measure(phase, update, microbatch):
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
        entry = {
            "phase": phase,
            "update": update,
            "microbatch": microbatch,
            "live_bytes": torch.mps.current_allocated_memory(),
            "driver_bytes": torch.mps.driver_allocated_memory(),
        }
        result["measurements"].append(entry)
        if entry["driver_bytes"] > 22 * 2**30:
            raise RuntimeError("Diagnostic memory guard: driver exceeded 22 GiB")
        return entry

    try:
        revision = read(Path(args.data) / "selection.json")["model_revision"]
        tokenizer = AutoTokenizer.from_pretrained(QWEN, revision=revision)
        model = (
            AutoModelForCausalLM.from_pretrained(
                QWEN,
                revision=revision,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            .to(device)
            .train()
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        optimizer = torch.optim.AdamW(
            optimizer_groups(model), lr=2e-5, betas=(0.9, 0.999), eps=1e-8
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_factor(step, 1250, 38)
        )
        scaler = make_scaler(device)
        rows = load_jsonl(Path(args.data) / "train.jsonl")
        for update in range(1, args.updates + 1):
            start = time.perf_counter()
            batch = rows[(update - 1) * 32 : update * 32]
            if args.shape == "bucketed" and update == args.updates:
                # Stress the largest shape too, rather than testing only short
                # source-prefix examples. This diagnostic never saves trainable
                # checkpoints or changes the real training selection/order.
                long_indices = [
                    i for i, row in enumerate(rows) if len(row["input_ids"]) > 1024
                ][:2]
                if len(long_indices) != 2:
                    raise ValueError(
                        "Two long examples required for shape stress check"
                    )
                batch = batch[:-2] + [rows[i] for i in long_indices]
                result["long_example_stress_indices"] = long_indices
            denominator = sum(
                sum(x != -100 for x in row["labels"][1:]) for row in batch
            )
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for offset in range(0, 32, 2):
                inputs = pad_text(
                    batch[offset : offset + 2], tokenizer.pad_token_id, device
                )
                if args.shape == "bucketed":
                    inputs = pad_to_bucket(inputs, tokenizer.pad_token_id)
                elif args.shape != "dynamic":
                    width = inputs["input_ids"].shape[1]
                    extra = 2048 - width
                    if extra < 0:
                        raise ValueError("Prepared example exceeds 2048 tokens")
                    for key, value in (
                        ("input_ids", tokenizer.pad_token_id),
                        ("labels", -100),
                        ("attention_mask", 0),
                    ):
                        inputs[key] = F.pad(inputs[key], (0, extra), value=value)
                with autocast_for(device):
                    loss = (
                        supervised_loss_sum(
                            model, inputs, fixed_shapes=args.shape == "fixed"
                        )
                        / denominator
                    )
                total += loss.detach().item()
                measure("forward", update, offset // 2 + 1)
                scaler.scale(loss).backward()
                del inputs, loss
                measure("backward", update, offset // 2 + 1)
                if (offset // 2 + 1) % 4 == 0:
                    print(
                        f"diagnostic update {update}/{args.updates}: "
                        f"microbatch {offset // 2 + 1}/16",
                        flush=True,
                    )
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            memory = measure("optimizer", update, 16)
            entry = {
                "update": update,
                "loss": total,
                "seconds": time.perf_counter() - start,
                "finite_grad_norm": bool(torch.isfinite(norm)),
                "skipped": scaler.get_scale() < before,
                **memory,
            }
            result["updates"].append(entry)
            dump(destination, result)
            print(
                f"{args.shape} diagnostic {update}/{args.updates}: loss={total:.4f} "
                f"{entry['seconds']:.1f}s | live={memory['live_bytes']/2**30:.1f}GiB "
                f"driver={memory['driver_bytes']/2**30:.1f}GiB",
                flush=True,
            )
        result["status"] = "completed"
    except Exception as error:
        result["status"] = "stopped"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        dump(destination, result)
        print(f"{result['status']}: {destination}", flush=True)


if __name__ == "__main__":
    main()
