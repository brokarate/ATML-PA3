from common.device import (
    autocast_for,
    make_scaler,
    memory_stats,
    require_accelerator,
    reset_peak_memory,
    synchronize,
)
import collections
import math
import time
from pathlib import Path
import numpy as np
import torch
from common.adapters import adapter_state
from common import (
    SEED,
    dump,
    environment,
    jsonl,
    load_jsonl,
    lr_factor,
    restore_rng,
    rng_state,
    seed_all,
    load_partial,
    read,
)
from .model import VLM
from .data import flatten
from common import fingerprint, rewind_log


def connector_gradient_status(connector):
    """Distinguish a disconnected connector from native AMP scale overflow."""
    gradients = [p.grad for p in connector.parameters() if p.requires_grad]
    if not gradients or any(g is None for g in gradients):
        return "missing"
    if not all(bool(torch.isfinite(g).all()) for g in gradients):
        return "nonfinite"
    return "finite_nonzero" if any(bool(g.ne(0).any()) for g in gradients) else "zero"


def merge_memory_peaks(*measurements):
    return {
        key: max((m.get(key, 0) for m in measurements), default=0)
        for key in ("peak_gpu_bytes", "peak_gpu_driver_bytes")
        if any(key in m for m in measurements)
    }


def development(model, data, features):
    device = model.scale.device
    was_training = model.training
    model.eval()
    results = {}
    with torch.no_grad(), autocast_for(device):
        for suite in ("caption", "grounding", "natural"):
            rows = load_jsonl(Path(data) / f"{suite}_dev.jsonl")
            if suite != "caption":
                rows = flatten(rows)
            conditions = (
                ("correct",) if suite == "caption" else ("correct", "partner", "none")
            )
            for condition in conditions:
                losses = []
                for offset in range(0, len(rows), 8):
                    losses.extend(
                        model.loss(
                            rows[offset : offset + 8],
                            features,
                            condition,
                            answer_nll=suite != "caption",
                        )
                        .cpu()
                        .tolist()
                    )
                results[f"{suite}_{condition}"] = float(np.mean(losses))
            if suite != "caption":
                results[f"{suite}_swap_penalty"] = (
                    results[f"{suite}_partner"] - results[f"{suite}_correct"]
                )
    model.train(was_training)
    return results


def train(args):
    device = require_accelerator()
    seed_all()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pools = {
        name: load_jsonl(Path(args.data) / f"{name}.jsonl")
        for name in ("caption_train", "grounding_train", "natural_train")
    }
    data_hashes = {p.name: fingerprint(p) for p in Path(args.data).glob("*.jsonl")}
    cap = pools["caption_train"]
    if len(cap) != 18000 or len(pools["natural_train"]) != 1536:
        raise ValueError("Incorrect prescribed Task 2 dataset sizes")
    presence = [p for p in pools["grounding_train"] if p["type"] == "presence"]
    counting = [p for p in pools["grounding_train"] if p["type"] == "counting"]
    previous = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    model = VLM(args.compressed, previous["revision"] if previous else None).to(device)
    calibration = model.calibrate(cap, args.features)
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.backbone.config.use_cache = False
    scaler = make_scaler(device)
    rng = np.random.default_rng(SEED)
    global_step = 0
    if args.resume:
        if previous.get("device", "cuda") != device.type:
            raise ValueError(
                "Resume training on the same backend to preserve optimizer and RNG behavior"
            )
        if previous["compressed"] != args.compressed:
            raise ValueError("Visual-token configuration mismatch")
        if previous["data_sha256"] != data_hashes:
            raise ValueError("Resume training/development data differs from checkpoint")
        if previous["stage"] > 1:
            model.add_adapters()
        load_partial(model, previous["model"])
        global_step = previous["global_step"]
        rng.bit_generator.state = previous["sampler"]
        rewind_log(out / "train.jsonl", global_step)
        rewind_log(out / "development.jsonl", global_step)
    elif (out / "train.jsonl").exists():
        raise FileExistsError("Use --resume or a fresh output directory")
    dump(out / "calibration.json", calibration)
    environment(
        out
        / (
            f"environment_resume_{global_step}.json" if previous else "environment.json"
        ),
        {
            "compressed": args.compressed,
            "model_revision": getattr(model.backbone.config, "_commit_hash", None),
        },
    )
    if not previous:
        jsonl(
            out / "development.jsonl",
            [{"update": 0, **development(model, args.data, args.features)}],
        )
    stage_settings = [(2250, 1e-3, None), (1500, 1e-4, 2e-5), (750, 5e-5, 1e-5)]
    counts = collections.Counter()
    elapsed_before = previous.get("elapsed", 0.0) if previous else 0.0
    prior_resource_path = out / f"resources_{global_step}.json"
    prior_resource = (
        read(prior_resource_path) if previous and prior_resource_path.exists() else {}
    )
    run_peaks = merge_memory_peaks(prior_resource)
    started = time.perf_counter()
    for stage, (steps, connector_lr, adapter_lr) in enumerate(stage_settings, 1):
        if previous and stage < previous["stage"]:
            continue
        if stage > 1:
            model.add_adapters()
        stage_start_elapsed = elapsed_before + time.perf_counter() - started
        stage_prefix_peaks = {}
        stage_coverage = True
        if previous and stage == previous["stage"]:
            stage_start_elapsed = previous.get(
                "stage_start_elapsed", 0.0 if stage == 1 else elapsed_before
            )
            if "stage_peak_gpu_bytes" in prior_resource:
                stage_prefix_peaks = {
                    key: prior_resource["stage_" + key]
                    for key in ("peak_gpu_bytes", "peak_gpu_driver_bytes")
                    if "stage_" + key in prior_resource
                }
                stage_coverage = prior_resource.get("stage_memory_complete", False)
            elif stage == 1 and prior_resource:
                # Before Stage 2, the old run-to-date peak is also the Stage 1
                # peak. Preserve it when upgrading a run at its first checkpoint.
                stage_prefix_peaks = merge_memory_peaks(prior_resource)
            else:
                stage_coverage = False
        reset_peak_memory(device)
        # Task 1 explicitly excludes biases/1-D parameters from decay; Task 2
        # instead states AdamW weight decay 0.01 without those exclusions.
        groups = [
            {
                "params": list(model.connector.parameters()),
                "lr": connector_lr,
                "weight_decay": 0.01,
            }
        ]
        if stage > 1:
            groups += [
                {
                    "params": [
                        p for p in model.backbone.parameters() if p.requires_grad
                    ],
                    "lr": adapter_lr,
                    "weight_decay": 0.01,
                }
            ]
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: lr_factor(s, steps, math.ceil(0.05 * steps), 0.1)
        )
        first_step = 0
        if previous and stage == previous["stage"]:
            opt.load_state_dict(previous["optimizer"])
            sched.load_state_dict(previous["scheduler"])
            scaler.load_state_dict(previous["scaler"])
            restore_rng(previous["rng"])
            counts.update(previous["mixture_counts"])
            first_step = previous["stage_step"]
        model.train()
        for step in range(first_step, steps):
            update_started = time.perf_counter()
            if stage == 1:
                rows = [cap[(step * 16 + j) % len(cap)] for j in range(16)]
                counts["caption"] += 16
            else:
                rows = []
                kinds = (
                    ("presence", "counting", "caption")
                    if stage == 2
                    else ("natural", "presence", "counting", "caption")
                )
                probs = (0.5, 0.25, 0.25) if stage == 2 else (0.7, 0.1, 0.1, 0.1)
                for _ in range(8):
                    kind = str(rng.choice(kinds, p=probs))
                    if kind == "caption":
                        rows.extend(cap[int(i)] for i in rng.integers(len(cap), size=2))
                    else:
                        pool = {
                            "presence": presence,
                            "counting": counting,
                            "natural": pools["natural_train"],
                        }[kind]
                        rows.extend(pool[int(rng.integers(len(pool)))]["examples"])
                    counts[kind] += 2
            opt.zero_grad(set_to_none=True)
            loss_total = 0.0
            for offset in (0, 8):
                with autocast_for(device):
                    loss = (
                        model.loss(rows[offset : offset + 8], args.features).sum() / 16
                    )
                loss_total += loss.detach().item()
                scaler.scale(loss).backward()
            connector_status = connector_gradient_status(model.connector)
            if connector_status == "missing":
                raise AssertionError(
                    "Connector gradients are missing; check the visual computation graph"
                )
            if stage == 1 and step == 0 and connector_status == "zero":
                raise AssertionError("Connector gradients are finite but all zero")
            # NaN/Inf gradients must reach GradScaler: it skips this optimizer
            # step and lowers its scale. NaN.sum() > 0 is false, which previously
            # misreported initial scale overflow as a disconnected connector.
            if any(
                p.requires_grad
                for n, p in model.backbone.named_parameters()
                if not n.endswith((".A", ".B"))
            ):
                raise AssertionError("Original backbone must remain frozen")
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            old_scale = scaler.get_scale()
            lrs = [g["lr"] for g in opt.param_groups]
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            jsonl(
                out / "train.jsonl",
                [
                    {
                        "update": global_step,
                        "stage": stage,
                        "stage_update": step + 1,
                        "loss": loss_total,
                        "lrs": lrs,
                        "mixture_counts": dict(counts),
                        "grad_norm": float(norm) if torch.isfinite(norm) else None,
                        "skipped_update": scaler.get_scale() < old_scale,
                        "scale_before": old_scale,
                        "scale_after": scaler.get_scale(),
                        "connector_gradient_status": connector_status,
                        "update_seconds": time.perf_counter() - update_started,
                        "elapsed_seconds": elapsed_before
                        + time.perf_counter()
                        - started,
                    }
                ],
                append=True,
            )
            print(
                f"Task 2 stage {stage} | update {global_step}/4500 | "
                f"loss={loss_total:.4f} | scale={old_scale:g}->{scaler.get_scale():g} | "
                f"connector={connector_status} | "
                f"{'SKIPPED (overflow)' if scaler.get_scale() < old_scale else 'UPDATED'}",
                flush=True,
            )
            if global_step % 500 == 0 or step + 1 == steps:
                metrics = development(model, args.data, args.features)
                jsonl(
                    out / "development.jsonl",
                    [{"update": global_step, "stage": stage, **metrics}],
                    append=True,
                )
            if global_step % 250 == 0 or step + 1 == steps:
                synchronize(device)
                saved = {
                    "connector": model.connector.state_dict(),
                    "scale": model.scale,
                }
                state_dict = {
                    "connector." + k: v for k, v in saved["connector"].items()
                }
                state_dict["scale"] = saved["scale"]
                state_dict.update(
                    {
                        "backbone." + k: v
                        for k, v in adapter_state(model.backbone).items()
                    }
                )
                state = {
                    "model": {n: v.cpu() for n, v in state_dict.items()},
                    "revision": model.revision,
                    "device": device.type,
                    "data_sha256": data_hashes,
                    "stage": stage,
                    "stage_step": step + 1,
                    "global_step": global_step,
                    "compressed": args.compressed,
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng": rng_state(),
                    "sampler": rng.bit_generator.state,
                    "mixture_counts": dict(counts),
                    "elapsed": elapsed_before + time.perf_counter() - started,
                    "stage_start_elapsed": stage_start_elapsed,
                }
                torch.save(state, out / f"update_{global_step}.pt")
                stage_memory = memory_stats(device)
                stage_peaks = merge_memory_peaks(stage_prefix_peaks, stage_memory)
                run_peaks = merge_memory_peaks(run_peaks, stage_peaks)
                dump(
                    out / f"resources_{global_step}.json",
                    {
                        "stage": stage,
                        "seconds": state["elapsed"],
                        **stage_memory,
                        **run_peaks,
                        "stage_seconds": state["elapsed"] - stage_start_elapsed,
                        **{"stage_" + k: v for k, v in stage_peaks.items()},
                        "stage_memory_complete": stage_coverage,
                        "memory_scope": "peak_gpu fields: run-to-date; stage_peak fields: current stage, including saved prefix on resume",
                        "trainable_parameters": sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        ),
                    },
                )
                if step + 1 == steps:
                    torch.save(state["model"], out / f"stage_{stage}_weights.pt")
        previous = None
    import shutil

    tracked = Path("results/task2/training")
    tracked.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.json*"):
        shutil.copy2(p, tracked / p.name)
