from common.device import model_backend_kwargs
from common.device import autocast_for, empty_cache, require_accelerator
import gc
import torch
from common.adapters import export_factors, inject, merge_into
from common import QWEN, chat_ids, read, seed_all, load_partial
from .config import METHODS


def load_condition(checkpoint=None, method=None, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = require_accelerator()
    state = (
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        if checkpoint
        else None
    )
    revision = state["revision"] if state else revision
    if revision is None:
        revision = read("data/task1/selection.json")["model_revision"]
    tok = AutoTokenizer.from_pretrained(QWEN, revision=revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            QWEN,
            revision=revision,
            torch_dtype=torch.float16,
            **model_backend_kwargs(device),
        )
        .to(device)
        .eval()
    )
    if checkpoint:
        method = state["method"]
        rank = METHODS[method][1]
        if rank:
            seed_all()
            inject(model, rank, 2 * rank, protection=128 if method == "oplora16" else 0)
            load_partial(model, state["model"])
            model.eval()
            test = torch.tensor(
                [chat_ids(tok, [{"role": "user", "content": "What is 2 + 2?"}])],
                device=device,
            )
            with torch.no_grad(), autocast_for(device):
                unmerged = model(test).logits.float().cpu()
            factors = export_factors(model)
            del model
            gc.collect()
            empty_cache(device)
            model = (
                AutoModelForCausalLM.from_pretrained(
                    QWEN,
                    revision=revision,
                    torch_dtype=torch.float16,
                    **model_backend_kwargs(device),
                )
                .to(device)
                .eval()
            )
            merge_into(model, factors)
            with torch.no_grad():
                merged = model(test).logits.float().cpu()
            error = (merged - unmerged).abs()
            # FP16 merge rounds backbone weights; keep absolute and relative errors, and fail gross discrepancies.
            check = {
                "max_abs": error.max().item(),
                "rms": error.square().mean().sqrt().item(),
                "relative_rms": (
                    error.square().mean().sqrt()
                    / unmerged.square().mean().sqrt().clamp_min(1e-8)
                ).item(),
            }
            if check["relative_rms"] > 0.02:
                raise RuntimeError(f"Merged adapter check failed: {check}")
            model._merge_check = check
        else:
            model.load_state_dict(state["model"])
    return model, tok
