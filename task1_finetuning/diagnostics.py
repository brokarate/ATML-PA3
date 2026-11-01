from common.device import model_backend_kwargs
from common.device import require_accelerator
from pathlib import Path
import torch
from common import QWEN, chat_ids, dump, load_jsonl, read, seed_all
from .config import METHODS

LAYERS = (0, 13, 27)


def cache(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_all()
    device = require_accelerator()
    revision = read(Path(args.data) / "selection.json")["model_revision"]
    model = (
        AutoModelForCausalLM.from_pretrained(
            QWEN,
            revision=revision,
            torch_dtype=torch.float32,
            **model_backend_kwargs(device),
        )
        .to(device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(QWEN, revision=revision)
    probes = load_jsonl(Path(args.data) / "probes.jsonl")
    values, weights, bases = {}, {}, {}
    captured = {}
    handles = []
    for layer in LAYERS:
        m = model.model.layers[layer].mlp.down_proj
        handles.append(
            m.register_forward_pre_hook(
                lambda module, inputs, l=layer: captured.update(
                    {l: inputs[0][:, -1].detach().float().cpu()}
                )
            )
        )
        w = m.weight.detach().float().cpu()
        u, _, vh = torch.linalg.svd(w, full_matrices=False)
        weights[layer], bases[layer] = w, {"U": u[:, :128], "V": vh[:128].T}
    for row in probes:
        # Match GSM's actual zero-shot harness prompt, rather than a bare question.
        text = row["prompt"]
        tokens = (
            chat_ids(tok, [{"role": "user", "content": text}])
            if row["task"] == "gsm8k"
            else tok.encode(text, add_special_tokens=False)
        )
        with torch.no_grad():
            model(torch.tensor([tokens], device=device))
        values[f"{row['task']}:{row['id']}"] = {
            l: captured[l].squeeze(0) for l in LAYERS
        }
    for h in handles:
        h.remove()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    torch.save(
        {"activations": values, "weights": weights, "bases": bases},
        Path(args.output) / "probe_cache.pt",
    )


def analyze(args):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    rows = []
    for layer in LAYERS:
        name = f"model.layers.{layer}.mlp.down_proj"
        w0 = cached["weights"][layer].float()
        u = cached["bases"][layer]["U"].float()
        if METHODS[state["method"]][1]:
            s = state["model"]
            a, b = s[name + ".A"].float(), s[name + ".B"].float()
            if state["method"] == "oplora16":
                uk, vk = s[name + ".U"].float(), s[name + ".V"].float()
                b, a = b - uk @ (uk.T @ b), a - (a @ vk) @ vk.T
            delta = 2 * b @ a
        else:
            delta = state["model"][name + ".weight"].float() - w0
        norm2 = delta.square().sum()
        rows.append(
            {
                "layer": layer,
                "rho128": (
                    float((u.T @ delta).square().sum() / norm2) if norm2 > 0 else None
                ),
                "eta": float(norm2.sqrt() / w0.norm()),
                "zero_update": bool(norm2 == 0),
                "activation": [
                    {
                        "id": key,
                        "R": float(
                            (delta @ data[layer]).norm()
                            / ((w0 @ data[layer]).norm() + 1e-8)
                        ),
                    }
                    for key, data in cached["activations"].items()
                ],
            }
        )
    displacement = None
    if not METHODS[state["method"]][1]:
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            QWEN, revision=state["revision"], torch_dtype=torch.float32
        )
        # named_parameters removes tied aliases: count each original parameter once.
        displacement = (
            sum(
                (state["model"][n].float() - p.detach()).double().square().sum().item()
                for n, p in base.named_parameters()
            )
            ** 0.5
        )
    dump(
        args.output,
        {
            "method": state["method"],
            "step": state["step"],
            "displacement": displacement,
            "layers": rows,
        },
    )
