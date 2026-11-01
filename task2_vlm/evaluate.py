from common.device import (
    autocast_for,
    require_accelerator,
    synchronize,
    reset_peak_memory,
    memory_stats,
)
import collections
import time
from pathlib import Path
import numpy as np
import torch
from common import vqa
from common import bootstrap, dump, jsonl, load_jsonl, read, seed_all, load_partial
from .model import VLM
from .data import flatten


def evaluate(args):
    device = require_accelerator()
    seed_all()
    vqa.configure(args.vqa_eval)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state["global_step"] != 4500:
        raise ValueError(
            "Final-test evaluation is restricted to the completed Stage 3 model"
        )
    model = VLM(state["compressed"], state["revision"]).to(device)
    model.add_adapters()
    load_partial(model, state["model"])
    model.eval()
    model.backbone.config.use_cache = True
    reset_peak_memory(device)
    mismatch = read(Path(args.data) / "mismatched.json")
    all_rows = []
    for suite in ("grounding", "natural"):
        rows = flatten(load_jsonl(Path(args.data) / f"{suite}_test.jsonl"))
        for r in rows:
            r["mismatched_image_id"] = mismatch[suite][str(r["question_id"])]
        for condition in ("correct", "partner", "mismatched", "none"):
            for offset in range(0, len(rows), 8):
                batch = rows[offset : offset + 8]
                synchronize(device)
                start = time.perf_counter()
                with torch.no_grad(), autocast_for(device):
                    predictions, tokens = model.generate(
                        batch, args.features, condition
                    )
                    synchronize(device)
                    seconds = time.perf_counter() - start
                    nll = (
                        model.loss(batch, args.features, condition, answer_nll=True)
                        .cpu()
                        .tolist()
                    )
                for r, pred, tok, loss in zip(batch, predictions, tokens, nll):
                    canonical = vqa.normalize(pred)
                    all_rows.append(
                        {
                            **r,
                            "suite": suite,
                            "condition": condition,
                            "prediction": pred,
                            "normalized": canonical,
                            "score": (
                                vqa.soft_score(pred, r["references"])
                                if suite == "natural"
                                else float(canonical == r["canonical"])
                            ),
                            "exact": float(
                                canonical == r["canonical"] and bool(canonical)
                            ),
                            "follows_partner": float(
                                canonical == r["partner_answer"] and bool(canonical)
                            ),
                            "empty": float(not canonical),
                            "nll": loss,
                            "seconds": seconds / len(batch),
                            "truncated": len(tok) == 16
                            and tok[-1].item()
                            not in (
                                model.tokenizer.eos_token_id,
                                model.tokenizer.pad_token_id,
                            ),
                        }
                    )
    out = Path(args.output)
    jsonl(out / "predictions.jsonl", all_rows)
    summarize(all_rows, out)
    from common import environment

    dump(out / "resources.json", {"device": device.type, **memory_stats(device)})
    environment(out / "environment.json")


def summarize(rows, out):
    summaries, pair_data = {}, {}
    for suite in ("grounding", "natural"):
        for condition in ("correct", "partner", "mismatched", "none"):
            subset = [
                r for r in rows if r["suite"] == suite and r["condition"] == condition
            ]
            grouped = collections.defaultdict(list)
            for r in subset:
                grouped[r["pair_id"]].append(r)
            table = []
            for pid in sorted(grouped):
                pair = grouped[pid]
                if len(pair) != 2:
                    raise AssertionError("Incomplete evaluation pair")
                table.append(
                    {
                        "pair_id": pid,
                        "type": pair[0]["type"],
                        **{
                            metric: float(np.mean([r[metric] for r in pair]))
                            for metric in (
                                "score",
                                "nll",
                                "empty",
                                "seconds",
                                "follows_partner",
                            )
                        },
                        "pair_accuracy": float(all(r["exact"] for r in pair)),
                        "identical_answer": float(
                            pair[0]["normalized"] == pair[1]["normalized"]
                        ),
                    }
                )
            pair_data[suite, condition] = table
            summaries[f"{suite}_{condition}"] = {}
            for kind in ("all", *sorted({p["type"] for p in table})):
                selected = (
                    table if kind == "all" else [p for p in table if p["type"] == kind]
                )
                summaries[f"{suite}_{condition}"][kind] = {
                    metric: bootstrap([p[metric] for p in selected], 4000)
                    for metric in (
                        "score",
                        "nll",
                        "empty",
                        "seconds",
                        "pair_accuracy",
                        "identical_answer",
                        "follows_partner",
                    )
                }
        a, b = pair_data[suite, "correct"], pair_data[suite, "partner"]
        if [x["pair_id"] for x in a] != [x["pair_id"] for x in b]:
            raise AssertionError("Unpaired bootstrap comparison")
        summaries[f"{suite}_correct_advantages"] = {
            "accuracy": bootstrap(
                [x["score"] - y["score"] for x, y in zip(a, b)], 4000
            ),
            "nll": bootstrap([y["nll"] - x["nll"] for x, y in zip(a, b)], 4000),
        }
    dump(Path(out) / "summary.json", summaries)
