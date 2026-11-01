from pathlib import Path
import numpy as np
from common import (
    QWEN,
    SEED,
    dump,
    environment,
    jsonl,
    normalize_text,
    read,
    seed_all,
    supervised,
    model_revision,
)


def prepare(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    seed_all()
    revision = model_revision()
    tok = AutoTokenizer.from_pretrained(QWEN, revision=revision)
    test = load_dataset("openai/gsm8k", "main", split="test")
    blocked = {normalize_text(r["question"]) for r in test}
    source = load_dataset("meta-math/MetaMathQA", split="train")
    rows, rejects = [], {"empty": 0, "overlap": 0, "length": 0}
    for i, r in enumerate(source):
        q, a = r.get("query", ""), r.get("response", "")
        if (
            not isinstance(q, str)
            or not isinstance(a, str)
            or not q.strip()
            or not a.strip()
        ):
            rejects["empty"] += 1
            continue
        if any(
            normalize_text(r.get(k, "")) in blocked
            for k in ("query", "original_question")
        ):
            rejects["overlap"] += 1
            continue
        encoded = supervised(tok, q, a)
        if len(encoded["input_ids"]) > 2048:
            rejects["length"] += 1
            continue
        rows.append({"source_id": i, "question": q, "response": a, **encoded})
        if len(rows) == 40000:
            break
    if len(rows) != 40000:
        raise ValueError("Fewer than 40,000 eligible training examples")
    out = Path(args.output)
    jsonl(out / "train.jsonl", rows)
    dump(
        out / "selection.json",
        {
            "source_ids": [r["source_id"] for r in rows],
            "rejected": rejects,
            "dataset_fingerprint": source._fingerprint,
            "model_revision": revision,
        },
    )
    hellaswag = load_dataset("Rowan/hellaswag", split="validation")
    rng = np.random.default_rng(SEED)
    probes = {
        "gsm8k": sorted(rng.choice(len(test), 32, replace=False).tolist()),
        "hellaswag": sorted(rng.choice(len(hellaswag), 32, replace=False).tolist()),
    }
    dump(out / "probe_ids.json", probes)
    # Retention probe prompts follow harness's stock HellaSwag preprocessing.
    from lm_eval.tasks.hellaswag.utils import process_docs

    processed = process_docs(hellaswag)
    from lm_eval.tasks import get_task_dict

    gsm_task = get_task_dict(["gsm8k_cot"])["gsm8k_cot"]
    jsonl(
        out / "probes.jsonl",
        [
            {
                "task": task,
                "id": i,
                "prompt": (
                    gsm_task.doc_to_text(test[i])
                    if task == "gsm8k"
                    else processed[i]["query"]
                ),
            }
            for task, ids in probes.items()
            for i in ids
        ],
    )
    for name in ("selection.json", "probe_ids.json"):
        dump(Path("manifests/task1") / name, read(out / name))
    environment(out / "environment.json")
