import collections
from pathlib import Path
import numpy as np
import pandas as pd
from common import SEED, dump, environment, fingerprint, jsonl, load_jsonl, read
from .scoring import (
    KEYS,
    NO_ANSWER,
    canonical,
    groups,
    math_equal,
    load_mathvista_normalizer,
)


def inspect(args):
    import pyarrow.parquet as pq

    root = Path(args.data)
    paths = list(root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError("ATML-PA3-Data parquet exports not found")
    dump(args.output, {str(p.relative_to(root)): str(pq.read_schema(p)) for p in paths})


def prepare(args):
    cfg = read(args.schema)
    root = Path(args.data)
    if not (root / "README.MD").is_file() and not list(root.rglob("README.MD")):
        raise FileNotFoundError(
            "The supplied README.MD must accompany the rollout exports"
        )
    metadata = read(args.mathvista_metadata) if args.mathvista_metadata else None
    normalizer = (
        load_mathvista_normalizer(args.mathvista_normalizer)
        if args.mathvista_normalizer
        else None
    )
    out = Path(args.output)
    manifest = {str(path): fingerprint(path) for path in root.rglob("README.MD")}
    for path in (args.schema, args.mathvista_metadata, args.mathvista_normalizer):
        if path:
            manifest[str(path)] = fingerprint(path)
    for kind in ("math", "mathvista"):
        conf = cfg[kind]
        tables = []
        for name in ("generations", "probability_metrics", "attention_metrics"):
            if name not in conf:
                continue
            spec = conf[name]
            frame = pd.read_parquet(root / spec["path"])
            # Explicit schema map must come from the actual export README. No guessed columns.
            frame = frame.rename(columns={v: k for k, v in spec["columns"].items()})
            needed = list(dict.fromkeys(KEYS + list(spec["columns"])))
            frame = frame[needed]
            if frame.duplicated(KEYS).any():
                raise ValueError(f"Duplicate rollout join keys: {kind}/{name}")
            manifest[str(root / spec["path"])] = fingerprint(root / spec["path"])
            tables.append(frame)
        joined = tables[0]
        for frame in tables[1:]:
            shared = (set(joined.columns) & set(frame.columns)) - set(KEYS)
            if shared:
                raise ValueError(f"Schema columns duplicated across tables: {shared}")
            joined = joined.merge(
                frame, on=KEYS, how="outer", validate="one_to_one", indicator=True
            )
            if not joined["_merge"].eq("both").all():
                raise ValueError("Unmatched rollout keys across tables")
            joined = joined.drop(columns="_merge")
        joined["sample_id"] = joined["sample_id"].astype(str)
        qids = sorted(joined.sample_id.unique())
        if len(qids) != 1000 or len(joined) != 32000:
            raise ValueError("Expected 1000 questions and 32,000 rollouts per model")
        order = np.random.default_rng(SEED).permutation(qids).tolist()
        split = {
            "development": order[:200],
            "evaluation": order[200:],
            "live": order[200:250] if kind == "math" else [],
        }
        dump(out / f"{kind}_splits.json", split)
        rows = []
        public = []
        for qid in qids:
            subset = joined[joined.sample_id == qid].sort_values("rollout_id")
            if len(subset) != 32:
                raise ValueError("Every question must have 32 unique rollout IDs")
            meta = metadata[qid] if kind == "mathvista" else None
            answers, temporary = [], []
            for r in subset.to_dict("records"):
                for field in ("token_ids", "logp", "kl_u_p") + (
                    ("image_attention", "prompt_attention", "generated_attention")
                    if kind == "mathvista"
                    else ()
                ):
                    r[field] = np.asarray(r[field]).tolist()
                    if len(r[field]) != int(r["length"]):
                        raise ValueError(f"Unaligned {field}: {qid}/{r['rollout_id']}")
                    if field != "token_ids" and not np.isfinite(r[field]).all():
                        raise ValueError("Nonfinite token measurements")
                r["answer"] = canonical(r["extracted_answer"], kind, meta, normalizer)
                answers.append(r["answer"])
                temporary.append(r)
            # Gold is used only for labels, never when forming answer groups.
            gids = groups(answers, kind)
            for r, gid in zip(temporary, gids):
                r["answer_group_id"] = gid
                gold = canonical(r["gold_answer"], kind, meta, normalizer)
                correct = r["answer"] != NO_ANSWER and (
                    math_equal(gold, r["answer"])
                    if kind == "math"
                    else gold == r["answer"]
                )
                rows.append(
                    {
                        "sample_id": qid,
                        "rollout_id": r["rollout_id"],
                        "correct": bool(correct),
                        "gold": gold,
                    }
                )
                public.append(
                    {
                        k: r[k]
                        for k in (
                            "sample_id",
                            "rollout_id",
                            "answer",
                            "answer_group_id",
                            "length",
                            "logp",
                            "kl_u_p",
                            "text",
                            "token_ids",
                        )
                        + (
                            (
                                "image_attention",
                                "prompt_attention",
                                "generated_attention",
                            )
                            if kind == "mathvista"
                            else ()
                        )
                        + tuple(k for k in ("question_text", "truncated") if k in r)
                    }
                )
        jsonl(out / f"{kind}_candidates.jsonl", public)
        jsonl(out / f"{kind}_labels.jsonl", rows)
        rng = np.random.default_rng(SEED)
        orders = {
            qid: [rng.permutation(32).tolist() for _ in range(10)] for qid in qids
        }
        dump(out / f"{kind}_orders.json", orders)
        dump(Path("manifests/task3") / f"{kind}_splits.json", split)
        dump(Path("manifests/task3") / f"{kind}_orders.json", orders)
    dump(out / "sources.json", manifest)
    dump(out / "schema.json", cfg)
    environment(
        out / "environment.json",
        {
            "mathvista_normalizer_sha256": (
                fingerprint(args.mathvista_normalizer)
                if args.mathvista_normalizer
                else None
            ),
            "mathvista_distance": "exact unit-cost Unicode Levenshtein distance; unchanged official normalization functions",
        },
    )


def load_pool(root, kind):
    public = collections.defaultdict(list)
    for r in load_jsonl(Path(root) / f"{kind}_candidates.jsonl"):
        if "correct" in r or "gold" in r:
            raise ValueError("Selector inputs must not contain labels")
        public[r["sample_id"]].append(r)
    labels = {
        (r["sample_id"], str(r["rollout_id"])): r["correct"]
        for r in load_jsonl(Path(root) / f"{kind}_labels.jsonl")
    }
    return (
        public,
        labels,
        read(Path(root) / f"{kind}_orders.json"),
        read(Path(root) / f"{kind}_splits.json"),
    )


def accuracy(candidate, labels):
    return float(labels[candidate["sample_id"], str(candidate["rollout_id"])])
