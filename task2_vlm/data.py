from common.device import model_backend_kwargs
from common.device import (
    memory_stats,
    require_accelerator,
    reset_peak_memory,
    synchronize,
)
import collections
from pathlib import Path
import numpy as np
import torch
from common import vqa
from common import SEED, dump, fingerprint, jsonl, read, seed_all


def annotations(root, split):
    def get(name):
        return read(Path(root) / name)

    caps = get(f"captions_{split}2014.json")
    instances = get(f"instances_{split}2014.json")
    questions = get(f"v2_OpenEnded_mscoco_{split}2014_questions.json")
    answers = get(f"v2_mscoco_{split}2014_annotations.json")
    pairs = get(f"v2_mscoco_{split}2014_complementary_pairs.json")
    return caps, instances, questions, answers, pairs


def natural_pool(questions, annotations_, pairs):
    qs = {x["question_id"]: x for x in questions["questions"]}
    ans = {x["question_id"]: x for x in annotations_["annotations"]}
    result, seen = [], set()
    for pair in pairs:
        x, y = (
            pair
            if isinstance(pair, list)
            else (pair["question_id_1"], pair["question_id_2"])
        )
        key = tuple(sorted((int(x), int(y))))
        if key in seen:
            continue
        seen.add(key)
        records = []
        for qid in key:
            a, q = ans[qid], qs[qid]
            refs = [r["answer"] for r in a["answers"]]
            counts = collections.Counter(vqa.normalize(s) for s in refs)
            winner, count = counts.most_common(1)[0]
            if len(refs) != 10 or count < 8:
                break
            records.append(
                {
                    "image_id": q["image_id"],
                    "question_id": qid,
                    "question": q["question"],
                    "answer": next(s for s in refs if vqa.normalize(s) == winner),
                    "canonical": winner,
                    "references": refs,
                    "type": a["answer_type"],
                    "kind": "natural",
                }
            )
        if len(records) != 2:
            continue
        a, b = records
        if (
            a["image_id"] == b["image_id"]
            or a["question"] != b["question"]
            or a["canonical"] == b["canonical"]
            or a["type"] != b["type"]
        ):
            continue
        result.append(
            {
                "pair_id": f"vqa:{key[0]}:{key[1]}",
                "type": a["type"],
                "examples": records,
            }
        )
    return result


def take_pairs(pool, quota, rng, blocked=None):
    selected = []
    blocked = blocked if blocked is not None else set()
    for idx in rng.permutation(len(pool)):
        row = pool[int(idx)]
        kind = row["type"]
        images = {e["image_id"] for e in row["examples"]}
        if quota.get(kind, 0) and not images & blocked:
            selected.append(row)
            quota[kind] -= 1
    if any(quota.values()):
        raise ValueError(f"Insufficient eligible image-disjoint pairs: {quota}")
    return selected


def grounding_pool(instances, eligible, rng, max_per_category=4096):
    counts = collections.defaultdict(collections.Counter)
    for a in instances["annotations"]:
        # COCO instances (including crowd annotations) are annotation counts, not inferred real counts.
        counts[a["image_id"]][a["category_id"]] += 1
    rows = []
    ids = sorted(eligible)
    for c in sorted(instances["categories"], key=lambda x: x["id"]):
        cat, word = c["id"], c["name"]
        present = [i for i in ids if counts[i][cat] > 0]
        absent = [i for i in ids if counts[i][cat] == 0]
        if present and absent:
            for a in rng.permutation(present)[:max_per_category]:
                b = int(rng.choice(absent))
                q = f"Does this image contain {word}?"
                rows.append(
                    pair_record(
                        f"presence:{cat}:{a}:{b}", "presence", q, int(a), "yes", b, "no"
                    )
                )
        by_count = {n: [i for i in ids if counts[i][cat] == n] for n in range(1, 5)}
        for n in range(1, 5):
            other = [m for m in range(1, 5) if m != n and by_count[m]]
            if not other:
                continue
            for a in rng.permutation(by_count[n])[: max_per_category // 4]:
                m = int(rng.choice(other))
                b = int(rng.choice(by_count[m]))
                rows.append(
                    pair_record(
                        f"counting:{cat}:{a}:{b}",
                        "counting",
                        f"How many {word} can you see?",
                        int(a),
                        str(n),
                        b,
                        str(m),
                    )
                )
    return rows


def pair_record(pid, kind, q, a, aa, b, ba):
    return {
        "pair_id": pid,
        "type": kind,
        "examples": [
            {
                "image_id": im,
                "question_id": f"{pid}:{i}",
                "question": q,
                "answer": ans,
                "canonical": vqa.normalize(ans),
                "references": [ans],
                "type": kind,
                "kind": "grounding",
            }
            for i, (im, ans) in enumerate(((a, aa), (b, ba)))
        ],
    }


def prepare(args):
    seed_all()
    vqa.configure(args.vqa_eval)
    rng = np.random.default_rng(SEED)
    train_cap, train_ins, tq, ta, tp = annotations(args.annotations, "train")
    val_cap, val_ins, vq, va, vp = annotations(args.annotations, "val")
    natural_train = take_pairs(
        natural_pool(tq, ta, tp), {"yes/no": 640, "number": 256, "other": 640}, rng
    )
    val_natural = natural_pool(vq, va, vp)
    natural_dev = take_pairs(val_natural, {"yes/no": 12, "number": 8, "other": 12}, rng)
    used_dev = {e["image_id"] for p in natural_dev for e in p["examples"]}
    natural_test = take_pairs(
        val_natural, {"yes/no": 36, "number": 24, "other": 36}, rng, used_dev
    )
    used_test = {e["image_id"] for p in natural_test for e in p["examples"]}
    # Partition remaining val images BEFORE generating grounding pools. Development and
    # final-test image sets are globally disjoint across both suites and captions.
    remaining = [
        i["id"] for i in val_cap["images"] if i["id"] not in used_dev | used_test
    ]
    rng.shuffle(remaining)
    half = len(remaining) // 2
    gd = grounding_pool(val_ins, set(remaining[:half]), rng, 128)
    gt = grounding_pool(val_ins, set(remaining[half:]), rng, 128)
    grounding_dev = take_pairs(gd, {"presence": 12, "counting": 12}, rng)
    grounding_test = take_pairs(gt, {"presence": 32, "counting": 32}, rng)
    used_test |= {e["image_id"] for p in grounding_test for e in p["examples"]}
    used_dev |= {e["image_id"] for p in grounding_dev for e in p["examples"]}
    train_ids = {i["id"] for i in train_cap["images"]}
    grounding_train = grounding_pool(train_ins, train_ids, rng)
    cap_train, cap_dev = [], []
    for caps, eligible, number, dest, per_image in [
        (train_cap, train_ids, 6000, cap_train, 3),
        (val_cap, set(remaining[:half]) - used_dev - used_test, 96, cap_dev, 1),
    ]:
        by_image = collections.defaultdict(list)
        for a in sorted(caps["annotations"], key=lambda x: x["id"]):
            by_image[a["image_id"]].append(a)
        candidates = sorted(i for i in eligible if len(by_image[i]) >= per_image)
        if len(candidates) < number:
            raise ValueError("Insufficient caption images")
        for im in rng.choice(candidates, number, replace=False):
            for c in by_image[int(im)][:per_image]:
                dest.append(
                    {
                        "image_id": int(im),
                        "caption_id": c["id"],
                        "question": "Describe this image.",
                        "answer": c["caption"],
                        "kind": "caption",
                        "type": "caption",
                    }
                )
    outputs = {
        "caption_train": cap_train,
        "caption_dev": cap_dev,
        "natural_train": natural_train,
        "natural_dev": natural_dev,
        "natural_test": natural_test,
        "grounding_train": grounding_train,
        "grounding_dev": grounding_dev,
        "grounding_test": grounding_test,
    }
    out = Path(args.output)
    for name, rows in outputs.items():
        jsonl(out / f"{name}.jsonl", rows)
    # A deterministic unrelated image for every individual example; exclude both source images.
    mappings = {}
    for suite in ("grounding", "natural"):
        pool = outputs[f"{suite}_test"]
        images = sorted({e["image_id"] for p in pool for e in p["examples"]})
        mappings[suite] = {}
        for p in pool:
            own = {e["image_id"] for e in p["examples"]}
            choices = [i for i in images if i not in own]
            for e in p["examples"]:
                mappings[suite][str(e["question_id"])] = int(rng.choice(choices))
    dump(out / "mismatched.json", mappings)
    manifests = {"train": set(), "dev": set(), "test": set()}
    for name, rows in outputs.items():
        split = name.rsplit("_", 1)[1]
        for r in rows:
            for e in r.get("examples", [r]):
                manifests[split].add(e["image_id"])
    if manifests["dev"] & manifests["test"] or manifests["train"] & (
        manifests["dev"] | manifests["test"]
    ):
        raise AssertionError("Image leakage across splits")
    dump(out / "image_ids.json", {k: sorted(v) for k, v in manifests.items()})
    sources = {
        str(p.name): fingerprint(p) for p in Path(args.annotations).glob("*.json")
    }
    vqa_source = Path(args.vqa_eval)
    if vqa_source.is_dir():
        vqa_source = vqa_source / "PythonEvaluationTools/vqaEvaluation/vqaEval.py"
    sources["official_vqa_normalizer"] = fingerprint(vqa_source)
    dump(out / "sources.json", sources)
    for name, rows in outputs.items():
        if name.startswith("caption"):
            manifest = [
                {"image_id": r["image_id"], "caption_id": r["caption_id"]} for r in rows
            ]
        else:
            manifest = [
                {
                    "pair_id": r["pair_id"],
                    "type": r["type"],
                    "examples": [
                        {k: e[k] for k in ("image_id", "question_id")}
                        for e in r["examples"]
                    ],
                }
                for r in rows
            ]
        dump(Path("manifests/task2") / f"{name}_ids.json", manifest)
    for name in ("image_ids.json", "mismatched.json", "sources.json"):
        dump(Path("manifests/task2") / name, read(out / name))


def cache(args):
    from PIL import Image
    from transformers import AutoImageProcessor, SiglipVisionModel

    device = require_accelerator()
    seed_all()
    model = (
        SiglipVisionModel.from_pretrained(
            "google/siglip-base-patch16-224",
            torch_dtype=torch.float32,
            **model_backend_kwargs(device),
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    manifests = read(Path(args.data) / "image_ids.json")
    ids = sorted(set().union(*(set(x) for x in manifests.values())))
    train_ids = set(manifests["train"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    import time

    start = time.perf_counter()
    reset_peak_memory(device)
    for offset in range(0, len(ids), args.batch_size):
        chunk = ids[offset : offset + args.batch_size]
        images = []
        for im in chunk:
            split = "train" if im in train_ids else "val"
            path = (
                Path(args.images) / f"{split}2014" / f"COCO_{split}2014_{im:012d}.jpg"
            )
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            # SiglipVisionModel's last_hidden_state has already received post_layernorm.
            features = model(**inputs).last_hidden_state
        if tuple(features.shape[1:]) != (196, 768):
            raise AssertionError(f"Unexpected SigLIP feature shape {features.shape}")
        for im, feat in zip(chunk, features):
            torch.save(feat.half().cpu(), out / f"{im}.pt")
    synchronize(device)
    dump(
        out / "resources.json",
        {
            "seconds": time.perf_counter() - start,
            **memory_stats(device),
            "images": len(ids),
            "model_revision": getattr(model.config, "_commit_hash", None),
            "normalization": "SiglipVisionModel.last_hidden_state, no extra layer norm",
            "processor": processor.to_dict(),
        },
    )


def flatten(pairs):
    return [
        {
            **e,
            "pair_id": p["pair_id"],
            "partner_image_id": p["examples"][1 - i]["image_id"],
            "partner_answer": p["examples"][1 - i]["canonical"],
        }
        for p in pairs
        for i, e in enumerate(p["examples"])
    ]
