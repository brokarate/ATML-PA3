from argparse import Namespace
import json
import numpy as np
from task2_vlm import data as vd
from common import vqa
from task3_inference import offline, adaptive, scoring


class SimpleNormalizer:
    def processDigitArticle(self, s):
        return s.lower().strip()

    def processPunctuation(self, s):
        return s.replace(".", "")


def test_natural_consensus_filter_and_soft_accuracy(monkeypatch):
    monkeypatch.setattr(vqa, "_EVALUATOR", SimpleNormalizer())
    questions = {
        "questions": [
            {"question_id": i, "question": "Is it red?", "image_id": i}
            for i in (1, 2, 3)
        ]
    }
    annotations = {
        "annotations": [
            {
                "question_id": 1,
                "answers": [{"answer": "Yes"}] * 8 + [{"answer": "no"}] * 2,
                "answer_type": "yes/no",
            },
            {
                "question_id": 2,
                "answers": [{"answer": "no"}] * 8 + [{"answer": "yes"}] * 2,
                "answer_type": "yes/no",
            },
            {
                "question_id": 3,
                "answers": [{"answer": "no"}] * 7 + [{"answer": "yes"}] * 3,
                "answer_type": "yes/no",
            },
        ]
    }
    pairs = vd.natural_pool(questions, annotations, [[1, 2], [2, 1], [1, 3]])
    assert len(pairs) == 1
    assert pairs[0]["examples"][0]["answer"] == "Yes"
    assert len(pairs[0]["examples"][0]["references"]) == 10
    # Official leave-one-out formula gives 0.3/0.6/0.9/1 for 1/2/3/4 matching refs.
    for n, expected in ((1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0)):
        assert (
            abs(vqa.soft_score("yes", ["yes"] * n + ["no"] * (10 - n)) - expected)
            < 1e-9
        )
    assert vqa.soft_score("", [""] * 10) == 0


def test_legacy_official_normalizer_loader(tmp_path, monkeypatch):
    source = """class VQAEval:
    def __init__(self, vqa, vqaRes, n=2):
        self.params = {"question_id": vqa.getQuesIds()}
        self.mapping = {"three": "3"}
    def evaluate(self):
        print "legacy Python 2 evaluation method"
    def processPunctuation(self, text):
        return re.sub(r"[?!]", "", text)
    def processDigitArticle(self, text):
        return " ".join(self.mapping.get(w, w) for w in text.lower().split() if w != "the")
    def updateProgress(self, n):
        print n
"""
    path = tmp_path / "vqaEval.py"
    path.write_text(source)
    monkeypatch.setattr(vqa, "_EVALUATOR", None)
    vqa.configure(path)
    assert vqa.normalize("The THREE?!") == "3"


def test_annotation_pairs_counts_and_presence(monkeypatch):
    monkeypatch.setattr(vqa, "_EVALUATOR", SimpleNormalizer())
    instances = {
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [{"image_id": i, "category_id": 1} for i in (1, 2, 2, 3, 3, 3)],
    }
    pools = vd.grounding_pool(instances, {1, 2, 3, 4}, np.random.default_rng(6304), 16)
    assert {p["type"] for p in pools} == {"presence", "counting"}
    for p in pools:
        a, b = p["examples"]
        assert a["image_id"] != b["image_id"]
        assert a["question"] == b["question"]
        assert a["canonical"] != b["canonical"]
        if p["type"] == "presence":
            assert b["image_id"] == 4


def fixture_pool(kind):
    pools = {}
    labels = {}
    for qid in ("dev", "heldout"):
        rows = []
        for i in range(32):
            r = {
                "sample_id": qid,
                "rollout_id": i,
                "answer": "A" if i < 20 else "B",
                "length": 2,
                "logp": [-2.0, -1.0],
                "kl_u_p": [1.0, 2.0],
                "text": "fixture",
                "token_ids": [1, 2],
            }
            if kind == "mathvista":
                r.update(
                    image_attention=[1.0, 1.0],
                    prompt_attention=[1.0, 1.0],
                    generated_attention=[1.0, 1.0],
                )
            rows.append(r)
            labels[qid, str(i)] = i < 20
        pools[qid] = rows
    orders = {q: [list(range(32)) for _ in range(10)] for q in pools}
    return (
        pools,
        labels,
        orders,
        {"development": ["dev"], "evaluation": ["heldout"], "live": ["heldout"]},
    )


def test_full_offline_outputs_and_frozen_development_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "math_equal", lambda a, b: a == b)
    monkeypatch.setattr(offline, "load_pool", lambda root, kind: fixture_pool(kind))
    monkeypatch.setattr(adaptive, "load_pool", lambda root, kind: fixture_pool(kind))
    offline.offline(Namespace(data="unused", output=str(tmp_path / "offline")))
    summary = __import__("pandas").read_csv(tmp_path / "offline/math_summary.csv")
    assert summary[summary.method == "first"].token_cost.eq(2).all()
    adaptive.adaptive(Namespace(data="unused", output=str(tmp_path / "adaptive")))
    before = json.loads((tmp_path / "adaptive/math_frozen_policies.json").read_text())

    def changed(root, kind):
        pools, labels, orders, splits = fixture_pool(kind)
        for key in labels:
            if key[0] == "heldout":
                labels[key] = not labels[key]
        return pools, labels, orders, splits

    monkeypatch.setattr(adaptive, "load_pool", changed)
    adaptive.adaptive(Namespace(data="unused", output=str(tmp_path / "adaptive2")))
    after = json.loads((tmp_path / "adaptive2/math_frozen_policies.json").read_text())
    assert before == after


def test_rollout_ingestion_join_alignment_and_split(tmp_path, monkeypatch):
    import pandas as pd
    from task3_inference import data as td
    from common import dump, load_jsonl, read

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scoring, "math_equal", lambda a, b: a == b)
    monkeypatch.setattr(td, "math_equal", lambda a, b: a == b)
    root = tmp_path / "course"
    root.mkdir()
    (root / "README.MD").write_text("Synthetic test fixture, not course measurements.")
    normalizer = tmp_path / "normalize.py"
    normalizer.write_text(
        "def get_most_similar(prediction, choices):\n    return choices[0]\n\n"
        "def normalize_extracted_answer(a, choices, question_type, answer_type, precision):\n    return a\n"
    )
    records = [
        {
            "dataset": "synthetic",
            "sample_id": str(q),
            "rollout_id": i,
            "text": "answer",
            "token_ids": [3, 4],
            "length": 2,
            "extracted_answer": "1",
            "gold_answer": "1",
        }
        for q in range(1000)
        for i in range(32)
    ]
    generations = pd.DataFrame(records)
    generations.to_parquet(root / "generations.parquet", index=False)
    keys = ["dataset", "sample_id", "rollout_id"]
    probability = generations[keys].copy()
    probability["logp"] = [[-2.0, -1.0]] * len(probability)
    probability["kl_u_p"] = [[1.0, 2.0]] * len(probability)
    # Reverse a table: joins must use explicit keys, never positional alignment.
    probability.iloc[::-1].to_parquet(root / "probability.parquet", index=False)
    attention = generations[keys].copy()
    for field in ("image_attention", "prompt_attention", "generated_attention"):
        attention[field] = [[1.0, 1.0]] * len(attention)
    attention.to_parquet(root / "attention.parquet", index=False)
    schema = {}
    for kind in ("math", "mathvista"):
        schema[kind] = {
            "generations": {
                "path": "generations.parquet",
                "columns": {k: k for k in generations},
            },
            "probability_metrics": {
                "path": "probability.parquet",
                "columns": {k: k for k in probability},
            },
        }
        if kind == "mathvista":
            schema[kind]["attention_metrics"] = {
                "path": "attention.parquet",
                "columns": {k: k for k in attention},
            }
    dump(tmp_path / "schema.json", schema)
    dump(
        tmp_path / "meta.json",
        {
            str(q): {
                "question_type": "free_form",
                "answer_type": "integer",
                "precision": 0,
                "choices": None,
            }
            for q in range(1000)
        },
    )
    args = Namespace(
        schema=tmp_path / "schema.json",
        data=root,
        output=tmp_path / "prepared",
        mathvista_metadata=tmp_path / "meta.json",
        mathvista_normalizer=normalizer,
    )
    td.prepare(args)
    split = read(tmp_path / "prepared/math_splits.json")
    assert len(split["development"]) == 200 and len(split["evaluation"]) == 800
    assert not set(split["development"]) & set(split["evaluation"])
    expected = (
        np.random.default_rng(6304)
        .permutation(sorted(str(q) for q in range(1000)))
        .tolist()
    )
    assert split["evaluation"] == expected[200:]
    assert split["live"] == expected[200:250]
    candidates = load_jsonl(tmp_path / "prepared/math_candidates.jsonl")
    assert len(candidates) == 32000 and not any(
        "gold" in r or "correct" in r for r in candidates
    )
    assert all(r["logp"] == [-2.0, -1.0] for r in candidates)
    assert all(r["answer_group_id"] == 0 for r in candidates)
    sources = read(tmp_path / "prepared/sources.json")
    from common import fingerprint

    for source in (
        root / "README.MD",
        tmp_path / "schema.json",
        tmp_path / "meta.json",
        normalizer,
    ):
        assert sources[str(source)] == fingerprint(source)
    probability.at[0, "logp"] = [-2.0]
    probability.to_parquet(root / "probability.parquet", index=False)
    import pytest

    with pytest.raises(ValueError, match="Unaligned logp"):
        td.prepare(args)
