"""Select retrospective evidence records; this writes no report interpretation."""

import argparse
import json
from pathlib import Path
from common import dump, load_jsonl
from task3_inference.scoring import math_equal


def stream(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/task3")
    parser.add_argument("--results", default="results/task3")
    args = parser.parse_args()
    root = Path(args.results)
    selected = []
    for kind in ("math", "mathvista"):
        path = root / f"offline/{kind}_per_question.jsonl"
        if not path.exists():
            continue
        comparisons = {}
        for row in stream(path):
            if (
                row["order"] == 0
                and row["budget"] == 32
                and row["method"] in ("consensus", "certainty_rwc", "visconf_rwc")
            ):
                comparisons.setdefault(row["sample_id"], {})[row["method"]] = row
        quality = "certainty_rwc" if kind == "math" else "visconf_rwc"
        for qid in sorted(comparisons):
            pair = comparisons[qid]
            a, b = (
                pair["consensus"]["selected_answer"],
                pair[quality]["selected_answer"],
            )
            differs = not math_equal(a, b) if kind == "math" else a != b
            if differs:
                selected.append(
                    {
                        "dataset": kind,
                        "sample_id": qid,
                        "category": "selection_disagreement",
                        "rule": "First lexicographic held-out ID with inequivalent selected answers under consensus and quality-RWC at N=32, order=0",
                        "measurements": pair,
                    }
                )
                break
        for family in ("agreement", "quality"):
            path = root / f"adaptive/{kind}_{family}_heldout.jsonl"
            if path.exists():
                failures = [r for r in stream(path) if r["wrong_early_stop"]]
                if failures:
                    row = min(
                        failures,
                        key=lambda r: (
                            not r["recoverable_by_32"],
                            r["sample_id"],
                            r["order"],
                        ),
                    )
                    selected.append(
                        {
                            "dataset": kind,
                            "sample_id": row["sample_id"],
                            "category": "premature_stop_" + family,
                            "rule": "Prefer a recoverable wrong early stop, then first lexicographic held-out ID and earliest order; correctness is used retrospectively only",
                            "measurements": row,
                        }
                    )
        ids = {r["sample_id"] for r in selected if r["dataset"] == kind}
        if ids:
            candidates = [
                r
                for r in stream(Path(args.data) / f"{kind}_candidates.jsonl")
                if r["sample_id"] in ids
            ]
            labels = [
                r
                for r in stream(Path(args.data) / f"{kind}_labels.jsonl")
                if r["sample_id"] in ids
            ]
            dump(
                root / f"examples/{kind}_records.json",
                {"candidates": candidates, "labels_for_retrospective_review": labels},
            )
    path = root / "live/per_question.jsonl"
    if path.exists():
        rows = load_jsonl(path)
        for transition in ("wrong_to_correct", "correct_to_wrong"):
            cases = [r for r in rows if r[transition]]
            if cases:
                row = min(cases, key=lambda r: r["sample_id"])
                selected.append(
                    {
                        "dataset": "math",
                        "sample_id": row["sample_id"],
                        "category": transition,
                        "rule": "First lexicographic live-subset ID exhibiting the specified transition, selected retrospectively",
                        "measurements": row,
                    }
                )
        ids = {
            r["sample_id"]
            for r in selected
            if r["category"] in ("wrong_to_correct", "correct_to_wrong")
        }
        dump(
            root / "examples/live_calls.json",
            [r for r in stream(root / "live/calls.jsonl") if r["sample_id"] in ids],
        )
    dump(
        root / "examples/selection.json",
        {
            "examples": selected,
            "absent_categories": "Categories without qualifying cases are omitted; no case is invented.",
        },
    )


if __name__ == "__main__":
    main()
