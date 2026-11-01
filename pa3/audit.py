from pathlib import Path
from common import dump, load_jsonl, read


def main(args):
    root = Path(args.root)
    checks = []

    def check(name, condition, detail):
        checks.append(
            {"requirement": name, "passed": bool(condition), "detail": detail}
        )

    check(
        "original qualitative prompts",
        (root / "first_plot_questions.yaml").exists(),
        "Course file required, do not replace its text",
    )
    check(
        "supplied rollout data",
        (root / "ATML-PA3-Data").exists(),
        "Course export and original README.MD required",
    )
    p = root / "data/task1/train.jsonl"
    rows = load_jsonl(p) if p.exists() else []
    check("Task 1 retained examples", len(rows) == 40000, f"{len(rows)} / 40000")
    for method in ("full", "l2sp", "lora4", "lora16", "oplora16"):
        p = root / f"checkpoints/task1/{method}/train.jsonl"
        logs = load_jsonl(p) if p.exists() else []
        check(
            f"Task 1 {method} updates",
            len(logs) == 1250 and [r["update"] for r in logs] == list(range(1, 1251)),
            f"{len(logs)} logged updates",
        )
        check(
            f"Task 1 {method} finite updates",
            bool(logs) and not any(r["skipped_update"] for r in logs),
            "Inspect and disclose any skipped updates",
        )
    for condition in (
        "baseline",
        "full_midpoint",
        "full",
        "l2sp",
        "lora4",
        "lora16",
        "oplora16",
    ):
        folder = root / f"results/task1/{condition}"
        check(
            f"Task 1 {condition} benchmarks",
            all((folder / x).exists() for x in ("gsm8k.json", "retention.json")),
            str(folder),
        )
        p = root / f"results/task1/qualitative/{condition}.jsonl"
        q = load_jsonl(p) if p.exists() else []
        check(f"Task 1 {condition} 24 responses", len(q) == 24, str(p))
        annotated = [r for r in q if r.get("requires_annotation")]
        check(
            f"Task 1 {condition} 6 manual annotations",
            len(annotated) == 6
            and all(
                all(value is not None for value in r.get("annotation", {}).values())
                and r.get("annotation")
                for r in annotated
            ),
            str(p),
        )
    p = root / "checkpoints/task2/train.jsonl"
    logs = load_jsonl(p) if p.exists() else []
    check(
        "Task 2 4500 updates",
        len(logs) == 4500 and (not logs or logs[-1]["update"] == 4500),
        f"{len(logs)} logged updates",
    )
    for name in ("grounding", "natural"):
        check(
            f"Task 2 {name} final results",
            (root / "results/task2/summary.json").exists(),
            "Four conditions, pair bootstrap",
        )
    for stage, endpoint in enumerate((2250, 3750, 4500), 1):
        p = root / f"checkpoints/task2/resources_{endpoint}.json"
        resource = read(p) if p.exists() else {}
        check(
            f"Task 2 stage {stage} resources",
            resource.get("stage") == stage
            and resource.get("stage_memory_complete", False)
            and all(
                key in resource
                for key in (
                    "stage_seconds",
                    "stage_peak_gpu_bytes",
                    "trainable_parameters",
                )
            ),
            "Separate stage timing/memory, resume prefix retained; MPS peaks are sampled lower bounds",
        )
    p = root / "results/task2/predictions.jsonl"
    predictions = load_jsonl(p) if p.exists() else []
    expected = {"grounding": 128, "natural": 192}
    for suite, size in expected.items():
        for condition in ("correct", "partner", "mismatched", "none"):
            rows = [
                r
                for r in predictions
                if r["suite"] == suite and r["condition"] == condition
            ]
            check(
                f"Task 2 {suite}/{condition} raw evidence",
                len(rows) == size
                and len({r["question_id"] for r in rows}) == size
                and all(
                    all(
                        k in r
                        for k in (
                            "references",
                            "prediction",
                            "normalized",
                            "score",
                            "exact",
                            "pair_id",
                            "type",
                            "nll",
                            "seconds",
                            "truncated",
                            "empty",
                            "follows_partner",
                        )
                    )
                    for r in rows
                ),
                f"{len(rows)}/{size} per-example records with scoring and resource fields",
            )
    for kind in ("math", "mathvista"):
        for name in (
            f"offline/{kind}_per_question.jsonl",
            f"offline/{kind}_accuracy_differences.json",
            f"adaptive/{kind}_frozen_policies.json",
            f"adaptive/{kind}_adaptive_summary.json",
            f"offline/{kind}_signals.jsonl",
            f"offline/{kind}_rollout_scores.jsonl",
            f"offline/{kind}_trajectories.jsonl",
            f"adaptive/{kind}_development_grid.json",
            f"adaptive/{kind}_agreement_heldout.jsonl",
            f"adaptive/{kind}_quality_heldout.jsonl",
        ):
            check("Task 3 " + name, (root / "results/task3" / name).exists(), name)
    p = root / "results/task3/live/per_question.jsonl"
    logs = load_jsonl(p) if p.exists() else []
    check(
        "Task 3 50 live questions × 3 strategies",
        len(logs) == 150,
        f"{len(logs)} / 150",
    )
    check(
        "Task 3 live cost/extraction/revision fields",
        bool(logs)
        and all(
            all(
                k in row
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "seconds",
                    "peak_gpu_bytes",
                    "truncation_rate",
                    "call_extraction_rate",
                    "extracted",
                    "wrong_to_correct",
                    "correct_to_wrong",
                )
            )
            for row in logs
        ),
        "Actual strategy costs, shared first response charged to both strategies, human interpretation pending",
    )
    dump(
        args.output,
        {
            "complete": all(c["passed"] for c in checks),
            "checks": checks,
            "note": "File-presence audit supplements semantic tests and manual review; it cannot certify experimental correctness.",
        },
    )
    print(
        f"{sum(c['passed'] for c in checks)}/{len(checks)} evidence checks passed. See {args.output}"
    )
