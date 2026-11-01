"""Figures only; interpretation and report text must be written by the student."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from common import load_jsonl, read


def main(args):
    results, checkpoints, out = (
        Path(args.results),
        Path(args.checkpoints),
        Path(args.output),
    )
    out.mkdir(parents=True, exist_ok=True)

    def save(name):
        plt.tight_layout()
        plt.savefig(out / (name + ".pdf"), bbox_inches="tight")
        plt.savefig(out / (name + ".png"), dpi=180, bbox_inches="tight")
        plt.close()

    paths = sorted((checkpoints / "task1").glob("*/train.jsonl"))
    if paths:
        plt.figure(figsize=(6, 3.4))
        for p in paths:
            df = pd.DataFrame(load_jsonl(p))
            plt.plot(
                df["update"],
                df["sft_loss"].rolling(25, min_periods=1).mean(),
                label=p.parent.name,
            )
        plt.xlabel("Optimizer update")
        plt.ylabel("Response-token SFT loss (25-update mean)")
        plt.legend(fontsize=8)
        save("task1_learning")
        plt.figure(figsize=(6, 3.4))
        for p in paths:
            df = pd.DataFrame(load_jsonl(p))
            plt.plot(df["update"], df["lr"], label=p.parent.name)
        plt.xlabel("Optimizer update")
        plt.ylabel("Learning rate used for update")
        plt.legend(fontsize=8)
        save("task1_learning_rates")
    p = checkpoints / "task1/l2sp/train.jsonl"
    if p.exists():
        df = pd.DataFrame(load_jsonl(p))
        plt.figure(figsize=(6, 3))
        for col in ("sft_loss", "lambda_omega", "total_loss"):
            plt.plot(df["update"], df[col], label=col)
        plt.xlabel("Optimizer update")
        plt.ylabel("Loss / penalty")
        plt.legend()
        save("task1_regularization")
    benchmarks = []
    for p in sorted((results / "task1").glob("*/gsm8k.json")):
        row = {"condition": p.parent.name}
        for source in (p, p.parent / "retention.json"):
            if not source.exists():
                continue
            for task, metrics in read(source)["results"].items():
                metric = (
                    "exact_match,flexible-extract"
                    if task == "gsm8k_cot"
                    else "acc,none" if task == "boolq" else "acc_norm,none"
                )
                if metric not in metrics:
                    raise ValueError(
                        f"Expected stock metric {metric} absent for {task}"
                    )
                row[task] = metrics[metric]
                if "acc,none" in metrics:
                    row[task + "_raw"] = metrics["acc,none"]
        benchmarks.append(row)
    if benchmarks:
        df = pd.DataFrame(benchmarks).set_index("condition")
        df.to_csv(out / "task1_benchmarks.csv")
        if "baseline" in df.index:
            (100 * (df - df.loc["baseline"])).to_csv(out / "task1_changes_pp.csv")
            primary = df[[c for c in df if not c.endswith("_raw")]]
            (100 * (primary - primary.loc["baseline"])).plot.bar(figsize=(7, 3.4))
            plt.axhline(0, color="grey", linewidth=0.8)
            plt.ylabel("Change from baseline (percentage points)")
            save("task1_changes_pp")
        df[[c for c in df if not c.endswith("_raw")]].plot.bar(figsize=(7, 3.4))
        plt.ylabel("Accuracy")
        save("task1_benchmarks")
    diagnostic_rows = []
    for p in sorted((results / "task1").glob("*/diagnostics.json")):
        d = read(p)
        for layer in d["layers"]:
            row = {
                "condition": p.parent.name,
                "layer": layer["layer"],
                "rho128": layer["rho128"],
                "eta": layer["eta"],
                "displacement": d["displacement"],
            }
            for task in ("gsm8k", "hellaswag"):
                values = [
                    r["R"]
                    for r in layer["activation"]
                    if r["id"].startswith(task + ":")
                ]
                row[task + "_R"] = sum(values) / len(values)
            diagnostic_rows.append(row)
    if diagnostic_rows:
        pd.DataFrame(diagnostic_rows).to_csv(out / "task1_diagnostics.csv", index=False)
    resources = []
    for p in sorted((checkpoints / "task1").glob("*/resources_1250.json")):
        resources.append({"method": p.parent.name, **read(p)})
    if resources:
        pd.DataFrame(resources).to_csv(out / "task1_resources.csv", index=False)
    qualitative = []
    for p in sorted((results / "task1/qualitative").glob("*.jsonl")):
        qualitative.extend(load_jsonl(p))
    if qualitative:
        frame = pd.json_normalize(qualitative)
        frame.sort_values(["question_id", "format", "condition"]).to_csv(
            out / "task1_qualitative.csv", index=False
        )
        frame[frame.requires_annotation].sort_values(
            ["question_id", "format", "condition"]
        ).to_csv(out / "task1_annotation_subset.csv", index=False)
    paths = sorted((checkpoints / "task2").glob("development.jsonl"))
    for p in paths:
        df = pd.DataFrame(load_jsonl(p)).set_index("update")
        for name, cols in [
            ("nll", [c for c in df if c.endswith(("correct", "partner", "none"))]),
            ("swap", [c for c in df if c.endswith("swap_penalty")]),
        ]:
            df[cols].plot(figsize=(6, 3.4))
            for boundary in (2250, 3750):
                plt.axvline(boundary, linestyle=":", color="grey")
            plt.ylabel("Reference NLL" if name == "nll" else "Partner − correct NLL")
            plt.xlabel("Main-run update")
            save("task2_development_" + name)
    p = checkpoints / "task2/train.jsonl"
    if p.exists():
        df = pd.DataFrame(load_jsonl(p))
        plt.figure(figsize=(6, 3))
        plt.plot(df["update"], df["loss"].rolling(25, min_periods=1).mean())
        for boundary in (2250, 3750):
            plt.axvline(boundary, linestyle=":", color="grey")
        plt.xlabel("Main-run update")
        plt.ylabel("Example-weighted SFT loss (25-update mean)")
        save("task2_learning")
    stage_resources = []
    previous_seconds = 0.0
    for stage, endpoint in enumerate((2250, 3750, 4500), 1):
        p = checkpoints / f"task2/resources_{endpoint}.json"
        if not p.exists():
            break
        resource = read(p)
        stage_resources.append(
            {
                "stage": stage,
                "end_update": endpoint,
                "stage_seconds": resource["seconds"] - previous_seconds,
                "cumulative_seconds": resource["seconds"],
                "trainable_parameters": resource["trainable_parameters"],
                "cumulative_peak_gpu_bytes": resource["peak_gpu_bytes"],
                "cumulative_peak_gpu_driver_bytes": resource.get(
                    "peak_gpu_driver_bytes"
                ),
                "memory_scope": "run-to-date sampled peak; not an isolated stage peak",
                "stage_peak_gpu_bytes": resource.get("stage_peak_gpu_bytes"),
                "stage_peak_gpu_driver_bytes": resource.get(
                    "stage_peak_gpu_driver_bytes"
                ),
                "stage_memory_complete": resource.get("stage_memory_complete", False),
            }
        )
        previous_seconds = resource["seconds"]
    if stage_resources:
        pd.DataFrame(stage_resources).to_csv(
            out / "task2_stage_resources.csv", index=False
        )
    p = Path("data/features/resources.json")
    if p.exists():
        # Feature extraction is separate from training; never add its peak memory
        # to a training peak. Keep the processor recipe in the original JSON.
        pd.json_normalize(read(p)).to_csv(
            out / "task2_feature_resources.csv", index=False
        )
    p = results / "task2/summary.json"
    if p.exists():
        table, advantages = [], []
        for comparison, groups in read(p).items():
            if comparison.endswith("_correct_advantages"):
                for metric, interval in groups.items():
                    advantages.append(
                        {"comparison": comparison, "metric": metric, **interval}
                    )
            else:
                for kind, metrics in groups.items():
                    for metric, interval in metrics.items():
                        table.append(
                            {
                                "comparison": comparison,
                                "type": kind,
                                "metric": metric,
                                **interval,
                            }
                        )
        pd.DataFrame(table).to_csv(out / "task2_final_intervals.csv", index=False)
        pd.DataFrame(advantages).to_csv(
            out / "task2_correct_advantages.csv", index=False
        )
    p = results / "task2/predictions.jsonl"
    if p.exists():
        df = pd.DataFrame(load_jsonl(p))
        df.groupby(["suite", "condition", "type"])[
            ["score", "exact", "nll", "empty", "seconds", "follows_partner"]
        ].mean().to_csv(out / "task2_types.csv")
        # Example selection is independent of correctness: first three pair IDs per suite.
        selected = []
        for suite in ("grounding", "natural"):
            ids = sorted(df[df.suite == suite].pair_id.unique())[:3]
            selected.extend(
                df[(df.suite == suite) & df.pair_id.isin(ids)].to_dict("records")
            )
        from common import jsonl

        jsonl(out / "task2_qualitative.jsonl", selected)
        if getattr(args, "images", None):
            from PIL import Image
            import textwrap

            for suite in ("grounding", "natural"):
                subset = df[(df.suite == suite) & (df.condition == "correct")]
                for number, pid in enumerate(sorted(subset.pair_id.unique())[:3]):
                    pair = subset[subset.pair_id == pid].sort_values("question_id")
                    fig, axes = plt.subplots(
                        2, 2, figsize=(8, 7), gridspec_kw={"height_ratios": [2, 1]}
                    )
                    for i, (_, row) in enumerate(pair.iterrows()):
                        path = (
                            Path(args.images)
                            / "val2014"
                            / f"COCO_val2014_{int(row.image_id):012d}.jpg"
                        )
                        with Image.open(path) as im:
                            axes[0, i].imshow(im.convert("RGB"))
                        axes[0, i].axis("off")
                        axes[1, i].axis("off")
                        matching = df[
                            (df.suite == suite)
                            & (df.pair_id == pid)
                            & (df.question_id == row.question_id)
                        ]
                        lines = ["Reference: " + str(row["canonical"])]
                        lines += [
                            f"{r['condition']}: {r['prediction']}"
                            for _, r in matching.iterrows()
                        ]
                        axes[1, i].text(
                            0,
                            1,
                            "\n".join(textwrap.fill(s, 45) for s in lines),
                            va="top",
                            fontsize=8,
                        )
                    fig.suptitle(textwrap.fill(pair.iloc[0].question, 90), fontsize=10)
                    save(f"task2_{suite}_pair_{number}")
    for kind in ("math", "mathvista"):
        p = results / f"task3/offline/{kind}_summary.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.to_csv(out / f"task3_{kind}_methods.csv", index=False)
        df[
            [
                "budget",
                "method",
                "sampling_failure",
                "selection_failure",
                "oracle",
                "accuracy",
                "groups",
            ]
        ].to_csv(out / f"task3_{kind}_failure_decomposition.csv", index=False)
        intervals = results / f"task3/offline/{kind}_accuracy_differences.json"
        if intervals.exists():
            pd.json_normalize(read(intervals)).to_csv(
                out / f"task3_{kind}_accuracy_differences.csv", index=False
            )
        adaptive = results / f"task3/adaptive/{kind}_adaptive_summary.json"
        if adaptive.exists():
            rows = [
                {"family": family, **metrics}
                for family, metrics in read(adaptive).items()
            ]
            pd.json_normalize(rows).to_csv(
                out / f"task3_{kind}_adaptive.csv", index=False
            )
        for name in ("frozen_policies", "development_grid"):
            source = results / f"task3/adaptive/{kind}_{name}.json"
            if source.exists():
                values = read(source)
                if isinstance(values, dict):
                    values = [{"family": key, **value} for key, value in values.items()]
                pd.json_normalize(values).to_csv(
                    out / f"task3_{kind}_{name}.csv", index=False
                )
        for axis in ("budget", "token_cost"):
            plt.figure(figsize=(6.5, 3.8))
            for method, rows in df.groupby("method"):
                plt.plot(rows[axis], rows["accuracy"], marker=".", label=method)
            oracle = df[df.method == "consensus"]
            plt.plot(oracle[axis], oracle.oracle, "k--", label="oracle coverage")
            if axis == "token_cost":
                a = results / f"task3/adaptive/{kind}_adaptive_summary.json"
                if a.exists():
                    for family, metrics in read(a).items():
                        plt.scatter(
                            metrics["token_cost"]["mean"],
                            metrics["accuracy"]["mean"],
                            marker="X",
                            s=65,
                            label=family,
                        )
            plt.xlabel(
                "Candidate budget" if axis == "budget" else "Generated-token cost"
            )
            plt.ylabel("Answer accuracy / coverage")
            plt.legend(fontsize=6, ncol=2)
            save(f"task3_{kind}_{axis}")
        for family in ("agreement", "quality"):
            p = results / f"task3/adaptive/{kind}_{family}_heldout.jsonl"
            if p.exists():
                stops = (
                    pd.DataFrame(load_jsonl(p))["rollouts"]
                    .value_counts(normalize=True)
                    .reindex([4, 8, 16, 32], fill_value=0)
                )
                stops.plot.bar(figsize=(4, 3))
                stops.rename("fraction").rename_axis("rollouts").to_csv(
                    out / f"task3_{kind}_{family}_stops.csv"
                )
                plt.xlabel("Rollouts consumed")
                plt.ylabel("Fraction of question/order trials")
                save(f"task3_{kind}_{family}_stops")
        p = results / f"task3/offline/{kind}_signals.jsonl"
        if p.exists():
            signals = pd.DataFrame(load_jsonl(p))
            signals.groupby("metric").mean(numeric_only=True).to_csv(
                out / f"task3_{kind}_signals.csv"
            )
        p = results / f"task3/offline/{kind}_trajectories.jsonl"
        if p.exists():
            for idx, row in enumerate(load_jsonl(p)):
                plt.figure(figsize=(6, 2.7))
                plt.plot(row["kl_u_p"], label="Token KL(U || p)")
                plt.xlabel("Generated token index")
                plt.ylabel("KL divergence")
                plt.legend()
                save(f"task3_{kind}_trajectory_{idx}_kl")
                if "image_attention" in row:
                    import numpy as np

                    image = np.array(row["image_attention"])
                    total = (
                        image
                        + np.array(row["prompt_attention"])
                        + np.array(row["generated_attention"])
                    )
                    plt.figure(figsize=(6, 2.7))
                    plt.plot(image / total)
                    plt.xlabel("Generated token index")
                    plt.ylabel("Normalized visual engagement")
                    save(f"task3_{kind}_trajectory_{idx}_vision")
    p = results / "task3/live/per_question.jsonl"
    if p.exists():
        df = pd.DataFrame(load_jsonl(p))
        df.groupby("strategy").mean(numeric_only=True).to_csv(out / "task3_live.csv")
        counts = df.groupby("strategy").agg(
            questions=("sample_id", "nunique"),
            correct=("accuracy", "sum"),
            extracted=("extracted", "sum"),
            truncated_calls=("truncated_calls", "sum"),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            seconds=("seconds", "sum"),
        )
        counts.to_csv(out / "task3_live_totals.csv")
        df[df.strategy == "refinement"].groupby("strategy")[
            ["wrong_to_correct", "correct_to_wrong"]
        ].sum().to_csv(out / "task3_live_revision_transitions.csv")
        means = df.groupby("strategy").mean(numeric_only=True)
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for axis, column, label in zip(
            axes,
            ("accuracy", "output_tokens", "seconds"),
            ("Answer accuracy", "Mean generated tokens", "Mean seconds per question"),
        ):
            means[column].plot.bar(ax=axis)
            axis.set_ylabel(label)
        save("task3_live_comparison")
