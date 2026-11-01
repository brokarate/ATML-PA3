from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import bootstrap, dump, jsonl, environment
from .scoring import BUDGETS, groups, scores, select
from .data import load_pool, accuracy


def offline(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    environment(out / "environment.json")
    for kind in ("math", "mathvista"):
        pools, labels, orders, splits = load_pool(args.data, kind)
        methods = [
            "first",
            "consensus",
            "logp",
            "certainty",
            "certainty_rwc",
            "shortest",
        ]
        if kind == "mathvista":
            methods += [
                "vision",
                "vision_rwc",
                "visconf",
                "visconf_rwc",
                "tokenwise",
                "tokenwise_rwc",
            ]
        details = []
        diagnostics = []
        rollout_scores = []
        trajectories = []
        print(
            f"Starting {kind} offline selection: {len(splits['evaluation'])} held-out questions",
            flush=True,
        )
        for question_index, qid in enumerate(splits["evaluation"]):
            base = pools[qid]
            # Fixed illustrative set: first three held-out IDs in saved permutation, independent of results.
            if qid in splits["evaluation"][:3]:
                trajectories.extend({"sample_id": qid, **r} for r in base[:2])
            for order_idx, order in enumerate(orders[qid]):
                ordered = [base[i] for i in order]
                for n in BUDGETS:
                    candidates = ordered[:n]
                    oracle = any(accuracy(c, labels) for c in candidates)
                    group_ids = groups([c["answer"] for c in candidates], kind)
                    distinct = len(set(group_ids))
                    prefix_scores = scores(candidates, include_correlation=False)
                    for method in methods:
                        idx = select(
                            candidates,
                            method,
                            kind,
                            precomputed=prefix_scores,
                            group_ids=group_ids,
                        )
                        chosen = candidates[idx]
                        correct = accuracy(chosen, labels)
                        consumed = candidates[:1] if method == "first" else candidates
                        details.append(
                            {
                                "sample_id": qid,
                                "order": order_idx,
                                "budget": n,
                                "method": method,
                                "selected_id": chosen["rollout_id"],
                                "selected_answer": chosen["answer"],
                                "candidate_ids": [c["rollout_id"] for c in consumed],
                                "accuracy": correct,
                                "oracle": float(oracle),
                                "groups": distinct,
                                "token_cost": sum(c["length"] for c in consumed),
                                "sampling_failure": float(not oracle),
                                "selection_failure": float(oracle and not correct),
                            }
                        )
            sc = scores(base)
            for i, candidate in enumerate(base):
                rollout_scores.append(
                    {
                        "sample_id": qid,
                        "rollout_id": candidate["rollout_id"],
                        "length": candidate["length"],
                        "correct": accuracy(candidate, labels),
                        **{
                            name: (
                                float(values[i]) if name != "correlation" else values[i]
                            )
                            for name, values in sc.items()
                        },
                        **(
                            {
                                "tokenwise_minus_trajectory": float(
                                    sc["tokenwise"][i] - sc["visconf"][i]
                                )
                            }
                            if kind == "mathvista"
                            else {}
                        ),
                    }
                )
            correctness = np.array([accuracy(r, labels) for r in base])
            for metric, values in sc.items():
                if metric == "correlation":
                    diagnostics.append(
                        {
                            "sample_id": qid,
                            "metric": "confidence_vision_correlation",
                            "value": (
                                float(np.mean([v for v in values if v is not None]))
                                if any(v is not None for v in values)
                                else None
                            ),
                        }
                    )
                    continue
                # Quartiles via deterministic ranking; exactly eight candidates in each tail.
                rank = np.argsort(values, kind="stable")

                def rho(a, b):
                    return (
                        float(spearmanr(a, b).statistic)
                        if np.std(a) and np.std(b)
                        else None
                    )

                diagnostics.append(
                    {
                        "sample_id": qid,
                        "metric": metric,
                        "top_minus_bottom_accuracy": float(
                            correctness[rank[-8:]].mean() - correctness[rank[:8]].mean()
                        ),
                        "length_spearman": rho(values, [r["length"] for r in base]),
                        "correctness_spearman": rho(values, correctness),
                    }
                )
            diagnostics.append(
                {
                    "sample_id": qid,
                    "metric": "length_correctness",
                    "value": (
                        float(
                            spearmanr(
                                [r["length"] for r in base], correctness
                            ).statistic
                        )
                        if correctness.std() and np.std([r["length"] for r in base])
                        else None
                    ),
                }
            )
            if (question_index + 1) % 25 == 0:
                print(
                    f"{kind} offline: {question_index + 1}/{len(splits['evaluation'])} questions",
                    flush=True,
                )
        jsonl(out / f"{kind}_per_question.jsonl", details)
        jsonl(out / f"{kind}_signals.jsonl", diagnostics)
        jsonl(out / f"{kind}_rollout_scores.jsonl", rollout_scores)
        jsonl(out / f"{kind}_trajectories.jsonl", trajectories)
        df = pd.DataFrame(details)
        per_question = (
            df.groupby(["sample_id", "budget", "method"])[
                [
                    "accuracy",
                    "oracle",
                    "groups",
                    "token_cost",
                    "sampling_failure",
                    "selection_failure",
                ]
            ]
            .mean()
            .reset_index()
        )
        per_question.to_csv(out / f"{kind}_question_means.csv", index=False)
        summary = (
            per_question.groupby(["budget", "method"])
            .mean(numeric_only=True)
            .reset_index()
        )
        summary.to_csv(out / f"{kind}_summary.csv", index=False)
        intervals = []
        for n in BUDGETS:
            table = per_question[per_question.budget == n].pivot(
                index="sample_id", columns="method", values="accuracy"
            )
            for method in methods:
                intervals.append(
                    {
                        "budget": n,
                        "method": method,
                        "reference": "consensus",
                        **bootstrap((table[method] - table["consensus"]).values, 1000),
                    }
                )
        dump(out / f"{kind}_accuracy_differences.json", intervals)
        print(
            f"Completed {kind} offline selection and paired bootstrap intervals",
            flush=True,
        )
