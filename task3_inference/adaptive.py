from pathlib import Path
import numpy as np
import pandas as pd
from common import bootstrap, dump, jsonl, environment
from .scoring import stopping, select
from .data import load_pool, accuracy


def policy_results(pools, labels, orders, qids, kind, policy):
    rows = []
    for qid in qids:
        for order_idx, order in enumerate(orders[qid]):
            candidates = [pools[qid][i] for i in order]
            n, idx = stopping(candidates, kind, **policy)
            chosen = candidates[idx]
            full = candidates[select(candidates, "consensus", kind)]
            correct = accuracy(chosen, labels)
            rows.append(
                {
                    "sample_id": qid,
                    "order": order_idx,
                    "rollouts": n,
                    "token_cost": sum(c["length"] for c in candidates[:n]),
                    "accuracy": correct,
                    "selected_answer": chosen["answer"],
                    "selected_id": chosen["rollout_id"],
                    "candidate_ids": [c["rollout_id"] for c in candidates[:n]],
                    "wrong_early_stop": bool(n < 32 and not correct),
                    "recoverable_by_32": bool(
                        n < 32 and not correct and accuracy(full, labels)
                    ),
                    "full_pool_correct_fraction": float(
                        np.mean([accuracy(c, labels) for c in candidates])
                    ),
                }
            )
    return rows


def adaptive(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    environment(out / "environment.json")
    for kind in ("math", "mathvista"):
        print(f"Starting {kind} adaptive-policy development", flush=True)
        pools, labels, orders, splits = load_pool(args.data, kind)
        baseline_policy = {"tau": 1.0, "always": True}
        base = policy_results(
            pools, labels, orders, splits["development"], kind, baseline_policy
        )
        target = np.mean([r["accuracy"] for r in base]) - 0.01
        chosen = {}
        grid_results = []
        # Only development labels enter selection. Freeze both policies before any test calculation.
        for family in ("agreement", "quality"):
            policies = [
                {"tau": t, "gamma": g}
                for t in (0.5, 0.625, 0.75, 0.875, 1.0)
                for g in ((None,) if family == "agreement" else (0.0, 0.1, 0.25, 0.5))
            ]
            policies.append(baseline_policy)
            feasible = []
            for p in policies:
                rows = policy_results(
                    pools, labels, orders, splits["development"], kind, p
                )
                score, cost = np.mean([r["accuracy"] for r in rows]), np.mean(
                    [r["token_cost"] for r in rows]
                )
                result = {
                    "family": family,
                    "policy": p,
                    "accuracy": float(score),
                    "token_cost": float(cost),
                }
                grid_results.append(result)
                print(f"{kind} {family}: checked {p}", flush=True)
                if score >= target - 1e-12:
                    feasible.append(result)
            best = min(
                feasible,
                key=lambda r: (
                    r["token_cost"],
                    -r["accuracy"],
                    r["policy"]["tau"],
                    r["policy"].get("gamma") or 0,
                ),
            )
            chosen[family] = best["policy"]
        dump(out / f"{kind}_frozen_policies.json", chosen)
        dump(out / f"{kind}_development_grid.json", grid_results)
        summaries = {}
        for family, policy in chosen.items():
            print(
                f"Evaluating frozen {kind} {family} policy on held-out questions",
                flush=True,
            )
            rows = policy_results(
                pools, labels, orders, splits["evaluation"], kind, policy
            )
            jsonl(out / f"{kind}_{family}_heldout.jsonl", rows)
            frame = pd.DataFrame(rows)
            averaged = frame.groupby("sample_id")[
                [
                    "accuracy",
                    "token_cost",
                    "rollouts",
                    "wrong_early_stop",
                    "recoverable_by_32",
                ]
            ].mean()
            summaries[family] = {
                k: bootstrap(averaged[k].values, 1000) for k in averaged.columns
            }
            summaries[family]["stopping_counts"] = {
                str(k): int(v) for k, v in frame.rollouts.value_counts().items()
            }
            full_rows = policy_results(
                pools, labels, orders, splits["evaluation"], kind, baseline_policy
            )
            full_mean = pd.DataFrame(full_rows).groupby("sample_id").accuracy.mean()
            summaries[family]["accuracy_minus_fixed32"] = bootstrap(
                (averaged.accuracy - full_mean).values, 1000
            )
        dump(out / f"{kind}_adaptive_summary.json", summaries)
        print(f"Completed {kind} adaptive evaluation", flush=True)
