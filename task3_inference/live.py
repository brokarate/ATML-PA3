from common.device import model_backend_kwargs
from common.device import (
    memory_stats,
    require_accelerator,
    reset_peak_memory,
    synchronize,
)
import time
from pathlib import Path
import torch
from common import (
    SEED,
    dump,
    environment,
    fingerprint,
    jsonl,
    load_jsonl,
    read,
    seed_all,
)
from .scoring import NO_ANSWER, groups, math_equal, vote

INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVISION = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."


def validate_completed_question(payload, qid):
    calls, strategies = payload["calls"], payload["strategies"]
    if (
        len(calls) != 5
        or {r["call"] for r in calls}
        != {"long", "parallel0", "parallel1", "parallel2", "revision"}
        or len(strategies) != 3
        or {r["strategy"] for r in strategies} != {"long", "parallel", "refinement"}
        or any(r["sample_id"] != qid for r in calls + strategies)
    ):
        raise ValueError(f"Incomplete live-generation recovery record for {qid}")
    return calls, strategies


def boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return NO_ANSWER
    start += len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                result = text[start:i].strip()
                return result if result else NO_ANSWER
    return NO_ANSWER


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = require_accelerator()
    if not args.revision or args.revision in ("main", "latest"):
        raise ValueError(
            "Supply the original immutable model revision from the rollout README.MD"
        )
    if not Path(args.rollout_readme).is_file():
        raise FileNotFoundError(
            "Supply original rollout README.MD to verify the extraction rule and model revision"
        )
    if args.revision not in Path(args.rollout_readme).read_text():
        raise ValueError("Model revision does not occur in supplied README.MD")
    seed_all()
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", revision=args.revision
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            revision=args.revision,
            torch_dtype=torch.float16,
            **model_backend_kwargs(device),
        )
        .to(device)
        .eval()
    )
    splits = read(Path(args.data) / "math_splits.json")
    # Manifest maps exact sample IDs to original problem statements, not already modified queries.
    questions = read(args.questions)
    labels = {
        (r["sample_id"], str(r["rollout_id"])): r["gold"]
        for r in load_jsonl(Path(args.data) / "math_labels.jsonl")
    }
    gold = {qid: value for (qid, _), value in labels.items()}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    journal = out / "questions"
    journal.mkdir(exist_ok=True)
    run_config = {
        "revision": args.revision,
        "device": str(device),
        "questions_sha256": fingerprint(args.questions),
        "splits_sha256": fingerprint(Path(args.data) / "math_splits.json"),
        "rollout_readme_sha256": fingerprint(args.rollout_readme),
        "seed": SEED,
        "instruction": INSTRUCTION,
        "revision_instruction": REVISION,
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0,
        },
        "caps": {"long": 3072, "parallel": 1024, "revision": 2048},
    }
    config_path = out / "run_config.json"
    if config_path.exists():
        if read(config_path) != run_config:
            raise ValueError(
                "Live recovery configuration changed; use a fresh output directory"
            )
    elif (out / "calls.jsonl").exists() or any(journal.glob("*.json")):
        raise ValueError(
            "Existing live outputs have no matching recovery configuration"
        )
    else:
        dump(config_path, run_config)
    records, summary_rows = [], []
    completed_seconds = 0.0

    def generate(messages, cap, seed):
        seed_all(seed)
        x = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        reset_peak_memory(device)
        synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            y = model.generate(
                x,
                attention_mask=torch.ones_like(x),
                max_new_tokens=cap,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.0,
                pad_token_id=tok.pad_token_id,
            )
        synchronize(device)
        seconds = time.perf_counter() - start
        ids = y[0, x.shape[1] :].tolist()
        text = tok.decode(ids, skip_special_tokens=True)
        return {
            "messages": messages,
            "seed": seed,
            "response": text,
            "answer": boxed(text),
            "input_tokens": x.numel(),
            "output_tokens": len(ids),
            "token_ids": ids,
            "seconds": seconds,
            **memory_stats(device),
            "truncated": len(ids) == cap and ids[-1] != tok.eos_token_id,
        }

    for i, qid in enumerate(splits["live"]):
        saved_question = journal / f"{i:03d}.json"
        if saved_question.exists():
            payload = read(saved_question)
            saved_calls, saved_strategies = validate_completed_question(payload, qid)
            records.extend(saved_calls)
            summary_rows.extend(saved_strategies)
            completed_seconds += payload["wall_seconds"]
            print(
                f"Already completed live question {i + 1}/{len(splits['live'])}: {qid}",
                flush=True,
            )
            continue
        question_started = time.perf_counter()
        print(
            f"Starting live question {i + 1}/{len(splits['live'])}: {qid}", flush=True
        )
        problem = questions[qid]
        if isinstance(problem, dict):
            problem = problem["problem"]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": problem + "\n\n" + INSTRUCTION},
        ]
        calls = {"long": generate(messages, 3072, SEED + i * 5)}
        for j in range(3):
            calls[f"parallel{j}"] = generate(messages, 1024, SEED + i * 5 + j + 1)
        calls["revision"] = generate(
            messages
            + [
                {"role": "assistant", "content": calls["parallel0"]["response"]},
                {"role": "user", "content": REVISION},
            ],
            2048,
            SEED + i * 5 + 4,
        )
        panswers = [calls[f"parallel{j}"]["answer"] for j in range(3)]
        selected = vote(groups(panswers, "math"))[0]
        for name, call in calls.items():
            call["correct"] = call["answer"] != NO_ANSWER and math_equal(
                gold[qid], call["answer"]
            )
            records.append({"sample_id": qid, "call": name, **call})
        for strategy, names, answer in [
            ("long", ["long"], calls["long"]["answer"]),
            ("parallel", ["parallel0", "parallel1", "parallel2"], panswers[selected]),
            ("refinement", ["parallel0", "revision"], calls["revision"]["answer"]),
        ]:
            summary_rows.append(
                {
                    "sample_id": qid,
                    "strategy": strategy,
                    "answer": answer,
                    "accuracy": float(
                        answer != NO_ANSWER and math_equal(gold[qid], answer)
                    ),
                    **{
                        k: sum(calls[n][k] for n in names)
                        for k in ("input_tokens", "output_tokens", "seconds")
                    },
                    "peak_gpu_bytes": max(calls[n]["peak_gpu_bytes"] for n in names),
                    "memory_measurement": calls[names[0]]["memory_measurement"],
                    "device": device.type,
                    "truncated_calls": sum(calls[n]["truncated"] for n in names),
                    "truncation_rate": sum(calls[n]["truncated"] for n in names)
                    / len(names),
                    "call_extraction_rate": sum(
                        calls[n]["answer"] != NO_ANSWER for n in names
                    )
                    / len(names),
                    "extracted": answer != NO_ANSWER,
                    "wrong_to_correct": strategy == "refinement"
                    and not calls["parallel0"]["correct"]
                    and calls["revision"]["correct"],
                    "correct_to_wrong": strategy == "refinement"
                    and calls["parallel0"]["correct"]
                    and not calls["revision"]["correct"],
                }
            )
        wall_seconds = time.perf_counter() - question_started
        payload = {
            "calls": records[-5:],
            "strategies": summary_rows[-3:],
            "wall_seconds": wall_seconds,
        }
        validate_completed_question(payload, qid)
        # Commit both raw calls and derived strategies together. An interruption
        # only repeats the unfinished question; each call has its own fixed seed.
        temporary = saved_question.with_suffix(".tmp")
        dump(temporary, payload)
        temporary.replace(saved_question)
        jsonl(out / "calls.jsonl", records)
        jsonl(out / "per_question.jsonl", summary_rows)
        completed_seconds += wall_seconds
        eta = completed_seconds / (i + 1) * (len(splits["live"]) - i - 1)
        print(
            f"Completed live question {i + 1}/{len(splits['live'])} | "
            f"{wall_seconds:.1f}s | ETA {eta / 3600:.1f}h",
            flush=True,
        )
    # Also reconstruct aggregates when recovering a run whose final aggregate
    # write was interrupted after committing the last question journal.
    jsonl(out / "calls.jsonl", records)
    jsonl(out / "per_question.jsonl", summary_rows)
    environment(
        out / "environment.json",
        {
            "revision": args.revision,
            "rollout_readme_sha256": fingerprint(args.rollout_readme),
            "questions_sha256": fingerprint(args.questions),
        },
    )
