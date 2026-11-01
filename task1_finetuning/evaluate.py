from pathlib import Path
import torch
from common import SEED, chat_ids, dump, environment, jsonl
from .model import load_condition
from .generation_trace import GenerationTraceMixin, validate_sample_counts


def evaluate(args):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    model, tok = load_condition(args.checkpoint)
    out = Path(args.output)

    # A separate wrapper applies the non-thinking template only for GSM8K.
    class NonThinking(GenerationTraceMixin, HFLM):
        def apply_chat_template(
            self, chat_history, add_generation_prompt=True, **kwargs
        ):
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )

    for tasks, chat in [
        (["gsm8k_cot"], True),
        (["hellaswag", "boolq", "arc_easy"], False),
    ]:
        lm = NonThinking(
            pretrained=model,
            tokenizer=tok,
            batch_size=args.batch_size,
            device=str(model.device),
            dtype="float16",
        )
        result = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=0,
            apply_chat_template=chat,
            gen_kwargs={"do_sample": False, "max_gen_toks": 1024} if chat else None,
            log_samples=True,
            random_seed=SEED,
            numpy_random_seed=SEED,
            torch_random_seed=SEED,
            fewshot_random_seed=SEED,
        )
        expected_counts = {
            "gsm8k_cot": 1319,
            "hellaswag": 10042,
            "boolq": 3270,
            "arc_easy": 2376,
        }
        for task in tasks:
            validate_sample_counts(result["samples"][task], expected_counts[task])
        dump(out / ("gsm8k.json" if chat else "retention.json"), result)
        if chat:
            samples = result["samples"]["gsm8k_cot"]
            compact = []
            for r in samples:
                if r["filter"] != "flexible-extract":
                    continue
                prompt = r["arguments"][0][0]
                if prompt not in lm.generation_trace:
                    raise ValueError("Missing raw generation trace for GSM8K sample")
                compact.append({**r, **lm.generation_trace[prompt]})
            if len(compact) != 1319:
                raise ValueError("Expected all 1319 flexible-extraction GSM8K samples")
            jsonl(out / "gsm8k_samples.jsonl", compact)
    dump(out / "merge_check.json", getattr(model, "_merge_check", {"adapter": False}))
    environment(out / "environment.json")


def load_qualitative_prompts(source):
    import yaml

    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(
            "Supply the original first_plot_questions.yaml; replacement prompts are not permitted"
        )
    prompts = yaml.safe_load(source.read_text())
    if isinstance(prompts, list) and prompts and "paraphrases" in prompts[0]:
        normalized = []
        for row in prompts:
            if len(row["paraphrases"]) != 1:
                raise ValueError(
                    "Expected exactly one original prompt per course YAML record"
                )
            identifier = row["id"]
            format_ = next(
                (f for f in ("json", "template") if identifier.endswith("_" + f)),
                "plain",
            )
            question = (
                identifier if format_ == "plain" else identifier[: -(len(format_) + 1)]
            )
            normalized.append(
                {
                    "question_id": question,
                    "format": format_,
                    "prompt": row["paraphrases"][0],
                    "system": row.get("system"),
                    "source_id": identifier,
                }
            )
        prompts = normalized
    # Normalized manifest avoids guessing the schema of an unavailable course YAML.
    if isinstance(prompts, dict) and "prompts" in prompts:
        prompts = prompts["prompts"]
    if (
        not isinstance(prompts, list)
        or len(prompts) != 24
        or any(
            not isinstance(p, dict)
            or not {"question_id", "format", "prompt"} <= p.keys()
            for p in prompts
        )
    ):
        raise ValueError(
            "Provide all 24 original prompts as a YAML list of {question_id, format, prompt}; preserve original order/text"
        )
    ids = list(
        dict.fromkeys(p["question_id"] for p in prompts if p["format"] == "plain")
    )
    if len(ids) != 8 or any(
        {p["format"] for p in prompts if p["question_id"] == i}
        != {"plain", "json", "template"}
        for i in ids
    ):
        raise ValueError("Expected 8 questions with plain/json/template formats")
    return prompts, ids


def qualitative(args):
    prompts, ids = load_qualitative_prompts(args.prompts)
    model, tok = load_condition(args.checkpoint)
    results = []
    for p in prompts:
        messages = []
        if p.get("system") is not None:
            messages.append({"role": "system", "content": p["system"]})
        messages.append({"role": "user", "content": p["prompt"]})
        x = torch.tensor(
            [chat_ids(tok, messages)],
            device=model.device,
        )
        with torch.no_grad():
            y = model.generate(
                x,
                attention_mask=torch.ones_like(x),
                do_sample=False,
                max_new_tokens=512,
                pad_token_id=tok.pad_token_id,
            )
        tail = y[0, x.shape[1] :].tolist()
        results.append(
            {
                **p,
                "condition": args.condition,
                "response": tok.decode(tail, skip_special_tokens=True),
                "tokens": tail,
                "truncated": len(tail) == 512 and tail[-1] != tok.eos_token_id,
                "requires_annotation": p["question_id"] in (ids[0], ids[-1]),
                "annotation": {
                    k: None
                    for k in (
                        "format_compliance",
                        "coherence_relevance",
                        "repetition",
                        "math_framing",
                        "inappropriate_content",
                        "excerpt",
                    )
                },
            }
        )
    jsonl(Path(args.output) / f"{args.condition}.jsonl", results)
