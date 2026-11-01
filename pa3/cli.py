import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="ATML PA3 prescribed experiments")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, module, function, fields=()):
        p = commands.add_parser(name)
        p.set_defaults(module=module, function=function)
        p.add_argument(
            "--device",
            choices=["auto", "mps", "cuda"],
            default=None,
            help="GPU backend (default: PA3_DEVICE or automatic selection)",
        )
        for flag, kwargs in fields:
            p.add_argument("--" + flag, **kwargs)
        return p

    def required(name):
        return name, {"required": True}

    command(
        "task1-prepare",
        "task1_finetuning.data",
        "prepare",
        [("output", {"default": "data/task1"})],
    )
    command(
        "task1-train",
        "task1_finetuning.train",
        "train",
        [
            (
                "method",
                {
                    "required": True,
                    "choices": ["full", "l2sp", "lora4", "lora16", "oplora16"],
                },
            ),
            ("data", {"default": "data/task1"}),
            ("output", {"default": "checkpoints/task1"}),
            ("resume", {}),
        ],
    )
    command(
        "task1-evaluate",
        "task1_finetuning.evaluate",
        "evaluate",
        [
            ("checkpoint", {}),
            required("output"),
            ("batch-size", {"type": int, "default": 8}),
        ],
    )
    command(
        "task1-qualitative",
        "task1_finetuning.evaluate",
        "qualitative",
        [
            ("checkpoint", {}),
            required("condition"),
            ("prompts", {"default": "first_plot_questions.yaml"}),
            ("output", {"default": "results/task1/qualitative"}),
        ],
    )
    command(
        "task1-cache-probes",
        "task1_finetuning.diagnostics",
        "cache",
        [("data", {"default": "data/task1"}), ("output", {"default": "data/task1"})],
    )
    command(
        "task1-diagnostics",
        "task1_finetuning.diagnostics",
        "analyze",
        [
            required("checkpoint"),
            ("cache", {"default": "data/task1/probe_cache.pt"}),
            required("output"),
        ],
    )
    command(
        "task2-prepare",
        "task2_vlm.data",
        "prepare",
        [
            required("annotations"),
            required("vqa-eval"),
            ("output", {"default": "data/task2"}),
        ],
    )
    command(
        "task2-cache",
        "task2_vlm.data",
        "cache",
        [
            ("data", {"default": "data/task2"}),
            required("images"),
            ("output", {"default": "data/features"}),
            ("batch-size", {"type": int, "default": 32}),
        ],
    )
    command(
        "task2-train",
        "task2_vlm.train",
        "train",
        [
            ("data", {"default": "data/task2"}),
            ("features", {"default": "data/features"}),
            ("output", {"default": "checkpoints/task2"}),
            ("resume", {}),
            ("compressed", {"action": "store_true"}),
        ],
    )
    command(
        "task2-evaluate",
        "task2_vlm.evaluate",
        "evaluate",
        [
            required("checkpoint"),
            required("vqa-eval"),
            ("data", {"default": "data/task2"}),
            ("features", {"default": "data/features"}),
            ("output", {"default": "results/task2"}),
        ],
    )
    command(
        "task3-inspect",
        "task3_inference.data",
        "inspect",
        [
            ("data", {"default": "ATML-PA3-Data"}),
            ("output", {"default": "results/task3/schemas.json"}),
        ],
    )
    command(
        "task3-prepare",
        "task3_inference.data",
        "prepare",
        [
            ("data", {"default": "ATML-PA3-Data"}),
            required("schema"),
            required("mathvista-metadata"),
            required("mathvista-normalizer"),
            ("output", {"default": "data/task3"}),
        ],
    )
    command(
        "task3-offline",
        "task3_inference.offline",
        "offline",
        [
            ("data", {"default": "data/task3"}),
            ("output", {"default": "results/task3/offline"}),
        ],
    )
    command(
        "task3-adaptive",
        "task3_inference.adaptive",
        "adaptive",
        [
            ("data", {"default": "data/task3"}),
            ("output", {"default": "results/task3/adaptive"}),
        ],
    )
    command(
        "task3-live",
        "task3_inference.live",
        "run",
        [
            ("data", {"default": "data/task3"}),
            required("questions"),
            required("revision"),
            required("rollout-readme"),
            ("output", {"default": "results/task3/live"}),
        ],
    )
    command(
        "plots",
        "pa3.plots",
        "main",
        [
            ("results", {"default": "results"}),
            ("checkpoints", {"default": "checkpoints"}),
            ("output", {"default": "report/figures"}),
            ("images", {}),
        ],
    )
    command(
        "audit",
        "pa3.audit",
        "main",
        [("root", {"default": "."}), ("output", {"default": "results/audit.json"})],
    )
    args = parser.parse_args()
    if args.device is not None:
        os.environ["PA3_DEVICE"] = args.device
    import importlib

    module = importlib.import_module(args.module)
    getattr(module, args.function)(args)


if __name__ == "__main__":
    main()
