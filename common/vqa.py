"""Official VQA normalization and local scoring via the upstream evaluator.

The official evaluator is loaded from a local checkout instead of reproducing its
large contraction dictionary incompletely. See README for the pinned source.
"""

import re
from pathlib import Path

_EVALUATOR = None


def normalization_class(source, filename):
    """Compile only the three unchanged normalizer methods from the official class.

    The original evaluator's unrelated evaluation/progress methods contain Python 2
    print statements. Extracting normalizer methods avoids rewriting official
    dictionaries or requiring a legacy Python runtime.
    """
    lines = source.splitlines(keepends=True)
    class_start = next(
        (i for i, line in enumerate(lines) if re.match(r"^class VQAEval\b", line)), None
    )
    if class_start is None:
        raise ValueError("Official source contains no VQAEval class")
    methods = []
    for i in range(class_start + 1, len(lines)):
        match = re.match(r"^([ \t]+)def ([A-Za-z_][A-Za-z_0-9]*)\(", lines[i])
        if match:
            methods.append((i, match.group(1), match.group(2)))
    wanted = {"__init__", "processPunctuation", "processDigitArticle"}
    selected = []
    for j, (start, indent, name) in enumerate(methods):
        if name in wanted:
            end = next(
                (other[0] for other in methods[j + 1 :] if other[1] == indent),
                len(lines),
            )
            selected.append("".join(lines[start:end]))
    if len(selected) != 3:
        raise ValueError(
            "Official VQA source must expose exactly the expected normalization methods"
        )
    namespace = {"re": re}
    exec(compile("class VQAEval:\n" + "\n".join(selected), filename, "exec"), namespace)
    return namespace["VQAEval"]


def configure(path):
    global _EVALUATOR
    path = Path(path)
    if path.is_dir():
        path = path / "PythonEvaluationTools/vqaEvaluation/vqaEval.py"
    if not path.is_file():
        raise FileNotFoundError(
            "Supply official GT-Vision-Lab/VQA vqaEval.py with --vqa-eval"
        )
    cls = normalization_class(path.read_text(), str(path))

    class EmptyVQA:
        def getQuesIds(self):
            return []

    _EVALUATOR = cls(EmptyVQA(), None)


def normalize(answer):
    if _EVALUATOR is None:
        raise RuntimeError("Official VQA normalization must be configured before use")
    answer = str(answer).replace("\n", " ").replace("\t", " ").strip()
    return _EVALUATOR.processDigitArticle(_EVALUATOR.processPunctuation(answer))


def soft_score(prediction, answers):
    pred = normalize(prediction)
    refs = [normalize(a) for a in answers]
    if not pred:
        return 0.0
    if len(refs) != 10:
        raise ValueError("VQAv2 scoring requires all ten references")
    return (
        sum(
            min(1.0, sum(pred == a for j, a in enumerate(refs) if j != i) / 3)
            for i in range(10)
        )
        / 10
    )
