import collections
import functools
import ast
from pathlib import Path
import numpy as np
from scipy.stats import rankdata


NO_ANSWER = "NO_ANSWER"


BUDGETS = (1, 2, 4, 8, 16, 32)


KEYS = ["dataset", "sample_id", "rollout_id"]


@functools.lru_cache(maxsize=200000)
def math_equal(a, b):
    from math_verify import parse, verify

    if a == NO_ANSWER or b == NO_ANSWER:
        return a == b

    # These are already extracted answers, not trajectories to be re-extracted.
    def parsed(s):
        return parse("\\boxed{" + str(s) + "}")

    return bool(verify(parsed(a), parsed(b)))


def missing(a):
    return a is None or (isinstance(a, float) and np.isnan(a)) or not str(a).strip()


def levenshtein_distance(a, b):
    """Exact unit-cost edit distance on Unicode strings (no fuzzy normalization)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    return previous[-1]


def load_mathvista_normalizer(path):
    """Load unchanged official scoring functions without unrelated image/API imports."""
    path = Path(path)
    import re

    source = ast.parse(path.read_text(), filename=str(path))
    wanted = {"get_most_similar", "normalize_extracted_answer"}
    functions = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in functions} != wanted:
        raise ValueError(
            "Official MathVista scorer must contain both normalization functions"
        )
    # Supply the same exact unit-cost edit distance used by python-Levenshtein;
    # preserve both official function bodies, including the first-choice tie rule.
    namespace = {"re": re, "distance": levenshtein_distance}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["normalize_extracted_answer"]


def canonical(a, kind, metadata=None, normalizer=None):
    if missing(a):
        return NO_ANSWER
    if kind == "math":
        return str(a).strip()
    if normalizer is None or metadata is None:
        raise ValueError(
            "MathVista requires official metadata and its released normalizer"
        )
    value = normalizer(
        str(a),
        metadata.get("choices"),
        metadata["question_type"],
        metadata["answer_type"],
        metadata.get("precision"),
    )
    if missing(value):
        return NO_ANSWER
    if metadata["question_type"] == "multi_choice":
        # Official normalizer returns choice text; voting uses option letters.
        choices = metadata["choices"]
        if value not in choices:
            raise ValueError(f"Official normalization returned a non-choice: {value}")
        return chr(65 + choices.index(value))
    return str(value)


def groups(answers, kind):
    representatives, ids = [], []
    for a in answers:
        found = None
        for i, b in enumerate(representatives):
            if math_equal(a, b) if kind == "math" else a == b:
                found = i
                break
        if found is None:
            found = len(representatives)
            representatives.append(a)
        ids.append(found)
    return ids


def scores(candidates, include_correlation=True):
    if not candidates or any(len(r["kl_u_p"]) == 0 for r in candidates):
        raise ValueError(
            "Empty rollout has undefined quality signals; inspect original export"
        )
    result = {
        "logp": np.array([np.mean(r["logp"]) for r in candidates]),
        "certainty": np.array([np.mean(r["kl_u_p"]) for r in candidates]),
        "shortest": -np.array([r["length"] for r in candidates], dtype=float),
    }
    if "image_attention" in candidates[0]:
        pooled = np.concatenate(
            [np.asarray(r["kl_u_p"], dtype=float) for r in candidates]
        )
        lo, hi = np.quantile(pooled, [0.05, 0.95], method="linear")
        vision, fused, token_fused, corr = [], [], [], []
        for r in candidates:
            confidence = np.clip(
                (np.asarray(r["kl_u_p"]) - lo) / max(hi - lo, 1e-8), 0, 1
            )
            image = np.asarray(r["image_attention"], dtype=float)
            total = (
                image
                + np.asarray(r["prompt_attention"])
                + np.asarray(r["generated_attention"])
            )
            if np.any(total <= 0):
                raise ValueError("Nonpositive total group attention")
            v = image / total
            vision.append(v.mean())
            fused.append(v.mean() * confidence.mean())
            token_fused.append((v * confidence).mean())
            corr.append(
                float(np.corrcoef(confidence, v)[0, 1])
                if include_correlation and confidence.std() > 0 and v.std() > 0
                else None
            )
        result.update(
            vision=np.array(vision),
            visconf=np.array(fused),
            tokenwise=np.array(token_fused),
        )
        result["correlation"] = corr
    return result


def vote(group_ids, weights=None):
    if weights is None:
        weights = np.ones(len(group_ids))
    totals = collections.defaultdict(float)
    first = {}
    for i, (g, w) in enumerate(zip(group_ids, weights)):
        totals[g] += float(w)
        first.setdefault(g, i)
    ranking = sorted(totals, key=lambda g: (-totals[g], first[g]))
    return first[ranking[0]], totals, ranking


def rank_weights(values):
    if len(values) == 1:
        return np.ones(1)
    ranks = rankdata(-np.asarray(values), method="average")
    return (len(values) - ranks) / (len(values) - 1)


def select(candidates, method, kind, *, precomputed=None, group_ids=None):
    # Whitelist public candidate fields at the call site; this routine never reads gold/correctness.
    if method == "first":
        return 0
    answer_groups = (
        groups([r["answer"] for r in candidates], kind)
        if group_ids is None
        else group_ids
    )
    if method == "consensus":
        return vote(answer_groups)[0]
    metric = method.removesuffix("_rwc")
    quality = (
        scores(candidates, include_correlation=False)
        if precomputed is None
        else precomputed
    )[metric]
    if method.endswith("_rwc"):
        return vote(answer_groups, rank_weights(quality))[0]
    return int(np.argmax(quality))


def stopping(candidates, kind, tau, gamma=None, always=False):
    if always:
        return len(candidates), select(candidates, "consensus", kind)
    for n in (4, 8, 16, 32):
        current = candidates[:n]
        gids = groups([r["answer"] for r in current], kind)
        idx, totals, ordered = vote(gids)
        unique = len(ordered) == 1 or totals[ordered[0]] > totals[ordered[1]]
        allowed = (
            unique
            and current[idx]["answer"] != NO_ANSWER
            and totals[ordered[0]] / n >= tau
        )
        if allowed and gamma is not None:
            metric = "certainty" if kind == "math" else "visconf"
            ridx, rt, ro = vote(
                gids, rank_weights(scores(current, include_correlation=False)[metric])
            )
            runner = rt[ro[1]] if len(ro) > 1 else 0.0
            margin = (rt[ro[0]] - runner) / max(sum(rt.values()), 1e-12)
            allowed = gids[ridx] == gids[idx] and margin >= gamma
        if allowed or n == 32:
            return n, idx
    raise AssertionError("Unreachable stopping state")
