import copy

import pytest

from task3_inference.live import validate_completed_question


def test_question_recovery_requires_all_calls_and_strategies():
    calls = ["long", "parallel0", "parallel1", "parallel2", "revision"]
    strategies = ["long", "parallel", "refinement"]
    payload = {
        "calls": [{"sample_id": "test:9", "call": name} for name in calls],
        "strategies": [
            {"sample_id": "test:9", "strategy": name} for name in strategies
        ],
    }
    assert validate_completed_question(payload, "test:9") == (
        payload["calls"],
        payload["strategies"],
    )
    for section in ("calls", "strategies"):
        broken = copy.deepcopy(payload)
        broken[section].pop()
        with pytest.raises(ValueError):
            validate_completed_question(broken, "test:9")
    with pytest.raises(ValueError):
        validate_completed_question(payload, "test:10")
