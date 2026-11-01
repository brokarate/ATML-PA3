from pathlib import Path

import pytest

from task3_inference.scoring import levenshtein_distance, load_mathvista_normalizer


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("kitten", "sitting", 3),
        ("", "abc", 3),
        ("same", "same", 0),
        ("café", "cafe", 1),
        ("ab", "ba", 2),
        ("abc", "", 3),
    ],
)
def test_exact_unicode_edit_distance(left, right, expected):
    assert levenshtein_distance(left, right) == expected


def test_official_scorer_without_image_dependencies():
    source = Path("external/MathVista/evaluation/calculate_score.py")
    if not source.exists():
        pytest.skip("Official MathVista checkout not installed")
    normalize = load_mathvista_normalizer(source)
    assert (
        normalize("(B) blue", ["red", "blue"], "multi_choice", "text", None) == "blue"
    )
    # Equal edit distances retain the upstream first-choice tie rule.
    assert normalize("bat", ["cat", "hat"], "multi_choice", "text", None) == "cat"
    assert normalize("4.8", None, "free_form", "integer", None) == "4"
    assert normalize("1.239", None, "free_form", "float", 2) == "1.24"
    assert normalize("not a number", None, "free_form", "float", 2) is None
