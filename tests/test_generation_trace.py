from types import SimpleNamespace

import pytest
import torch

from task1_finetuning.generation_trace import (
    GenerationTraceMixin,
    generation_metadata,
    validate_sample_counts,
)


@pytest.mark.parametrize(
    "ids,expected,truncated",
    [
        ([1, 2, 9, 0], 3, False),  # EOS followed by batch padding.
        ([1, 2, 3, 4], 4, True),  # Actual token cap, no stop.
        ([1, 2, 3, 9], 4, False),  # EOS exactly at cap is not truncation.
    ],
)
def test_raw_token_accounting(ids, expected, truncated):
    record = generation_metadata(ids, 4, {9}, 0, [], lambda tokens: "answer")
    assert record["generated_tokens"] == expected
    assert record["truncated"] == truncated


def test_stock_stop_at_token_cap_is_not_truncated():
    record = generation_metadata([1, 2, 3, 4], 4, {9}, 0, ["Q:"], lambda ids: "2 Q:")
    assert not record["truncated"]
    assert record["matched_stock_stop"]


def test_harness_filters_do_not_double_the_document_count():
    samples = [
        {"filter": name, "doc_id": i}
        for name in ("strict-match", "flexible-extract")
        for i in range(3)
    ]
    validate_sample_counts(samples, 3)
    with pytest.raises(ValueError):
        validate_sample_counts(samples[:-1], 3)
    with pytest.raises(ValueError):
        validate_sample_counts(samples + [samples[0]], 3)


def test_trace_preserves_base_generation_and_maps_actual_contexts():
    class Base:
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9]))
        tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kw: "answer")

        def tok_batch_encode(self, strings):
            return torch.tensor([[7, 8], [5, 6]]), torch.ones(2, 2)

        def _model_generate(self, **kwargs):
            return torch.tensor([[7, 8, 1, 9, 0], [5, 6, 2, 3, 4]])

    class Traced(GenerationTraceMixin, Base):
        pass

    model = Traced()
    context, _ = model.tok_batch_encode(["long prompt", "short prompt"])
    actual = model._model_generate(context=context, max_length=5, stop=[])
    torch.testing.assert_close(actual, Base()._model_generate())
    assert model.generation_trace["long prompt"]["generated_tokens"] == 2
    assert model.generation_trace["short prompt"]["truncated"]
