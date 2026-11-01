"""Record actual generation tokens while leaving harness scoring unchanged."""

from collections import defaultdict


def generation_metadata(ids, cap, eos_ids, pad_id, stop, decode):
    raw = list(ids)
    retained = []
    ended = False
    for token in raw:
        if token == pad_id and token not in eos_ids:
            break
        retained.append(token)
        if token in eos_ids:
            ended = True
            break
    text = decode(retained)
    stock_stop = any(s and s in text for s in stop)
    return {
        "generated_token_ids": retained,
        "generated_tokens": len(retained),
        "truncated": len(retained) >= cap and not ended and not stock_stop,
        "ended_with_eos": ended,
        "matched_stock_stop": stock_stop,
    }


def validate_sample_counts(samples, expected):
    # Harness 0.4.9.1 logs each document once PER answer filter. GSM8K's
    # strict-match and flexible-extract rows must not be counted as two examples.
    by_filter = defaultdict(list)
    for sample in samples:
        by_filter[sample["filter"]].append(sample["doc_id"])
    if not by_filter or any(
        len(ids) != expected or len(set(ids)) != expected for ids in by_filter.values()
    ):
        raise ValueError("Incomplete, duplicated, or changed benchmark split")


class GenerationTraceMixin:
    def tok_batch_encode(self, strings, *args, **kwargs):
        encoded = super().tok_batch_encode(strings, *args, **kwargs)
        self._generation_contexts = list(strings)
        return encoded

    def _model_generate(self, context, max_length, stop, **kwargs):
        generated = super()._model_generate(
            context=context, max_length=max_length, stop=stop, **kwargs
        )
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (tuple, list)) else [eos])
        trace = getattr(self, "generation_trace", {})
        for prompt, row in zip(self._generation_contexts, generated.tolist()):
            trace[prompt] = generation_metadata(
                row[context.shape[1] :],
                max_length - context.shape[1],
                eos,
                self.tokenizer.pad_token_id,
                stop,
                lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False),
            )
        self.generation_trace = trace
        return generated
