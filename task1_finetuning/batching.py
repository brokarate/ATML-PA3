"""Shape reuse through masked right padding, without reordering examples."""

import torch.nn.functional as F

MPS_LENGTH_BUCKETS = (512, 1024, 2048)


def pad_to_bucket(inputs, pad_id):
    width = inputs["input_ids"].shape[1]
    target = next((n for n in MPS_LENGTH_BUCKETS if n >= width), None)
    if target is None:
        raise ValueError("Prepared Task 1 example exceeds 2048 tokens; never truncate")
    extra = target - width
    return {
        key: F.pad(inputs[key], (0, extra), value=value)
        for key, value in (
            ("input_ids", pad_id),
            ("labels", -100),
            ("attention_mask", 0),
        )
    }
