"""Random tiny models only: no downloaded weights, no benchmark measurements."""

import torch
from torch import nn
import pytest

transformers = pytest.importorskip("transformers")
from transformers import Qwen3Config, Qwen3ForCausalLM
from common import load_partial
from common.adapters import adapter_state, inject, export_factors, merge_into
from task2_vlm.model import VLM
from test_visual_interface import Tokenizer


def test_qwen_adapter_checkpoint_and_merge():
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    torch.manual_seed(6304)
    base = Qwen3ForCausalLM(config)
    fresh = Qwen3ForCausalLM(config)
    fresh.load_state_dict(base.state_dict())
    inject(base, rank=4, alpha=8, protection=4)
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base.config.use_cache = False
    base.train()
    ids = torch.tensor([[1, 2, 3, 4]])
    base(ids).logits.square().mean().backward()
    assert all(
        p.grad is None for n, p in base.named_parameters() if n.endswith("base.weight")
    )
    with torch.no_grad():
        for n, p in base.named_parameters():
            if n.endswith(".B"):
                p.normal_(std=0.01)
    saved = adapter_state(base)
    load_partial(base, saved)
    with pytest.raises(ValueError, match="checkpoint keys"):
        load_partial(base, {k: v for k, v in saved.items() if k != next(iter(saved))})
    base.eval()
    fresh.eval()
    merge_into(fresh, export_factors(base))
    torch.testing.assert_close(
        base(ids).logits, fresh(ids).logits, rtol=1e-4, atol=1e-5
    )


def test_qwen_visual_generation_positions(tmp_path):
    # Small layer count; width matches the real connector so the actual chat/visual
    # embedding and GenerationMixin code paths execute without model downloads.
    torch.manual_seed(6304)
    model = VLM.__new__(VLM)
    nn.Module.__init__(model)
    model.tokenizer = Tokenizer()
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=1024,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        eos_token_id=None,
        pad_token_id=0,
    )
    model.backbone = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    model.connector = nn.Sequential(
        nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, 1024)
    )
    model.register_buffer("scale", torch.tensor(1.0))
    model.compressed = False
    model.eval()
    torch.save(torch.randn(196, 768).half(), tmp_path / "1.pt")
    rows = [
        {"image_id": 1, "question": q, "answer": "yes", "kind": "grounding"}
        for q in ("hi", "longer")
    ]
    inputs, _, _ = model.batch(rows, tmp_path, generation=True)
    with torch.no_grad():
        expected_first = model.backbone(**inputs).logits[:, -1].argmax(-1)
        extra = model.backbone.get_input_embeddings()(expected_first[:, None])
        inputs["inputs_embeds"] = torch.cat((inputs["inputs_embeds"], extra), dim=1)
        inputs["attention_mask"] = torch.cat(
            (inputs["attention_mask"], torch.ones((2, 1), dtype=torch.long)), dim=1
        )
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        expected_second = model.backbone(**inputs).logits[:, -1].argmax(-1)
        gen_inputs, _, _ = model.batch(rows, tmp_path, generation=True)
        gen_inputs.pop("position_ids")
        generated = model.backbone.generate(
            **gen_inputs,
            do_sample=False,
            max_new_tokens=2,
            use_cache=True,
            pad_token_id=0,
        )
    assert generated.shape == (2, 2)
    torch.testing.assert_close(generated[:, 0], expected_first)
    torch.testing.assert_close(generated[:, 1], expected_second)
