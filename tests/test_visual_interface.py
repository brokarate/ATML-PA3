from types import SimpleNamespace
import torch
from torch import nn
from task2_vlm.model import VLM
from task2_vlm.evaluate import summarize


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 40 + 2 for c in text]

    def decode(self, ids, **kwargs):
        return "q" * len(ids)

    def convert_tokens_to_ids(self, token):
        return 1

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "</user><assistant>"


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 1024)
        self.out = nn.Linear(1024, 64)
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, **kwargs):
        # Causal cumulative mixing ensures visual tokens can influence the target.
        return SimpleNamespace(logits=self.out(inputs_embeds.cumsum(1)))


def tiny_model():
    model = VLM.__new__(VLM)
    nn.Module.__init__(model)
    model.backbone = Backbone()
    model.tokenizer = Tokenizer()
    model.connector = nn.Sequential(
        nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, 1024)
    )
    model.register_buffer("scale", torch.tensor(1.0))
    model.compressed = False
    return model


def test_visual_tokens_masks_positions_and_gradient(tmp_path):
    model = tiny_model()
    torch.save(torch.randn(196, 768).half(), tmp_path / "7.pt")
    rows = [
        {"question": "Count?", "answer": "2", "image_id": 7, "kind": "grounding"},
        {
            "question": "Longer question?",
            "answer": "3",
            "image_id": 7,
            "kind": "grounding",
        },
    ]
    full, labels, nll = model.batch(rows, tmp_path)
    none, no_labels, _ = model.batch(rows, tmp_path, "none")
    assert full["inputs_embeds"].shape[1] - none["inputs_embeds"].shape[1] == 196
    assert labels.ne(-100).sum(1).tolist() == [2, 2]
    assert nll.ne(-100).sum(1).tolist() == [1, 1]
    assert full["position_ids"][0, -1] == full["attention_mask"][0].sum() - 1
    assert not labels[full["attention_mask"] == 0].ne(-100).any()
    model.loss(rows, tmp_path).mean().backward()
    assert model.connector[0].weight.grad.norm() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    model.compressed = True
    assert model.feature(7, tmp_path).shape == (49, 768)


def test_pair_bootstrap_preserves_condition_alignment(tmp_path):
    rows = []
    for suite in ("grounding", "natural"):
        for condition in ("correct", "partner", "mismatched", "none"):
            for pair_id in ("x", "y"):
                for i in range(2):
                    rows.append(
                        {
                            "suite": suite,
                            "condition": condition,
                            "pair_id": pair_id,
                            "type": "presence" if suite == "grounding" else "yes/no",
                            "score": float(condition == "correct"),
                            "exact": float(condition == "correct"),
                            "nll": 1.0 if condition == "correct" else 2.0,
                            "empty": 0.0,
                            "seconds": 0.1,
                            "normalized": str(i),
                            "follows_partner": float(condition == "partner"),
                        }
                    )
    summarize(rows, tmp_path)
    import json

    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["grounding_correct_advantages"]["accuracy"]["ci95"] == [1.0, 1.0]
    assert result["natural_correct_advantages"]["nll"]["ci95"] == [1.0, 1.0]
