import math
from pathlib import Path
import torch
from torch import nn
from common.adapters import ATTENTION, inject
from common import QWEN, token_losses, model_revision
from common.device import current_device, model_backend_kwargs


class VLM(nn.Module):
    def __init__(self, compressed=False, revision=None, device=None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = current_device(device)

        self.revision = revision or model_revision()
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN, revision=self.revision)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            QWEN,
            revision=self.revision,
            torch_dtype=torch.float16,
            **model_backend_kwargs(device),
        ).requires_grad_(False)
        if self.backbone.config.hidden_size != 1024:
            raise ValueError("The prescribed Qwen connector requires hidden size 1024")
        self.connector = nn.Sequential(
            nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, 1024)
        )
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        self.compressed = compressed
        self.has_adapters = False

    def add_adapters(self):
        if not self.has_adapters:
            inject(self.backbone, targets=ATTENTION)
            self.has_adapters = True

    def feature(self, image_id, root):
        x = torch.load(Path(root) / f"{image_id}.pt", weights_only=True).to(
            self.scale.device
        )
        if tuple(x.shape) != (196, 768):
            raise ValueError("Invalid cached patch features")
        if self.compressed:
            x = (
                x.view(14, 14, 768)
                .reshape(7, 2, 7, 2, 768)
                .float()
                .mean((1, 3))
                .reshape(49, 768)
                .half()
            )
        return x

    def prompt(self, question):
        tok = self.tokenizer
        # Content caps apply to content only, leaving the chat syntax intact.
        question = tok.decode(
            tok.encode(question, add_special_tokens=False)[:80],
            skip_special_tokens=False,
        )
        marker = "PA3_VISUAL_INSERTION_POINT"
        text = tok.apply_chat_template(
            [{"role": "user", "content": marker + question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        left, right = text.split(marker)
        return tok.encode(left, add_special_tokens=False), tok.encode(
            right, add_special_tokens=False
        )

    def batch(self, rows, features, condition="correct", generation=False):
        sequences, label_rows, nll_rows = [], [], []
        embed = self.backbone.get_input_embeddings()
        tok, device = self.tokenizer, self.scale.device
        for r in rows:
            before, after = self.prompt(r["question"])
            left = embed(torch.tensor(before, device=device))
            right = embed(torch.tensor(after, device=device))
            if condition != "none":
                image = (
                    r["image_id"]
                    if condition == "correct"
                    else r[condition + "_image_id"]
                )
                visual = (
                    self.connector(self.feature(image, features).float()) * self.scale
                )
                prefix = torch.cat((left, visual.to(left.dtype), right))
            else:
                prefix = torch.cat((left, right))
            cap = 64 if r["kind"] == "caption" else 16
            target = tok.encode(r["answer"], add_special_tokens=False)[:cap]
            end = tok.convert_tokens_to_ids("<|im_end|>")
            if generation:
                sequences.append(prefix)
                label_rows.append([-100] * len(prefix))
                nll_rows.append([-100] * len(prefix))
            else:
                answer = target + [end]
                sequences.append(
                    torch.cat((prefix, embed(torch.tensor(answer, device=device))))
                )
                label_rows.append([-100] * len(prefix) + answer)
                nll_rows.append([-100] * len(prefix) + target + [-100])
        width = max(len(s) for s in sequences)
        # Left padding also supports batched autoregressive generation.
        embeddings = sequences[0].new_zeros((len(rows), width, 1024))
        mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
        labels = torch.full_like(mask, -100)
        nll = torch.full_like(mask, -100)
        for i, (s, lab, nl) in enumerate(zip(sequences, label_rows, nll_rows)):
            embeddings[i, -len(s) :] = s
            mask[i, -len(s) :] = 1
            labels[i, -len(s) :] = torch.tensor(lab, device=device)
            nll[i, -len(s) :] = torch.tensor(nl, device=device)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        return (
            {
                "inputs_embeds": embeddings,
                "attention_mask": mask,
                "position_ids": positions,
            },
            labels,
            nll,
        )

    def loss(self, rows, features, condition="correct", answer_nll=False):
        inputs, labels, nll = self.batch(rows, features, condition)
        output = self.backbone(**inputs).logits
        losses, valid = token_losses(output, nll if answer_nll else labels)
        if not valid.sum(1).gt(0).all():
            raise ValueError("Empty supervised target")
        return losses.sum(1) / valid.sum(1)

    @torch.no_grad()
    def generate(self, rows, features, condition):
        inputs, _, _ = self.batch(rows, features, condition, generation=True)
        # GenerationMixin extends the attention mask and recomputes positions each
        # step. Supplying static position_ids here would freeze later positions.
        inputs.pop("position_ids")
        tokens = self.backbone.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=16,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True), tokens

    @torch.no_grad()
    def calibrate(self, rows, features):
        # Training-only RMS matching, one fixed scale before optimization.
        visual_sq, text_sq, nv, nt = 0.0, 0.0, 0, 0
        embed = self.backbone.get_input_embeddings()
        for r in rows[:128]:
            visual = self.connector(
                self.feature(r["image_id"], features).float()
            ).float()
            ids = self.tokenizer.encode(r["answer"], add_special_tokens=False)[:64]
            text = embed(torch.tensor(ids, device=self.scale.device)).float()
            visual_sq += visual.square().sum().item()
            text_sq += text.square().sum().item()
            nv += visual.numel()
            nt += text.numel()
        self.scale.fill_(math.sqrt((text_sq / nt) / (visual_sq / nv)))
        return {
            "rule": "RMS of first 128 training-caption text embeddings / connector outputs",
            "scale": self.scale.item(),
        }
