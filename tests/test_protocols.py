import numpy as np
import pytest
import torch
from torch import nn
from common.adapters import Adapter
from common import bootstrap, lr_factor, token_losses
from common import jsonl, load_jsonl, rewind_log
from task3_inference.live import boxed
from task3_inference.scoring import NO_ANSWER, rank_weights, scores, select, stopping


def candidate(answer, length=2, kl=(1.0, 2.0), logp=(-2.0, -1.0)):
    return {"answer": answer, "length": length, "kl_u_p": list(kl), "logp": list(logp)}


def test_projected_adapter_constraints_and_gradients():
    torch.manual_seed(6304)
    base = nn.Linear(12, 10, bias=False)
    adapter = Adapter(base, rank=3, alpha=6, protection=4)
    with torch.no_grad():
        adapter.B.normal_()
    delta = adapter.delta()
    assert (adapter.U.T @ delta).norm() < 1e-5
    assert (delta @ adapter.V).norm() < 1e-5
    x = torch.randn(2, 12)
    adapter.eval()
    torch.testing.assert_close(adapter(x), nn.functional.linear(x, base.weight + delta))
    adapter(x).square().mean().backward()
    assert base.weight.grad is None
    assert adapter.A.grad is not None and adapter.B.grad is not None


def test_matched_lora_initialization():
    base = nn.Linear(12, 10)
    torch.manual_seed(6304)
    a = Adapter(base, 3, 6)
    torch.manual_seed(6304)
    b = Adapter(base, 3, 6, protection=4)
    torch.testing.assert_close(a.A, b.A)
    torch.testing.assert_close(a.B, b.B)


def test_response_mask_and_two_loss_weightings():
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, -100, -100, -100, 4]])
    losses, valid = token_losses(logits, labels)
    assert valid.sum(1).tolist() == [3, 1]
    expected = nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1), reduction="sum"
    )
    torch.testing.assert_close(losses.sum(), expected)
    assert not torch.isclose(
        losses.sum() / valid.sum(), (losses.sum(1) / valid.sum(1)).mean()
    )


def test_prefix_calibration_and_fusion_identity():
    a = candidate("A", 2, (1, 5))
    a.update(
        image_attention=[1, 9], prompt_attention=[9, 1], generated_attention=[0, 0]
    )
    b = candidate("B", 2, (100, 200))
    b.update(
        image_attention=[1, 1], prompt_attention=[1, 1], generated_attention=[0, 0]
    )
    one = scores([a])
    both = scores([a, b])
    assert one["visconf"][0] != both["visconf"][0]
    lo, hi = np.quantile([1, 5], [0.05, 0.95])
    c = np.clip((np.array([1, 5]) - lo) / (hi - lo), 0, 1)
    v = np.array([0.1, 0.9])
    assert one["tokenwise"][0] - one["visconf"][0] == pytest.approx(
        np.mean((c - c.mean()) * (v - v.mean()))
    )


def test_ties_missing_answers_and_cost_independent_selector():
    np.testing.assert_allclose(rank_weights([4, 4, 1]), [0.75, 0.75, 0])
    pool = [candidate("B"), candidate("A")]
    assert select(pool, "consensus", "mathvista") == 0
    assert select(pool, "certainty_rwc", "mathvista") == 0
    assert select([candidate(NO_ANSWER)], "consensus", "mathvista") == 0
    missing_pool = [candidate(NO_ANSWER) for _ in range(32)]
    assert stopping(missing_pool, "mathvista", 0.5)[0] == 32


def test_stopping_never_sees_future_candidates():
    prefix = [candidate("A") for _ in range(4)]
    future_a = prefix + [candidate("B", kl=(9999, 9999)) for _ in range(28)]
    future_b = prefix + [candidate("C", kl=(0.001, 0.001)) for _ in range(28)]
    for r in future_a + future_b:
        r.update(
            image_attention=[1, 2], prompt_attention=[2, 1], generated_attention=[1, 1]
        )
    assert (
        stopping(future_a, "mathvista", 1, 0)
        == stopping(future_b, "mathvista", 1, 0)
        == (4, 0)
    )


def test_bootstrap_unit_and_schedule():
    assert bootstrap([1, 1, 1], 100)["ci95"] == [1.0, 1.0]
    assert lr_factor(37, 1250, 38) == 1
    assert lr_factor(1249, 1250, 38) == pytest.approx(0)
    assert lr_factor(1499, 1500, 75, 0.1) == pytest.approx(0.1)


def test_balanced_last_box():
    assert boxed(r"old \boxed{2}, new \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert boxed(r"\boxed{2} then \boxed{unfinished") == NO_ANSWER
    assert boxed("No extracted answer") == NO_ANSWER


def test_resume_log_rewinds_to_checkpoint(tmp_path):
    path = tmp_path / "train.jsonl"
    jsonl(path, [{"update": i, "loss": 1.0} for i in range(1, 6)])
    rewind_log(path, 3)
    jsonl(path, [{"update": 4, "loss": 2.0}], append=True)
    assert [r["update"] for r in load_jsonl(path)] == [1, 2, 3, 4]
