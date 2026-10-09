"""Tests for the from-scratch focal loss."""

import pytest
import torch
import torch.nn.functional as F

from src.focal_loss import FocalLoss


def test_reduces_to_half_bce_when_gamma_zero_alpha_half():
    torch.manual_seed(0)
    logits, targets = torch.randn(500), (torch.rand(500) > 0.7).float()
    fl = FocalLoss(alpha=0.5, gamma=0.0)(logits, targets)
    assert torch.allclose(fl, 0.5 * F.binary_cross_entropy_with_logits(logits, targets), atol=1e-6)


def test_logit_and_probability_inputs_agree():
    torch.manual_seed(1)
    logits, targets = torch.randn(200), (torch.rand(200) > 0.5).float()
    from_logits = FocalLoss(from_logits=True)(logits, targets)
    from_probs = FocalLoss(from_logits=False)(torch.sigmoid(logits), targets)
    assert torch.allclose(from_logits, from_probs, atol=1e-5)


def test_easy_examples_are_down_weighted():
    logit_easy = torch.tensor([3.0])          # p ≈ 0.95 for a true churner
    target = torch.tensor([1.0])
    bce = F.binary_cross_entropy_with_logits(logit_easy, target)
    focal = FocalLoss(alpha=1.0, gamma=2.0)(logit_easy, target)
    assert focal < bce / 100


def test_extreme_logits_stay_finite_and_have_gradients():
    logits = torch.tensor([-200.0, 200.0, 0.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = FocalLoss()(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_reduction_modes_and_shapes():
    logits, targets = torch.randn(8, 1), torch.ones(8, 1)
    assert FocalLoss(reduction="none")(logits, targets).shape == (8, 1)
    total = FocalLoss(reduction="sum")(logits, targets)
    mean = FocalLoss(reduction="mean")(logits, targets)
    assert torch.allclose(total / 8, mean)


@pytest.mark.parametrize("kwargs", [{"alpha": 1.5}, {"gamma": -1.0}, {"reduction": "max"}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        FocalLoss(**kwargs)
