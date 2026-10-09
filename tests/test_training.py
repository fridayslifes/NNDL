"""Tests for the MLP, the training loop and the imbalance strategies."""

import numpy as np
import pytest
import torch
from torch import nn

from src import config
from src.model import CalibratedChurnModel, ChurnMLP, count_parameters, predict_logits, predict_proba
from src.training import EarlyStopping, make_loader, prepare_imbalance_strategy, train_mlp


def _data(n=300, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.6).astype(int)
    return X, y


def test_architecture_matches_brief():
    model = ChurnMLP(40)
    assert count_parameters(model) == 4929
    kinds = [type(m).__name__ for m in model.backbone]
    assert kinds == ["Linear", "BatchNorm1d", "ReLU", "Dropout"] * 2


def test_loader_drops_single_sample_last_batch():
    X, y = _data(n=129)
    loader = make_loader(X, y, batch_size=64)
    assert all(xb.shape[0] == 64 for xb, _ in loader)


def test_early_stopping_restores_best_weights():
    model = nn.Linear(2, 1)
    stopper = EarlyStopping(patience=2)
    stopper.step(1.0, model, epoch=1)
    best = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(5.0)
    assert not stopper.step(1.5, model, epoch=2)
    assert stopper.step(1.6, model, epoch=3)
    stopper.restore_best(model)
    assert torch.equal(model.weight, best["weight"])


def test_training_reduces_loss_and_is_reproducible():
    X, y = _data()
    runs = []
    for _ in range(2):
        config.set_seed(7)
        model = ChurnMLP(X.shape[1], hidden_dims=(16, 8))
        model, hist = train_mlp(model, X, y, X, y, nn.BCELoss(), max_epochs=15, patience=None,
                                seed=7, device="cpu", verbose=False)
        runs.append(predict_proba(model, X))
    assert hist.train_loss[-1] < hist.train_loss[0]
    np.testing.assert_allclose(runs[0], runs[1])


def test_weighted_bce_pos_weight_and_oversampling_balance():
    X, y = _data()
    weighted = prepare_imbalance_strategy("weighted_bce", X, y)
    assert weighted.details["pos_weight"] == pytest.approx((y == 0).sum() / (y == 1).sum())
    over = prepare_imbalance_strategy("oversampling", X, y)
    assert (over.y_fit == 1).sum() == (over.y_fit == 0).sum()
    with pytest.raises(ValueError):
        prepare_imbalance_strategy("plain_bce", X, np.zeros_like(y))


def test_calibrated_wrapper_matches_manual_platt():
    X, _ = _data()
    model = ChurnMLP(X.shape[1]).eval()
    wrapped = CalibratedChurnModel(model, a=0.7, b=-0.4).eval()
    z = predict_logits(model, X)
    with torch.no_grad():
        out = wrapped(torch.from_numpy(X)).numpy().ravel()
    np.testing.assert_allclose(out, 1 / (1 + np.exp(-(0.7 * z - 0.4))), rtol=1e-5)
