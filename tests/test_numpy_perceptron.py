"""Tests for the from-scratch NumPy perceptron."""

import numpy as np
import pytest

from src.numpy_perceptron import NumpyPerceptron


def _toy_data(n=400, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = (X @ w + 0.2 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_gradient_check_matches_finite_differences():
    X, y = _toy_data()
    model = NumpyPerceptron(n_features=X.shape[1], seed=1)
    assert model.gradient_check(X[:60], y[:60]) < 1e-7


def test_gradient_check_with_l2_penalty():
    X, y = _toy_data()
    model = NumpyPerceptron(n_features=X.shape[1], l2=0.1, init_std=0.5, seed=2)
    assert model.gradient_check(X[:60], y[:60]) < 1e-7


def test_sigmoid_is_stable_for_extreme_inputs():
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    with np.errstate(over="raise"):
        out = NumpyPerceptron.sigmoid(z)
    assert np.all((out >= 0) & (out <= 1))
    assert out[2] == pytest.approx(0.5)


def test_bce_never_returns_infinity():
    loss = NumpyPerceptron.bce_loss(np.array([0.0, 1.0]), np.array([1, 0]))
    assert np.isfinite(loss)


def test_learns_a_linearly_separable_problem():
    X, y = _toy_data(n=800)
    model = NumpyPerceptron(n_features=X.shape[1], learning_rate=0.5, n_epochs=100).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95
    assert model.history.train_loss[-1] < model.history.train_loss[0]


def test_input_validation():
    model = NumpyPerceptron(n_features=3)
    with pytest.raises(ValueError):
        model.fit(np.zeros((5, 4)), np.zeros(5))
    with pytest.raises(ValueError):
        model.fit(np.zeros((5, 3)), np.array([0, 1, 2, 0, 1]))
    with pytest.raises(ValueError):
        NumpyPerceptron(n_features=3, learning_rate=0)


def test_save_and_load_roundtrip(tmp_path):
    X, y = _toy_data()
    model = NumpyPerceptron(n_features=X.shape[1], n_epochs=5).fit(X, y)
    path = tmp_path / "p.npz"
    model.save(path)
    restored = NumpyPerceptron.load(path)
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
