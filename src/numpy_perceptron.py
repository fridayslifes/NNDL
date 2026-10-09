r"""
Single-Layer Perceptron implemented from scratch in pure NumPy (mandatory component).

No scikit-learn estimator, no PyTorch module, no autograd: the forward pass, the
loss, the gradients and the parameter updates are all written by hand below.

Model
-----
For a mini-batch :math:`X \in \mathbb{R}^{m \times d}` (m customers, d features):

.. math::

    z = XW + b, \qquad W \in \mathbb{R}^{d \times 1},\; b \in \mathbb{R}

    \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)

:math:`\hat{y}` is the predicted probability that the customer churns.

Loss — Binary Cross-Entropy (BCE)
---------------------------------
.. math::

    L = -\frac{1}{m}\sum_{i=1}^{m}\big[y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)\big]

BCE is the negative log-likelihood of a Bernoulli model, so minimising it is
maximum-likelihood estimation of W and b.

Gradient derivation (chain rule / back-propagation for one layer)
-----------------------------------------------------------------
1. :math:`\dfrac{\partial L}{\partial \hat{y}_i} = \dfrac{1}{m}\cdot\dfrac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)}`
2. Sigmoid derivative: :math:`\dfrac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i(1-\hat{y}_i)`
3. Multiply (the :math:`\hat{y}(1-\hat{y})` terms cancel — this is why sigmoid + BCE
   pair so nicely): :math:`\dfrac{\partial L}{\partial z_i} = \dfrac{1}{m}(\hat{y}_i - y_i)`
4. Since :math:`z = XW + b`: :math:`\dfrac{\partial z_i}{\partial W} = x_i^\top` and
   :math:`\dfrac{\partial z_i}{\partial b} = 1`, therefore

.. math::

    \frac{\partial L}{\partial W} = \frac{1}{m} X^\top(\hat{y} - y), \qquad
    \frac{\partial L}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)

Update rule (mini-batch gradient descent with learning rate η):

.. math::

    W \leftarrow W - \eta\,\frac{\partial L}{\partial W}, \qquad
    b \leftarrow b - \eta\,\frac{\partial L}{\partial b}

Note for the viva
-----------------
Rosenblatt's original perceptron used a hard step activation and the
"perceptron learning rule". Using a *sigmoid* output with BCE makes the
single-layer perceptron differentiable and gives calibrated probabilities —
mathematically it is the same model as logistic regression, but trained with
our own mini-batch gradient descent instead of scikit-learn's L-BFGS solver.
That is why its metrics should land very close to the sklearn baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PerceptronHistory:
    """Per-epoch loss curves recorded during :meth:`NumpyPerceptron.fit`."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


class NumpyPerceptron:
    """
    Sigmoid single-layer perceptron trained with mini-batch gradient descent.

    Parameters
    ----------
    n_features : int
        Input dimensionality d (number of encoded features).
    learning_rate : float, default 0.1
        Step size η of gradient descent. Must be > 0.
    n_epochs : int, default 200
        Number of full passes over the training data.
    batch_size : int, default 64
        Mini-batch size m. Smaller → noisier but more frequent updates.
    l2 : float, default 0.0
        Optional L2 penalty λ; adds :math:`\\frac{\\lambda}{2}\\lVert W\\rVert^2` to the loss
        and :math:`\\lambda W` to :math:`\\partial L/\\partial W` (bias is not penalised).
    init_std : float, default 0.01
        Standard deviation of the random normal initialisation of W and b.
        Small values keep the initial z near 0 where the sigmoid is steepest.
    seed : int, default 42
        Seed for the private random generator (initialisation + shuffling).
    verbose : bool, default False
        Print the loss every ``print_every`` epochs.
    print_every : int, default 25

    Attributes
    ----------
    W : np.ndarray, shape (d, 1)
    b : np.ndarray, shape (1,)
    history : PerceptronHistory
    """

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.1,
        n_epochs: int = 200,
        batch_size: int = 64,
        l2: float = 0.0,
        init_std: float = 0.01,
        seed: int = 42,
        verbose: bool = False,
        print_every: int = 25,
    ) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be a positive integer")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if n_epochs <= 0 or batch_size <= 0:
            raise ValueError("n_epochs and batch_size must be positive integers")
        if l2 < 0:
            raise ValueError("l2 must be >= 0")

        self.n_features = int(n_features)
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.l2 = float(l2)
        self.verbose = verbose
        self.print_every = max(1, int(print_every))
        self._rng = np.random.default_rng(seed)

        # Random initialisation of the weight matrix W and the bias b.
        self.W: np.ndarray = self._rng.normal(0.0, init_std, size=(self.n_features, 1))
        self.b: np.ndarray = self._rng.normal(0.0, init_std, size=(1,))
        self.history = PerceptronHistory()

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable logistic function σ(z) = 1 / (1 + e^(−z)).

        For very negative z, e^(−z) overflows float64. We therefore use the
        algebraically equivalent form e^z / (1 + e^z) whenever z < 0, so the
        exponent is always ≤ 0 and never overflows.
        """
        z = np.asarray(z, dtype=np.float64)
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    @staticmethod
    def bce_loss(y_hat: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
        """
        Mean binary cross-entropy.

        ŷ is clipped to [ε, 1−ε] so log(0) = −∞ can never occur when the
        model becomes (over-)confident.
        """
        y_hat = np.clip(y_hat.reshape(-1), eps, 1.0 - eps)
        y = y.reshape(-1).astype(np.float64)
        return float(-np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)))

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass: z = XW + b, ŷ = σ(z).

        Returns
        -------
        (z, y_hat) : both shape (m, 1)
        """
        z = X @ self.W + self.b          # (m, d) @ (d, 1) + (1,) -> (m, 1) via broadcasting
        return z, self.sigmoid(z)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Total objective = BCE + (λ/2)·‖W‖² (the L2 term is 0 when ``l2 == 0``)."""
        _, y_hat = self.forward(X)
        return self.bce_loss(y_hat, y) + 0.5 * self.l2 * float(np.sum(self.W ** 2))

    def compute_gradients(
        self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Analytic gradients derived in the module docstring.

        dL/dW = (1/m) Xᵀ(ŷ − y) + λW,     dL/db = (1/m) Σ(ŷ − y)

        Returns
        -------
        (dW, db) with shapes (d, 1) and (1,)
        """
        m = X.shape[0]
        error = y_hat - y.reshape(-1, 1)            # (m, 1): the "delta" at the output
        dW = (X.T @ error) / m + self.l2 * self.W   # (d, m) @ (m, 1) -> (d, 1)
        db = np.array([error.sum() / m])            # scalar wrapped as shape (1,)
        return dW, db

    # ------------------------------------------------------------------
    # Training / prediction
    # ------------------------------------------------------------------
    def _validate(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (n, {self.n_features}); got {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        if y is None:
            return X, None
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("y must contain only 0/1 labels")
        return X, y

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "NumpyPerceptron":
        """
        Train with mini-batch gradient descent.

        Each epoch: shuffle the rows (so batches differ between epochs and the
        gradient estimates are unbiased), then for every mini-batch run the
        forward pass, compute the analytic gradients and take one step
        ``θ ← θ − η ∇θ L``. The full-data training loss (and validation loss if
        given) is recorded once per epoch for plotting.
        """
        X, y = self._validate(X, y)
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val, y_val = self._validate(X_val, y_val)

        n = X.shape[0]
        for epoch in range(1, self.n_epochs + 1):
            order = self._rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = order[start:start + self.batch_size]
                X_b, y_b = X[batch], y[batch]
                _, y_hat = self.forward(X_b)
                dW, db = self.compute_gradients(X_b, y_b, y_hat)
                self.W -= self.learning_rate * dW     # gradient-descent step
                self.b -= self.learning_rate * db

            self.history.train_loss.append(self.loss(X, y))
            if has_val:
                self.history.val_loss.append(self.loss(X_val, y_val))
            if self.verbose and (epoch % self.print_every == 0 or epoch == 1):
                msg = f"epoch {epoch:4d} | train BCE {self.history.train_loss[-1]:.4f}"
                if has_val:
                    msg += f" | val BCE {self.history.val_loss[-1]:.4f}"
                print(msg)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Churn probabilities ŷ, shape (n,)."""
        X, _ = self._validate(X)
        return self.forward(X)[1].reshape(-1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Hard 0/1 predictions: churn if ŷ ≥ threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save W, b, hyper-parameters and loss history to a ``.npz`` file."""
        np.savez(
            path, W=self.W, b=self.b,
            hyper=np.array([self.learning_rate, self.n_epochs, self.batch_size, self.l2]),
            train_loss=np.array(self.history.train_loss), val_loss=np.array(self.history.val_loss),
        )

    @classmethod
    def load(cls, path: str) -> "NumpyPerceptron":
        """Rebuild a trained perceptron from :meth:`save` output."""
        data = np.load(path)
        lr, epochs, batch, l2 = data["hyper"]
        model = cls(n_features=data["W"].shape[0], learning_rate=float(lr), n_epochs=int(epochs),
                    batch_size=int(batch), l2=float(l2))
        model.W, model.b = data["W"].copy(), data["b"].copy()
        model.history = PerceptronHistory(list(data["train_loss"]), list(data["val_loss"]))
        return model

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def gradient_check(self, X: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
        """
        Compare analytic gradients with central finite differences.

        For each parameter θ_k:

            numerical_k = [L(θ_k + ε) − L(θ_k − ε)] / (2ε)

        Returns the maximum relative error
        ‖analytic − numerical‖ / (‖analytic‖ + ‖numerical‖). Values below ~1e-7
        prove the hand-derived back-propagation formulas are correct.
        """
        X, y = self._validate(X, y)
        _, y_hat = self.forward(X)
        dW, db = self.compute_gradients(X, y, y_hat)
        analytic = np.concatenate([dW.ravel(), db.ravel()])

        numerical = np.zeros_like(analytic)
        params = [(self.W, i) for i in range(self.W.size)] + [(self.b, 0)]
        for k, (tensor, idx) in enumerate(params):
            flat = tensor.reshape(-1)          # a view: edits change the real parameter
            original = flat[idx]
            flat[idx] = original + eps
            loss_plus = self.loss(X, y)
            flat[idx] = original - eps
            loss_minus = self.loss(X, y)
            flat[idx] = original               # restore
            numerical[k] = (loss_plus - loss_minus) / (2 * eps)

        denom = np.linalg.norm(analytic) + np.linalg.norm(numerical)
        return float(np.linalg.norm(analytic - numerical) / denom) if denom > 0 else 0.0
