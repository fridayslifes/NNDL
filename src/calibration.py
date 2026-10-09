r"""
Probability calibration with Platt scaling.

Why calibrate?
--------------
Class-imbalance tricks change *how the probabilities are scaled*:
``pos_weight`` ≈ 2.8, oversampling to a 50/50 mix, and α = 0.75 in focal loss
all push predicted probabilities **upwards** (the model "believes" churn is more
common than the true 26.5 %). The ranking of customers can still be excellent
(high PR-AUC), but a raw score of 0.70 no longer means "70 % chance of churn".

The dashboard needs probabilities that mean what they say because

* the risk tiers are defined in probability terms (High > 70 %), and
* the cost slider computes **expected** losses as :math:`\sum_i p_i \cdot C_{lost}`.

Platt scaling
-------------
Fit a 1-D logistic regression on the network's validation-set logits z:

.. math::

    p_{cal} = \sigma(a \cdot z + b)

Only two parameters are learned (a, b), so over-fitting the validation set is
negligible. With a > 0 the transformation is strictly increasing, therefore it
**never changes the ranking**: PR-AUC, ROC-AUC and the set of achievable
confusion matrices are identical before and after calibration — only the
probability scale (and so the numerical value of the optimal threshold) moves.

Reference: Platt (1999), *Probabilistic Outputs for Support Vector Machines*.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


@dataclass
class PlattCalibrator:
    """
    Two-parameter logistic calibration :math:`p = \\sigma(a z + b)`.

    ``a = 1, b = 0`` is the identity (plain sigmoid of the logit).
    """

    a: float = 1.0
    b: float = 0.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        """
        Estimate (a, b) by maximum likelihood on held-out (validation) data.

        A very large ``C`` (inverse regularisation strength) makes the
        scikit-learn logistic regression effectively unregularised.

        Raises
        ------
        ValueError
            If inputs are empty, misaligned, or contain a single class.
        """
        z = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y).astype(int).reshape(-1)
        if z.shape[0] != y.shape[0] or z.shape[0] == 0:
            raise ValueError("logits and y must be non-empty and the same length")
        if np.unique(y).size < 2:
            raise ValueError("Calibration needs both classes in y")
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(z, y)
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        if a <= 0:
            warnings.warn("Platt slope a <= 0 (model ranks worse than random); keeping identity calibration.")
            a, b = 1.0, 0.0
        self.a, self.b = a, b
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Calibrated probabilities σ(a·z + b)."""
        s = self.a * np.asarray(logits, dtype=np.float64) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))

    def to_dict(self) -> dict[str, Any]:
        return {"method": "platt", "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlattCalibrator":
        """Rebuild from ``model_config.json``; missing/empty → identity calibration."""
        if not data:
            return cls()
        return cls(a=float(data.get("a", 1.0)), b=float(data.get("b", 0.0)))


def reliability_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """
    Data for a reliability diagram plus the Brier score.

    * Reliability diagram: split customers into ``n_bins`` equal-count bins of
      predicted probability and compare the mean prediction with the observed
      churn rate in each bin. A perfectly calibrated model lies on y = x.
    * Brier score: :math:`\\frac{1}{n}\\sum (p_i - y_i)^2` — lower is better.

    Returns
    -------
    dict with ``prob_pred``, ``prob_true`` (lists) and ``brier`` (float).
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return {
        "prob_pred": prob_pred.tolist(),
        "prob_true": prob_true.tolist(),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "mean_predicted": float(y_prob.mean()),
        "observed_rate": float(y_true.mean()),
    }
