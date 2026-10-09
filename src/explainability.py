r"""
SHAP explanations for the churn MLP (Part 8).

SHAP in one paragraph
---------------------
SHAP (SHapley Additive exPlanations) assigns every feature j a contribution
:math:`\phi_j` to one customer's prediction such that

.. math::

    f(x) = \underbrace{E[f(X)]}_{\text{base value}} + \sum_{j=1}^{d} \phi_j

i.e. the contributions *exactly add up* from the average prediction (over a
background sample of training customers) to this customer's prediction.
:math:`\phi_j > 0` pushes churn risk **up**, :math:`\phi_j < 0` pushes it **down**.

``shap.DeepExplainer`` computes these efficiently for neural networks with
DeepLIFT-style back-propagation of contribution scores through every layer
(Linear, BatchNorm, ReLU, Sigmoid), using ~200 training rows as background.

Aggregating one-hot features
----------------------------
The model sees ``Contract_Month-to-month``, ``Contract_One year`` and
``Contract_Two year`` as three inputs. Because SHAP values are additive, the
total effect of the business concept *Contract* is simply their sum. The UI and
the ``get_customer_top_drivers`` helper report these aggregated, human-readable
drivers (e.g. "Contract = Month-to-month: +9.8 pp").
"""

from __future__ import annotations

import threading
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


def _to_2d(values: Any, n_rows: int, n_features: int) -> np.ndarray:
    """
    Normalise SHAP's output to shape ``(n_rows, n_features)``.

    Depending on the SHAP version, a single-output model returns a list with one
    array, an array of shape (n, d, 1), or an array of shape (n, d).
    """
    if isinstance(values, list):
        values = values[0]
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.reshape(n_rows, n_features)


class ChurnExplainer:
    """
    Thin, thread-safe wrapper around ``shap.DeepExplainer`` for a churn model.

    Parameters
    ----------
    model : nn.Module
        Model whose output is the churn probability, shape (N, 1)
        (``ChurnMLP`` or ``CalibratedChurnModel``). Put into ``eval()`` mode.
    background : np.ndarray, shape (B, d)
        Encoded *training* rows used to define the base value E[f(X)].
    feature_names : sequence of str
        Encoded feature names (length d).
    feature_origins : sequence of str
        Raw column each encoded feature came from (length d).
    device : str or torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        background: np.ndarray,
        feature_names: Sequence[str],
        feature_origins: Sequence[str],
        device: torch.device | str = "cpu",
    ) -> None:
        import shap  # imported lazily: heavy, and only needed when explaining

        background = np.asarray(background, dtype=np.float32)
        if background.ndim != 2 or background.shape[0] < 2:
            raise ValueError("background must be a 2-D array with at least 2 rows")
        if background.shape[1] != len(feature_names) or len(feature_names) != len(feature_origins):
            raise ValueError("background, feature_names and feature_origins must agree on d")

        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.feature_names = list(feature_names)
        self.feature_origins = list(feature_origins)
        self._lock = threading.Lock()   # SHAP registers hooks on the model: serialise calls
        bg = torch.from_numpy(background).to(self.device)

        try:
            self._explainer = shap.DeepExplainer(self.model, bg)
            self.method = "DeepExplainer"
        except Exception as exc:  # pragma: no cover - depends on SHAP/torch versions
            warnings.warn(f"DeepExplainer unavailable ({exc}); falling back to GradientExplainer.")
            self._explainer = shap.GradientExplainer(self.model, bg)
            self.method = "GradientExplainer"

        with torch.no_grad():
            self.base_value = float(self.model(bg).mean().item())

    # ------------------------------------------------------------------
    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        SHAP matrix for encoded rows ``X``: shape ``(n, d)``, in probability units.

        Row i satisfies ``base_value + shap[i].sum() ≈ model(X[i])``.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        if X.shape[0] == 0:
            return np.zeros((0, X.shape[1]))
        xt = torch.from_numpy(X).to(self.device)
        with self._lock:
            if self.method == "DeepExplainer":
                try:
                    raw = self._explainer.shap_values(xt)
                except AssertionError:  # additivity check failed (numerical edge case)
                    raw = self._explainer.shap_values(xt, check_additivity=False)
            else:
                raw = self._explainer.shap_values(xt)
        return _to_2d(raw, X.shape[0], X.shape[1])

    def aggregate(self, shap_matrix: np.ndarray) -> pd.DataFrame:
        """
        Sum encoded-feature SHAP values into their raw columns.

        Returns
        -------
        pd.DataFrame, shape (n, n_raw_features)
            Columns in first-appearance order of ``feature_origins``.
        """
        frame = pd.DataFrame(np.atleast_2d(shap_matrix), columns=self.feature_names)
        origins = pd.Series(self.feature_origins, index=self.feature_names)
        ordered = list(dict.fromkeys(self.feature_origins))
        return frame.T.groupby(origins).sum().T[ordered]

    def global_importance(self, shap_matrix: np.ndarray) -> pd.Series:
        """Mean |aggregated SHAP| per raw feature, sorted descending (global ranking)."""
        return self.aggregate(shap_matrix).abs().mean().sort_values(ascending=False)

    def get_customer_top_drivers(
        self,
        customer_features: np.ndarray,
        raw_record: Mapping[str, Any] | None = None,
        top_k: int = 5,
        aggregate: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Top ``top_k`` features pushing ONE customer's churn probability up or down.

        Parameters
        ----------
        customer_features : np.ndarray, shape (d,) or (1, d)
            The customer's *encoded* feature vector.
        raw_record : mapping, optional
            The customer's cleaned raw values (e.g. ``{"Contract": "Month-to-month",
            "tenure": 2, ...}``) used to display human-readable values.
        top_k : int
            Number of drivers returned (sorted by |contribution|).
        aggregate : bool
            ``True`` → one driver per raw column (one-hot groups summed);
            ``False`` → individual encoded columns.

        Returns
        -------
        list of dict
            ``feature``, ``value`` (display string), ``shap_value`` (probability
            units), ``impact_pp`` (percentage points), ``direction``
            ("increases" / "decreases" churn risk).
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        x = np.asarray(customer_features, dtype=np.float32).reshape(1, -1)
        phi = self.shap_values(x)[0]

        if aggregate:
            contributions = self.aggregate(phi.reshape(1, -1)).iloc[0]
        else:
            contributions = pd.Series(phi, index=self.feature_names)

        order = contributions.abs().sort_values(ascending=False).index[:top_k]
        drivers = []
        for name in order:
            value = float(contributions[name])
            if raw_record is not None and name in raw_record:
                display = _format_raw_value(name, raw_record[name])
            else:
                display = f"{float(x[0, self.feature_names.index(name)]):.3f}" if name in self.feature_names else "—"
            drivers.append({
                "feature": name,
                "value": display,
                "shap_value": round(value, 6),
                "impact_pp": round(100.0 * value, 2),
                "direction": "increases" if value > 0 else "decreases",
            })
        return drivers


def _format_raw_value(feature: str, value: Any) -> str:
    """Human-readable raw value (tenure in months, charges with 2 decimals)."""
    if feature == "tenure":
        try:
            months = int(round(float(value)))
            return f"{months} month" + ("" if months == 1 else "s")
        except (TypeError, ValueError):
            return str(value)
    if feature in {"MonthlyCharges", "TotalCharges"}:
        try:
            return f"{float(value):,.2f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def get_customer_top_drivers(
    customer_features: np.ndarray,
    explainer: ChurnExplainer,
    raw_record: Mapping[str, Any] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Functional form required by the brief: top-5 drivers for one customer.

    See :meth:`ChurnExplainer.get_customer_top_drivers`.
    """
    return explainer.get_customer_top_drivers(customer_features, raw_record=raw_record, top_k=top_k)


def select_background(X_train: np.ndarray, n: int = 200, seed: int = 42) -> np.ndarray:
    """
    Random sample of ``n`` training rows used as the SHAP background.

    Sampling (rather than the first n rows) avoids any ordering artefacts;
    200 rows balance explanation accuracy against speed.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    if X_train.shape[0] <= n:
        return X_train.copy()
    idx = np.random.default_rng(seed).choice(X_train.shape[0], size=n, replace=False)
    return X_train[np.sort(idx)]
