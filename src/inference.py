"""
``ChurnPredictor`` — load every saved artefact once and score new customers.

This is the only object the FastAPI backend talks to. It guarantees that an
uploaded CSV goes through *exactly* the training-time pipeline:

    raw CSV ─► clean_dataframe ─► scaler.transform / encoder.transform
            ─► ChurnMLP logits ─► Platt calibration ─► probability ─► risk tier

Artefacts loaded from ``models/``
---------------------------------
* ``scaler.pkl``, ``encoder.pkl``, ``feature_columns.pkl`` – fitted on train (notebook 02)
* ``best_model.pth``        – winning MLP weights (notebooks 04/05)
* ``model_config.json``     – architecture, winning strategy, calibration (a, b),
                              optimal threshold τ*, business costs (notebooks 04–07)
* ``shap_background.npy``   – 200 encoded training rows for SHAP (notebook 07)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from . import config
from .calibration import PlattCalibrator
from .data_preprocessing import (
    ID_COL,
    RAW_FEATURE_COLS,
    CleaningReport,
    get_feature_origins,
    load_preprocessors,
    prepare_inference_features,
)
from .model import CalibratedChurnModel, load_model

ADD_ON_SERVICES: list[str] = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "MultipleLines",
]


class ModelNotReadyError(RuntimeError):
    """Raised when required artefacts are missing (the notebooks have not been run yet)."""


def risk_tier(probability: float, high: float = config.HIGH_RISK_MIN,
              medium: float = config.MEDIUM_RISK_MIN) -> str:
    """
    Map a calibrated probability to the dashboard's traffic-light tier.

    High: p > 70 %  ·  Medium: 40 % ≤ p ≤ 70 %  ·  Low: p < 40 %
    """
    if probability > high:
        return "High"
    if probability >= medium:
        return "Medium"
    return "Low"


def summarize_services(row: Mapping[str, Any]) -> dict[str, Any]:
    """Compact "key services" summary shown in the risk table."""
    return {
        "contract": row.get("Contract"),
        "internet_service": row.get("InternetService"),
        "phone_service": row.get("PhoneService"),
        "tenure_months": int(row.get("tenure", 0)),
        "monthly_charges": round(float(row.get("MonthlyCharges", 0.0)), 2),
        "payment_method": row.get("PaymentMethod"),
        "add_ons": [s for s in ADD_ON_SERVICES if row.get(s) == "Yes"],
    }


class ChurnPredictor:
    """
    Loads the preprocessing objects + calibrated MLP and scores raw customer tables.

    Parameters
    ----------
    models_dir : path
        Folder containing the artefacts listed in the module docstring.
    device : str
        ``"cpu"`` is plenty for inference (a batch of 7,000 customers takes ms).

    Raises
    ------
    ModelNotReadyError
        If any artefact is missing or inconsistent.
    """

    def __init__(self, models_dir: str | Path = config.MODELS_DIR, device: str = "cpu") -> None:
        self.models_dir = Path(models_dir)
        self.device = torch.device(device)
        try:
            self.scaler, self.encoder, self.feature_columns = load_preprocessors(self.models_dir)
        except (FileNotFoundError, RuntimeError) as exc:
            raise ModelNotReadyError(str(exc)) from exc
        self.feature_origins = get_feature_origins(self.encoder)

        self.config = config.load_model_config(self.models_dir / config.MODEL_CONFIG_PATH.name)
        if not self.config:
            raise ModelNotReadyError(
                "models/model_config.json not found — run notebooks 04–07 to train and configure the model."
            )
        arch = self.config.get("architecture", {})
        input_dim = int(arch.get("input_dim", len(self.feature_columns)))
        if input_dim != len(self.feature_columns):
            raise ModelNotReadyError(
                f"Model expects {input_dim} features but the encoder produces {len(self.feature_columns)}; "
                "re-run the notebooks so all artefacts come from the same run."
            )
        try:
            base = load_model(self.models_dir / config.BEST_MODEL_PATH.name, input_dim, arch, self.device)
        except (FileNotFoundError, RuntimeError) as exc:
            raise ModelNotReadyError(f"Could not load best_model.pth: {exc}") from exc

        # Audit trail: "<strategy>-<first 8 hex chars of sha256(best_model.pth)>" is written into
        # every prediction log record, so each decision can be traced to the exact weights used.
        weights_hash = hashlib.sha256((self.models_dir / config.BEST_MODEL_PATH.name).read_bytes()).hexdigest()
        strategy_name = self.config.get("imbalance_strategy", {}).get("name", "mlp")
        self.model_version = f"{strategy_name}-{weights_hash[:8]}"

        self.calibrator = PlattCalibrator.from_dict(self.config.get("calibration"))
        self.model = CalibratedChurnModel(base, self.calibrator.a, self.calibrator.b).to(self.device).eval()

        threshold_cfg = self.config.get("threshold", {})
        self.default_threshold = float(threshold_cfg.get("optimal", config.NAIVE_THRESHOLD))
        costs = self.config.get("costs", {})
        self.c_offer = float(costs.get("c_offer", config.C_OFFER))
        self.c_lost = float(costs.get("c_lost", config.C_LOST))
        self.currency = str(costs.get("currency", config.CURRENCY_SYMBOL))
        tiers = self.config.get("risk_tiers", {})
        self.high_risk = float(tiers.get("high", config.HIGH_RISK_MIN))
        self.medium_risk = float(tiers.get("medium", config.MEDIUM_RISK_MIN))
        self._explainer = None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba_encoded(self, X: np.ndarray) -> np.ndarray:
        """Calibrated churn probabilities for already-encoded rows, shape ``(n,)``."""
        X = np.asarray(X, dtype=np.float32)
        if X.shape[0] == 0:
            return np.zeros(0)
        out = self.model(torch.from_numpy(X).to(self.device))
        return out.squeeze(1).cpu().numpy().astype(np.float64)

    def predict_dataframe(
        self, raw_df: pd.DataFrame, threshold: float | None = None
    ) -> tuple[pd.DataFrame, np.ndarray, CleaningReport]:
        """
        Score a raw customer table.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Uploaded rows with the 19 raw feature columns (customerID optional,
            Churn ignored if present).
        threshold : float, optional
            Decision threshold τ; defaults to the validation-optimal τ*.

        Returns
        -------
        (results, X, report)
            ``results`` — cleaned rows + ``churn_probability``, ``risk_tier``,
            ``flagged_for_offer``, ``rank``, sorted by probability (highest first);
            ``X`` — encoded features in the same row order (for explanations);
            ``report`` — the cleaning audit.
        """
        tau = self.default_threshold if threshold is None else float(threshold)
        if not 0.0 <= tau <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        X, clean_df, report = prepare_inference_features(raw_df, self.scaler, self.encoder)
        probs = self.predict_proba_encoded(X)

        results = clean_df.reset_index(drop=True).copy()
        results["churn_probability"] = probs
        results["risk_tier"] = [risk_tier(p, self.high_risk, self.medium_risk) for p in probs]
        results["flagged_for_offer"] = probs >= tau

        order = np.argsort(-probs, kind="stable")     # descending probability, stable for ties
        results = results.iloc[order].reset_index(drop=True)
        X = X[order]
        results.insert(0, "rank", np.arange(1, len(results) + 1))
        return results, X, report

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------
    @property
    def explainer(self):
        """Lazily-built :class:`~src.explainability.ChurnExplainer` (SHAP is slow to import)."""
        if self._explainer is None:
            from .explainability import ChurnExplainer

            bg_path = self.models_dir / config.SHAP_BACKGROUND_PATH.name
            if not bg_path.exists():
                raise ModelNotReadyError(
                    "models/shap_background.npy not found — run notebooks/07_final_evaluation.ipynb."
                )
            background = np.load(bg_path)
            self._explainer = ChurnExplainer(self.model, background, self.feature_columns,
                                             self.feature_origins, device=self.device)
        return self._explainer

    def explain_record(self, record: Mapping[str, Any], top_k: int = 5) -> dict[str, Any]:
        """
        Probability + top-k SHAP drivers for one raw customer record.

        Raises
        ------
        ValueError
            If the record cannot be scored (e.g. non-numeric tenure).
        """
        frame = pd.DataFrame([dict(record)])
        X, clean_df, report = prepare_inference_features(frame, self.scaler, self.encoder)
        if len(clean_df) == 0:
            reasons = "; ".join(r["reason"] for r in report.dropped_rows) or "invalid record"
            raise ValueError(f"Record could not be scored: {reasons}")
        raw = clean_df.iloc[0].to_dict()
        probability = float(self.predict_proba_encoded(X)[0])
        drivers = self.explainer.get_customer_top_drivers(X[0], raw_record=raw, top_k=top_k)
        return {
            "customer_id": str(raw.get(ID_COL)),
            "churn_probability": probability,
            "risk_tier": risk_tier(probability, self.high_risk, self.medium_risk),
            "base_value": self.explainer.base_value,
            "drivers": drivers,
            "explainer": self.explainer.method,
            "warnings": report.warnings,
        }

    # ------------------------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        """Configuration exposed to the UI via ``GET /api/config``."""
        return {
            "model_version": self.model_version,
            "strategy": self.config.get("imbalance_strategy", {}),
            "architecture": self.config.get("architecture", {}),
            "calibration": self.calibrator.to_dict(),
            "threshold": {"optimal": self.default_threshold,
                          **{k: v for k, v in self.config.get("threshold", {}).items() if k != "optimal"}},
            "costs": {"c_offer": self.c_offer, "c_lost": self.c_lost, "currency": self.currency},
            "risk_tiers": {"high": self.high_risk, "medium": self.medium_risk},
            "required_columns": RAW_FEATURE_COLS,
            "id_column": ID_COL,
            "test_metrics": self.config.get("test_metrics", {}),
            "n_features": len(self.feature_columns),
        }
