"""
Evaluation metrics focused on the churn (positive) class.

Why not accuracy?
-----------------
With 26.5 % churners, a model that predicts "no churn" for everyone is 73.5 %
accurate yet catches zero churners. We therefore report:

* **Precision** = TP / (TP + FP) — of the customers we flag, how many really churn?
  (Low precision = wasted retention offers.)
* **Recall** = TP / (TP + FN) — of the real churners, how many did we catch?
  (Low recall = lost revenue.)
* **F1** = 2·P·R / (P + R) — harmonic mean; high only when both are high.
* **PR-AUC** — area under the precision–recall curve, estimated with
  *average precision* :math:`AP = \\sum_n (R_n - R_{n-1}) P_n`. A random
  classifier scores ≈ the positive rate (0.265), so it is sensitive to how well
  the minority class is ranked. **Primary model-selection metric.**
* **ROC-AUC** — probability that a random churner is ranked above a random
  non-churner. Random = 0.5. It uses the false-positive *rate*, whose large
  denominator (all negatives) can make it look optimistic on imbalanced data.

Precision/recall/F1 depend on a decision threshold; PR-AUC and ROC-AUC do not
(they summarise the ranking over all thresholds).
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true has {y_true.size} values but y_prob has {y_prob.size}")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty array")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains NaN or infinite values")
    return y_true, y_prob


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """
    All churn-class metrics at a given decision threshold.

    A customer is predicted to churn when ``y_prob >= threshold``.

    Returns
    -------
    dict
        precision, recall, f1, pr_auc, roc_auc, accuracy, threshold,
        tp, fp, tn, fn, n_flagged. AUCs are NaN if only one class is present.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    if np.unique(y_true).size < 2:
        warnings.warn("Only one class present in y_true; AUC metrics are undefined (NaN).")
        pr_auc = roc_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_flagged": int(tp + fp),
    }


def find_f1_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None
) -> tuple[float, float]:
    """
    Threshold in ``thresholds`` (default 0.01…0.99) that maximises churn-class F1.

    Used only as a *reference* operating point — the project's decision
    threshold is chosen by business cost, not by F1.

    Returns
    -------
    (best_threshold, best_f1)
    """
    y_true, y_prob = _validate(y_true, y_prob)
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def bootstrap_pr_auc_difference(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """
    Paired bootstrap confidence interval for PR-AUC(model A) − PR-AUC(model B).

    Procedure: resample the *same* customer indices with replacement for both
    models (paired → the comparison is not confounded by which customers were
    drawn), recompute both PR-AUCs, and take the empirical percentiles of the
    difference. If the interval contains 0 the two models are statistically
    indistinguishable on this test set — the honest meaning of "matches".

    Returns
    -------
    dict with ``diff`` (observed), ``ci_low``, ``ci_high``, ``p_a_better``
    (share of resamples where A ≥ B) and ``n_valid`` resamples.
    """
    y_true, prob_a = _validate(y_true, prob_a)
    _, prob_b = _validate(y_true, prob_b)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n = y_true.size
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        if y_b.min() == y_b.max():   # a resample with one class has no PR-AUC
            continue
        diffs.append(average_precision_score(y_b, prob_a[idx]) - average_precision_score(y_b, prob_b[idx]))
    if not diffs:
        raise ValueError("No valid bootstrap resamples (need both classes)")
    diffs_arr = np.asarray(diffs)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(diffs_arr, [tail, 100 - tail])
    return {
        "diff": float(average_precision_score(y_true, prob_a) - average_precision_score(y_true, prob_b)),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_a_better": float(np.mean(diffs_arr >= 0)),
        "n_valid": int(diffs_arr.size),
        "confidence": confidence,
    }


METRIC_COLUMNS: list[str] = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
METRIC_LABELS: dict[str, str] = {
    "precision": "Precision", "recall": "Recall", "f1": "F1 (Churn)",
    "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
}


def metrics_table(
    results: Mapping[str, Mapping[str, Any]],
    extra_columns: Mapping[str, Mapping[str, Any]] | None = None,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Build a report-ready comparison table.

    Parameters
    ----------
    results : mapping
        ``{model_name: classification_metrics(...) dict}``.
    extra_columns : mapping, optional
        ``{model_name: {column: value}}`` prepended (e.g. the "Implementation" column).
    """
    rows = []
    for name, m in results.items():
        row: dict[str, Any] = {"Model": name}
        if extra_columns and name in extra_columns:
            row.update(extra_columns[name])
        for key in METRIC_COLUMNS:
            row[METRIC_LABELS[key]] = round(float(m[key]), decimals)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no ``tabulate`` dependency)."""
    frame = df.reset_index()
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for _, row in frame.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
