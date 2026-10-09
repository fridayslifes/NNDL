r"""
Business-cost-based decision threshold (Part 6).

Cost model
----------
For a threshold τ, every customer with :math:`p \ge \tau` receives a retention offer.

=====================  =========================  ==========================
Outcome                Meaning                    Cost
=====================  =========================  ==========================
TP  (flagged churner)  offer sent, customer saved :math:`C_{offer}`
FP  (flagged loyal)    unnecessary offer          :math:`C_{offer}`
FN  (missed churner)   customer leaves            :math:`C_{lost}`
TN  (ignored loyal)    nothing happens            0
=====================  =========================  ==========================

.. math::

    \text{TotalCost}(\tau) = FN(\tau)\,C_{lost} + \big(TP(\tau) + FP(\tau)\big)\,C_{offer}

Assumption (state it in the viva): an offer always retains a true churner.
In reality only a fraction r of offers succeed; that would replace
:math:`TP\cdot C_{offer}` by :math:`TP\,(C_{offer} + (1-r)C_{lost})`.

Bayes-optimal threshold (a neat closed form)
--------------------------------------------
For one customer with calibrated churn probability p:

* expected cost if we **send** an offer   = :math:`C_{offer}`
* expected cost if we **don't**           = :math:`p \cdot C_{lost}`

Send the offer iff :math:`p\,C_{lost} > C_{offer}`, i.e.

.. math::

    \tau^{*}_{theory} = \frac{C_{offer}}{C_{lost}} = \frac{1{,}500}{15{,}000} = 0.10

So when a lost customer costs 10× an offer, we should contact anyone with more
than a 10 % churn risk. The empirical sweep on the validation set should land
near this value when the probabilities are well calibrated. Raising
:math:`C_{offer}` raises τ* (fewer, higher-risk customers targeted) — viva Q3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class CostParams:
    """
    Unit business costs.

    Parameters
    ----------
    c_offer : float
        Cost of one retention offer (₹1,500 by default). Must be ≥ 0.
    c_lost : float
        Revenue lost per undetected churner (₹15,000 by default). Must be ≥ 0.
    currency : str
        Display symbol only.
    """

    c_offer: float = config.C_OFFER
    c_lost: float = config.C_LOST
    currency: str = config.CURRENCY_SYMBOL

    def __post_init__(self) -> None:
        if self.c_offer < 0 or self.c_lost < 0:
            raise ValueError("Costs must be non-negative")
        if self.c_offer == 0 and self.c_lost == 0:
            raise ValueError("At least one of c_offer / c_lost must be positive")

    @property
    def breakeven_probability(self) -> float:
        """
        Theoretical optimal threshold C_offer / C_lost, clipped to [0, 1].

        Edge cases: C_lost = 0 → never worth sending an offer → 1.0;
        C_offer = 0 → offers are free → 0.0 (target everyone).
        """
        if self.c_lost == 0:
            return 1.0
        return float(min(max(self.c_offer / self.c_lost, 0.0), 1.0))


def default_thresholds() -> np.ndarray:
    """The brief's sweep grid: 0.01, 0.02, …, 0.99 (99 values)."""
    return np.round(np.arange(0.01, 1.0, 0.01), 2)


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same length")
    if y_true.size == 0:
        raise ValueError("Empty inputs")
    return y_true, y_prob


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, int]:
    """TP / FP / TN / FN when flagging customers with ``p >= threshold``."""
    y_true, y_prob = _validate(y_true, y_prob)
    flagged = y_prob >= threshold
    positive = y_true == 1
    return {
        "tp": int(np.sum(flagged & positive)),
        "fp": int(np.sum(flagged & ~positive)),
        "tn": int(np.sum(~flagged & ~positive)),
        "fn": int(np.sum(~flagged & positive)),
    }


def total_cost(tp: int, fp: int, fn: int, params: CostParams = CostParams()) -> float:
    """TotalCost = FN·C_lost + (TP + FP)·C_offer."""
    return float(fn * params.c_lost + (tp + fp) * params.c_offer)


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    params: CostParams = CostParams(),
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evaluate total cost at every threshold (vectorised with NumPy broadcasting).

    ``flags[k, i]`` is True when customer i is flagged at threshold k, so the
    confusion counts for all 99 thresholds are computed in a handful of array
    operations instead of a Python loop.

    Returns
    -------
    pd.DataFrame
        One row per threshold with columns threshold, tp, fp, tn, fn,
        n_flagged, precision, recall, total_cost, cost_per_customer.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    thresholds = default_thresholds() if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    if thresholds.size == 0:
        raise ValueError("thresholds must not be empty")

    flags = y_prob[None, :] >= thresholds[:, None]           # (K, n)
    positive = (y_true == 1)[None, :]                        # (1, n)
    tp = np.sum(flags & positive, axis=1)
    fp = np.sum(flags & ~positive, axis=1)
    fn = np.sum(~flags & positive, axis=1)
    tn = np.sum(~flags & ~positive, axis=1)
    cost = fn * params.c_lost + (tp + fp) * params.c_offer

    with np.errstate(divide="ignore", invalid="ignore"):   # 0/0 when nobody is flagged
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)

    return pd.DataFrame({
        "threshold": thresholds,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_flagged": tp + fp,
        "precision": precision,
        "recall": recall,
        "total_cost": cost.astype(float),
        "cost_per_customer": cost / y_true.size,
    })


def find_optimal_threshold(sweep: pd.DataFrame) -> pd.Series:
    """
    Row of the sweep with the minimum total cost (τ*).

    Ties (a flat cost plateau) are broken by the **smallest** threshold, i.e.
    the first minimum — ``idxmin`` returns the first occurrence.
    """
    if sweep.empty:
        raise ValueError("Empty sweep")
    return sweep.loc[sweep["total_cost"].idxmin()]


def baseline_costs(y_true: np.ndarray, params: CostParams = CostParams()) -> dict[str, float]:
    """
    Costs of the two naive policies.

    * Do nothing   – every churner is lost:          n_churners × C_lost
    * Target all   – everyone gets an offer:         n_customers × C_offer
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    return {
        "do_nothing": float(y_true.sum() * params.c_lost),
        "target_everyone": float(y_true.size * params.c_offer),
    }


def _pct(saving: float, reference: float) -> float | None:
    """Saving as a % of a reference cost; ``None`` when the reference is 0 (avoid ÷0)."""
    return None if reference == 0 else 100.0 * saving / reference


def savings_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    params: CostParams = CostParams(),
) -> dict[str, Any]:
    """
    Cost of a threshold policy and its savings versus both naive baselines.

    Returns
    -------
    dict
        threshold, confusion counts, n_customers, n_flagged, total_cost,
        do_nothing_cost, target_everyone_cost, savings_vs_do_nothing(_pct),
        savings_vs_target_everyone(_pct).
    """
    y_true, y_prob = _validate(y_true, y_prob)
    counts = confusion_counts(y_true, y_prob, threshold)
    cost = total_cost(counts["tp"], counts["fp"], counts["fn"], params)
    base = baseline_costs(y_true, params)
    save_nothing = base["do_nothing"] - cost
    save_all = base["target_everyone"] - cost
    return {
        "threshold": float(threshold),
        **counts,
        "n_customers": int(y_true.size),
        "n_flagged": counts["tp"] + counts["fp"],
        "total_cost": cost,
        "do_nothing_cost": base["do_nothing"],
        "target_everyone_cost": base["target_everyone"],
        "savings_vs_do_nothing": save_nothing,
        "savings_vs_do_nothing_pct": _pct(save_nothing, base["do_nothing"]),
        "savings_vs_target_everyone": save_all,
        "savings_vs_target_everyone_pct": _pct(save_all, base["target_everyone"]),
    }


def expected_cost_unlabeled(
    probs: np.ndarray, threshold: float, params: CostParams = CostParams()
) -> dict[str, float]:
    """
    Expected campaign economics for **unlabelled** customers (the web app's cost slider).

    With calibrated probabilities p_i the expected number of churners is Σ p_i, so

    * campaign cost          = n_targeted × C_offer
    * expected missed loss   = Σ_{p_i < τ} p_i × C_lost
    * expected total cost    = campaign cost + expected missed loss
    * do-nothing cost        = Σ_i p_i × C_lost
    * net savings            = do-nothing cost − expected total cost

    ``app/static/js/app.js`` implements exactly the same formulas client-side
    so the numbers update instantly while the slider moves.
    """
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    targeted = p >= threshold
    campaign = float(targeted.sum() * params.c_offer)
    missed = float(p[~targeted].sum() * params.c_lost)
    do_nothing = float(p.sum() * params.c_lost)
    total = campaign + missed
    return {
        "threshold": float(threshold),
        "n_customers": int(p.size),
        "n_targeted": int(targeted.sum()),
        "expected_churners": float(p.sum()),
        "expected_churners_reached": float(p[targeted].sum()),
        "campaign_cost": campaign,
        "expected_missed_loss": missed,
        "expected_total_cost": total,
        "do_nothing_cost": do_nothing,
        "net_savings": do_nothing - total,
    }


def format_currency(value: float, symbol: str = config.CURRENCY_SYMBOL) -> str:
    """Indian digit grouping, e.g. 1234567 → '₹12,34,567'."""
    negative = value < 0
    s = f"{abs(value):.0f}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    return f"{'-' if negative else ''}{symbol}{s}"


def write_recommendation(summary: dict[str, Any], naive: dict[str, Any], params: CostParams = CostParams(),
                         split_name: str = "validation") -> str:
    """
    Generate the one-paragraph written recommendation required by the brief.

    Parameters
    ----------
    summary : dict
        ``savings_summary`` at the optimal threshold τ*.
    naive : dict
        ``savings_summary`` at the default 0.5 threshold, for comparison.
    """
    fc = lambda v: format_currency(v, params.currency)  # noqa: E731
    pct = summary["savings_vs_do_nothing_pct"]
    pct_all = summary["savings_vs_target_everyone_pct"]
    return (
        f"We recommend a decision threshold of τ* = {summary['threshold']:.2f}: every customer whose "
        f"calibrated churn probability is at least {summary['threshold']:.0%} receives a retention offer. "
        f"On the {split_name} set ({summary['n_customers']:,} customers) this flags "
        f"{summary['n_flagged']:,} customers ({summary['n_flagged'] / summary['n_customers']:.1%}), catching "
        f"{summary['tp']:,} of {summary['tp'] + summary['fn']:,} churners, for an expected total cost of "
        f"{fc(summary['total_cost'])}. Doing nothing would cost {fc(summary['do_nothing_cost'])}, so the "
        f"policy saves {fc(summary['savings_vs_do_nothing'])}"
        + (f" ({pct:.1f}%)" if pct is not None else "")
        + f"; targeting everyone would cost {fc(summary['target_everyone_cost'])}, so it also saves "
        f"{fc(summary['savings_vs_target_everyone'])}"
        + (f" ({pct_all:.1f}%)" if pct_all is not None else "")
        + f" against that policy. The naive 0.5 threshold would flag only {naive['n_flagged']:,} customers and cost "
        f"{fc(naive['total_cost'])} — {fc(naive['total_cost'] - summary['total_cost'])} more — because each "
        f"missed churner ({fc(params.c_lost)}) costs {params.c_lost / max(params.c_offer, 1e-9):.0f}× an offer "
        f"({fc(params.c_offer)}), so it pays to accept many false alarms to avoid false negatives."
    )


def save_sweep(sweep: pd.DataFrame, path: str | Path) -> Path:
    """Persist the sweep table as CSV for the report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(path, index=False)
    return path
