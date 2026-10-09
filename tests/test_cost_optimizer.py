"""Tests for the business-cost threshold logic."""

import numpy as np
import pytest

from src.cost_optimizer import (
    CostParams,
    baseline_costs,
    confusion_counts,
    expected_cost_unlabeled,
    find_optimal_threshold,
    format_currency,
    savings_summary,
    sweep_thresholds,
    total_cost,
)

Y = np.array([1, 1, 0, 0, 0, 1, 0, 0])
P = np.array([0.9, 0.4, 0.35, 0.05, 0.6, 0.15, 0.2, 0.01])
PARAMS = CostParams(c_offer=1500, c_lost=15000)


def test_confusion_counts_and_cost_formula():
    counts = confusion_counts(Y, P, 0.3)
    # flagged: 0.9(TP) 0.4(TP) 0.35(FP) 0.6(FP); missed churner 0.15 (FN)
    assert counts == {"tp": 2, "fp": 2, "tn": 3, "fn": 1}
    assert total_cost(counts["tp"], counts["fp"], counts["fn"], PARAMS) == 1 * 15000 + 4 * 1500


def test_sweep_minimum_matches_brute_force():
    sweep = sweep_thresholds(Y, P, PARAMS)
    best = find_optimal_threshold(sweep)
    brute = min(
        (total_cost(**{k: v for k, v in confusion_counts(Y, P, t).items() if k != "tn"}, params=PARAMS), t)
        for t in sweep["threshold"]
    )
    assert best["total_cost"] == pytest.approx(brute[0])
    assert best["threshold"] == pytest.approx(brute[1])     # ties -> smallest threshold


def test_baselines():
    base = baseline_costs(Y, PARAMS)
    assert base["do_nothing"] == 3 * 15000
    assert base["target_everyone"] == 8 * 1500


def test_breakeven_edge_cases():
    assert CostParams(1500, 15000).breakeven_probability == pytest.approx(0.1)
    assert CostParams(1500, 0).breakeven_probability == 1.0
    assert CostParams(0, 15000).breakeven_probability == 0.0
    with pytest.raises(ValueError):
        CostParams(-1, 100)
    with pytest.raises(ValueError):
        CostParams(0, 0)


def test_savings_percentage_is_none_without_churners():
    summary = savings_summary(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4]), 0.5, PARAMS)
    assert summary["do_nothing_cost"] == 0
    assert summary["savings_vs_do_nothing_pct"] is None


def test_expected_cost_unlabeled_matches_hand_calculation():
    probs = np.array([0.8, 0.3, 0.05])
    out = expected_cost_unlabeled(probs, 0.1, PARAMS)
    assert out["n_targeted"] == 2
    assert out["campaign_cost"] == 2 * 1500
    assert out["expected_missed_loss"] == pytest.approx(0.05 * 15000)
    assert out["do_nothing_cost"] == pytest.approx(1.15 * 15000)
    assert out["net_savings"] == pytest.approx(1.15 * 15000 - 3000 - 750)


def test_indian_currency_format():
    assert format_currency(1687500) == "₹16,87,500"
    assert format_currency(999) == "₹999"
    assert format_currency(-15000) == "-₹15,000"
