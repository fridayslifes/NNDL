"""
Shared matplotlib styling and the project's standard figures.

Colour choices
--------------
* Categorical series use a fixed, colour-blind-validated order
  (blue, orange, aqua, yellow …) — "Retained" is always blue and "Churned" is
  always orange in every figure, so readers never have to re-learn the legend.
* Signed quantities (correlations, SHAP contributions) use a diverging
  blue ↔ red scale with a neutral grey midpoint: red = pushes towards churn.
* Magnitudes (confusion-matrix counts) use a single-hue blue ramp.
* Gridlines and axes are thin and light so the data carries the ink.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW = SERIES[:4]
RED = SERIES[7]
RETAINED_COLOR, CHURN_COLOR = BLUE, ORANGE

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
DIVERGING_MID = "#f0efec"

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("churn_blues", BLUE_RAMP)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "churn_diverging", ["#184f95", "#6da7ec", DIVERGING_MID, "#ee9291", "#b8302f"]
)


def apply_style() -> None:
    """Set project-wide rcParams (call once at the top of each notebook)."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Save at 150 dpi with a tight bounding box, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_learning_curves(history: Any, title: str = "MLP learning curves") -> plt.Figure:
    """
    Two panels: (left) train vs validation loss with the early-stopping epoch
    marked; (right) validation PR-AUC per epoch.
    """
    epochs = np.arange(1, len(history.train_loss) + 1)
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, history.train_loss, color=BLUE, label="Train loss")
    ax_loss.plot(epochs, history.val_loss, color=ORANGE, label="Validation loss")
    ax_loss.axvline(history.best_epoch, color=MUTED, linestyle="--", linewidth=1)
    ax_loss.annotate(f"best epoch {history.best_epoch}", xy=(history.best_epoch, max(history.val_loss)),
                     xytext=(4, 0), textcoords="offset points", color=INK_SECONDARY, fontsize=9)
    ax_loss.set(xlabel="Epoch", ylabel="BCE loss", title="Loss")
    ax_loss.legend()

    ax_auc.plot(epochs, history.val_pr_auc, color=AQUA)
    ax_auc.axvline(history.best_epoch, color=MUTED, linestyle="--", linewidth=1)
    ax_auc.set(xlabel="Epoch", ylabel="PR-AUC", title="Validation PR-AUC")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_optimizer_comparison(histories: Mapping[str, Any]) -> plt.Figure:
    """Train (left) and validation (right) loss per epoch for each optimiser — small multiples, one y-axis each."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for color, (name, hist) in zip(SERIES, histories.items()):
        epochs = np.arange(1, len(hist.train_loss) + 1)
        axes[0].plot(epochs, hist.train_loss, color=color, label=name)
        axes[1].plot(epochs, hist.val_loss, color=color, label=name)
    axes[0].set(xlabel="Epoch", ylabel="BCE loss", title="Training loss")
    axes[1].set(xlabel="Epoch", title="Validation loss")
    axes[0].legend()
    fig.suptitle("Optimiser convergence — identical MLP, seed 42, 30 epochs", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Evaluation figures
# ---------------------------------------------------------------------------
def plot_metric_bars(table: pd.DataFrame, metrics: Sequence[str], title: str) -> plt.Figure:
    """Grouped horizontal bars: one group per metric, one bar per model (fixed colour order)."""
    models = list(table.index)
    n_models = len(models)
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(metrics) * max(n_models, 2) / 2 + 1.2))
    height = 0.8 / n_models
    y = np.arange(len(metrics))
    for i, (model, color) in enumerate(zip(models, SERIES)):
        vals = table.loc[model, list(metrics)].to_numpy(dtype=float)
        pos = y - 0.4 + height * (i + 0.5)
        ax.barh(pos, vals, height=height * 0.85, color=color, label=model)
        for p, v in zip(pos, vals):
            ax.text(v + 0.005, p, f"{v:.3f}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(y, metrics)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion matrix") -> plt.Figure:
    """Annotated 2×2 heatmap (rows = actual, columns = predicted) with counts and row %."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm, cmap=SEQUENTIAL_CMAP)
    labels = ["Retained (0)", "Churned (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    names = [["TN", "FP"], ["FN", "TP"]]
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else INK
            ax.text(j, i, f"{names[i][j]}\n{cm[i, j]:,}\n({cm[i, j] / row_sums[i, 0]:.0%} of row)",
                    ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc_pr_curves(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], positive_rate: float) -> plt.Figure:
    """
    ROC (left) and precision–recall (right) curves for several models.

    Parameters
    ----------
    curves : mapping
        ``{label: (y_true, y_prob)}``.
    positive_rate : float
        Churn prevalence — the PR curve of a random classifier (horizontal reference line).
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    for color, (label, (y_true, y_prob)) in zip(SERIES, curves.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax_roc.plot(fpr, tpr, color=color, label=f"{label} (AUC {roc_auc_score(y_true, y_prob):.3f})")
        ax_pr.step(rec, prec, where="post", color=color,
                   label=f"{label} (AP {average_precision_score(y_true, y_prob):.3f})")
    ax_roc.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Random (0.500)")
    ax_pr.axhline(positive_rate, color=MUTED, linestyle="--", linewidth=1, label=f"Random ({positive_rate:.3f})")
    ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate (recall)", title="ROC curve",
               xlim=(0, 1), ylim=(0, 1.01))
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve", xlim=(0, 1), ylim=(0, 1.01))
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_pr.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_cost_curve(
    sweep: pd.DataFrame,
    optimal_threshold: float,
    optimal_cost: float,
    currency: str = "₹",
    naive_threshold: float = 0.5,
    breakeven: float | None = None,
    baselines: Mapping[str, float] | None = None,
) -> plt.Figure:
    """
    Total business cost vs threshold with τ* marked by a vertical dashed line.

    Optional horizontal reference lines show the do-nothing / target-everyone costs.
    """
    fig, ax = plt.subplots(figsize=(9, 4.6))
    scale = 1e5  # show costs in lakhs of rupees for readable ticks
    ax.plot(sweep["threshold"], sweep["total_cost"] / scale, color=BLUE, label="Total cost (model policy)")
    ax.axvline(optimal_threshold, color=RED, linestyle="--", linewidth=1.4,
               label=f"τ* = {optimal_threshold:.2f} (min cost)")
    ax.scatter([optimal_threshold], [optimal_cost / scale], color=RED, s=40, zorder=3,
               edgecolor=SURFACE, linewidth=2)
    naive_row = sweep.iloc[(sweep["threshold"] - naive_threshold).abs().argmin()]
    ax.scatter([naive_row["threshold"]], [naive_row["total_cost"] / scale], color=INK_SECONDARY, s=36,
               zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"naive τ = {naive_threshold:.1f}\n{currency}{naive_row['total_cost'] / scale:.2f} L",
                xy=(naive_row["threshold"], naive_row["total_cost"] / scale), xytext=(8, -28),
                textcoords="offset points", color=INK_SECONDARY, fontsize=9)
    ax.annotate(f"{currency}{optimal_cost / scale:.2f} L", xy=(optimal_threshold, optimal_cost / scale),
                xytext=(8, -16), textcoords="offset points", color=RED, fontsize=9)
    if breakeven is not None:
        ax.axvline(breakeven, color=MUTED, linestyle=":", linewidth=1.2,
                   label=f"Theory: C_offer / C_lost = {breakeven:.2f}")
    if baselines:
        styles = {"do_nothing": ("Do nothing", ORANGE), "target_everyone": ("Target everyone", AQUA)}
        for key, value in baselines.items():
            label, color = styles.get(key, (key, MUTED))
            ax.axhline(value / scale, color=color, linewidth=1.2, linestyle="-.", label=f"{label}")
    ax.set(xlabel="Decision threshold τ", ylabel=f"Total cost ({currency} lakh)",
           title="Business cost vs decision threshold (validation set)", xlim=(0, 1))
    ax.set_ylim(bottom=0.8 * optimal_cost / scale)   # room for the τ* label under the curve
    # Legend in the empty region below the rising curve and above the target-everyone line.
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=8, frameon=True,
              facecolor=SURFACE, edgecolor=GRID)
    fig.tight_layout()
    return fig


def plot_reliability(summaries: Mapping[str, Mapping[str, Any]]) -> plt.Figure:
    """Reliability diagram: mean predicted probability vs observed churn rate per bin."""
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Perfect calibration")
    for color, (label, s) in zip(SERIES, summaries.items()):
        ax.plot(s["prob_pred"], s["prob_true"], color=color, marker="o", markersize=5,
                markeredgecolor=SURFACE, label=f"{label} (Brier {s['brier']:.4f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate", xlim=(0, 1), ylim=(0, 1),
           title="Reliability diagram (validation)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_driver_bars(drivers: Sequence[Mapping[str, Any]], title: str) -> plt.Figure:
    """Horizontal diverging bars of one customer's top drivers (red = raises churn risk)."""
    labels = [f"{d['feature']} = {d['value']}" for d in drivers][::-1]
    values = [d["impact_pp"] for d in drivers][::-1]
    colors = [RED if v > 0 else BLUE for v in values]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(drivers) + 1.2))
    ax.barh(labels, values, color=colors, height=0.6)
    ax.axvline(0, color=AXIS, linewidth=1)
    for i, v in enumerate(values):
        ax.text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.1f} pp", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK_SECONDARY)
    span = max(abs(min(values)), abs(max(values)), 1) * 1.35
    ax.set_xlim(-span, span)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="SHAP contribution to churn probability (percentage points)", title=title)
    fig.tight_layout()
    return fig


def plot_global_importance(importance: pd.Series, top_n: int = 12) -> plt.Figure:
    """Bar chart of mean |SHAP| per raw feature (one colour: a single series)."""
    top = importance.head(top_n)[::-1] * 100
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(top) + 1.2))
    ax.barh(top.index, top.to_numpy(), color=BLUE, height=0.62)
    for i, v in enumerate(top.to_numpy()):
        ax.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Mean |SHAP| (percentage points of churn probability)",
           title="Global feature importance (one-hot groups aggregated)")
    fig.tight_layout()
    return fig
