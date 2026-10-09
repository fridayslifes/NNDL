"""
Training utilities: mini-batch loop, early stopping and class-imbalance strategies.

Training loop (one epoch)
-------------------------
For every mini-batch (x, y):

1. ``optimizer.zero_grad()``   – clear gradients accumulated from the previous step.
2. forward pass                 – probabilities (BCELoss) or logits (weighted BCE / focal).
3. ``loss.backward()``          – autograd applies the chain rule through every layer
                                  (the same maths we derived by hand for the perceptron).
4. ``optimizer.step()``         – update θ using the optimiser's rule (SGD / RMSprop / Adam).

After each epoch the model is switched to ``eval()`` (Dropout off, BatchNorm
running stats) and the validation loss is measured.

L2 weight decay
---------------
Passing ``weight_decay=λ`` to the optimiser adds λ·θ to every gradient, which
is equivalent to minimising ``loss + (λ/2)·‖θ‖²``. It shrinks weights towards
zero and discourages the network from relying heavily on any single feature.

Early stopping
--------------
Validation loss typically falls, flattens, then rises as the network starts to
memorise the training set. We keep a copy of the weights with the **lowest
validation loss** and stop after ``patience`` epochs with no improvement, then
restore that best copy. This is a form of regularisation: it picks the model
complexity (number of effective training steps) using held-out data.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .focal_loss import FocalLoss
from .metrics import classification_metrics
from .model import ChurnMLP, predict_proba, save_weights


@dataclass
class TrainingHistory:
    """Per-epoch curves and early-stopping summary returned by :func:`train_mlp`."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_pr_auc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    epochs_run: int = 0
    stopped_early: bool = False
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EarlyStopping:
    """
    Stop training when validation loss has not improved for ``patience`` epochs.

    Parameters
    ----------
    patience : int
        Number of consecutive non-improving epochs tolerated (brief: 10).
    min_delta : float
        Minimum decrease that counts as an improvement.
    """

    def __init__(self, patience: int = config.PATIENCE, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.best_state: dict[str, torch.Tensor] | None = None

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Record this epoch's validation loss; return ``True`` if training should stop.

        On improvement a deep copy of ``state_dict`` is kept in memory (on CPU),
        so later epochs cannot overwrite the best weights.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Load the best weights back into ``model`` (no-op if none were recorded)."""
        if self.best_state is not None:
            model.load_state_dict(copy.deepcopy(self.best_state))


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    seed: int = config.SEED,
    drop_last: bool = True,
) -> DataLoader:
    """
    Wrap NumPy arrays in a reproducible PyTorch ``DataLoader``.

    ``drop_last=True`` matters: 4,225 training rows / 64 leaves a final batch
    of **one** customer, and BatchNorm cannot compute a variance from a single
    sample (PyTorch raises "Expected more than 1 value per channel"). Dropping
    that partial batch avoids the crash; because the data is reshuffled each
    epoch, a different row is left out every time, so no customer is ignored.

    A seeded ``torch.Generator`` makes the shuffle order identical across runs.
    """
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    if len(X) == 0:
        raise ValueError("Cannot build a DataLoader from zero rows")
    dataset = TensorDataset(
        torch.as_tensor(np.asarray(X, dtype=np.float32)),
        torch.as_tensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1),
    )
    generator = torch.Generator().manual_seed(seed)
    effective_drop_last = drop_last and len(X) > batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, drop_last=effective_drop_last)


def _loss(model: ChurnMLP, xb: torch.Tensor, yb: torch.Tensor,
          criterion: nn.Module, on_logits: bool) -> torch.Tensor:
    """Feed logits or probabilities to ``criterion`` depending on what it expects."""
    outputs = model.logits(xb) if on_logits else model(xb)
    return criterion(outputs, yb)


@torch.no_grad()
def evaluate_loss(model: ChurnMLP, X: np.ndarray, y: np.ndarray, criterion: nn.Module,
                  on_logits: bool, device: torch.device | str) -> float:
    """Loss on a full split in ``eval()`` mode (Dropout off, BN running statistics)."""
    model.eval()
    xb = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    yb = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device).reshape(-1, 1)
    return float(_loss(model, xb, yb, criterion, on_logits).item())


def train_mlp(
    model: ChurnMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    criterion: nn.Module,
    *,
    criterion_on_logits: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    batch_size: int = config.BATCH_SIZE,
    max_epochs: int = config.MAX_EPOCHS,
    patience: int | None = config.PATIENCE,
    seed: int = config.SEED,
    device: torch.device | str | None = None,
    checkpoint_path: str | Path | None = None,
    verbose: bool = True,
    print_every: int = 10,
) -> tuple[ChurnMLP, TrainingHistory]:
    """
    Train ``model`` with mini-batches, tracking validation loss and PR-AUC each epoch.

    Parameters
    ----------
    model : ChurnMLP
        Freshly initialised network (call ``config.set_seed`` before building it).
    X_train, y_train, X_val, y_val : np.ndarray
        Encoded features and 0/1 labels. The validation split is *only* used to
        monitor loss for early stopping — never for gradient updates.
    criterion : nn.Module
        Loss function (``BCELoss``, ``BCEWithLogitsLoss``, ``FocalLoss``...).
    criterion_on_logits : bool
        ``True`` if ``criterion`` expects logits rather than probabilities.
    optimizer : torch.optim.Optimizer, optional
        Defaults to ``Adam(lr=lr, weight_decay=weight_decay)``.
    patience : int or None
        Early-stopping patience; ``None`` trains for exactly ``max_epochs``
        (used by the optimiser-comparison experiment).
    checkpoint_path : path, optional
        Where to save the best ``state_dict``.

    Returns
    -------
    (model, history)
        ``model`` holds the best weights (lowest validation loss) and is in eval mode.
    """
    device = torch.device(device) if device is not None else config.get_device()
    model.to(device)
    if isinstance(criterion, nn.Module):
        criterion.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    stopper = EarlyStopping(patience=patience) if patience else None
    history = TrainingHistory()
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()   # Dropout active, BatchNorm uses mini-batch statistics
        running, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = _loss(model, xb, yb, criterion, criterion_on_logits)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss became {loss.item()} at epoch {epoch}; lower the learning rate.")
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)

        train_loss = running / max(seen, 1)
        val_loss = evaluate_loss(model, X_val, y_val, criterion, criterion_on_logits, device)
        val_prob = predict_proba(model, X_val, device=device)
        val_pr_auc = classification_metrics(y_val, val_prob)["pr_auc"]

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_pr_auc.append(val_pr_auc)
        history.epochs_run = epoch

        if verbose and (epoch == 1 or epoch % print_every == 0):
            print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
                  f"| val PR-AUC {val_pr_auc:.4f}")

        if stopper is not None and stopper.step(val_loss, model, epoch):
            history.stopped_early = True
            if verbose:
                print(f"Early stopping at epoch {epoch}: no val-loss improvement for "
                      f"{stopper.patience} epochs (best epoch {stopper.best_epoch}).")
            break

    if stopper is not None:
        stopper.restore_best(model)
        history.best_epoch = stopper.best_epoch
        history.best_val_loss = stopper.best_loss
    else:
        history.best_epoch = history.epochs_run
        history.best_val_loss = history.val_loss[-1]

    history.seconds = time.perf_counter() - start_time
    model.eval()
    if checkpoint_path is not None:
        save_weights(model, checkpoint_path)
    return model, history


# ---------------------------------------------------------------------------
# Class-imbalance strategies (Part 5)
# ---------------------------------------------------------------------------
IMBALANCE_STRATEGIES: dict[str, str] = {
    "plain_bce": "1. Plain BCE",
    "weighted_bce": "2. Weighted BCE",
    "oversampling": "3. Random Oversampling",
    "focal_loss": "4. Focal Loss (From Scratch)",
}


@dataclass
class StrategySetup:
    """Everything that differs between imbalance experiments."""

    name: str
    label: str
    X_fit: np.ndarray
    y_fit: np.ndarray
    criterion: nn.Module
    on_logits: bool
    details: dict[str, Any]


def prepare_imbalance_strategy(
    strategy: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = config.SEED,
    focal_alpha: float = config.FOCAL_ALPHA,
    focal_gamma: float = config.FOCAL_GAMMA,
) -> StrategySetup:
    """
    Build the training data + loss for one imbalance-handling strategy.

    * ``plain_bce``    – unweighted ``nn.BCELoss`` on probabilities.
    * ``weighted_bce`` – ``nn.BCEWithLogitsLoss(pos_weight = N_neg / N_pos)``: every
      churner's loss term is multiplied by ≈2.77, so both classes contribute
      equally to the total loss.
    * ``oversampling`` – ``RandomOverSampler`` duplicates random churners **in the
      training split only** until the classes are balanced (validation/test keep
      their true 26.5 % churn rate so metrics stay honest). Loss = plain BCE.
    * ``focal_loss``   – our from-scratch :class:`FocalLoss` (α = 0.75, γ = 2).

    Raises
    ------
    ValueError
        Unknown strategy, or the training labels lack one of the classes
        (``pos_weight`` would divide by zero).
    """
    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {list(IMBALANCE_STRATEGIES)}")
    y_train = np.asarray(y_train).astype(int).reshape(-1)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Training labels must contain both classes (n_pos={n_pos}, n_neg={n_neg})")

    label = IMBALANCE_STRATEGIES[strategy]
    details: dict[str, Any] = {"n_pos": n_pos, "n_neg": n_neg}

    if strategy == "plain_bce":
        return StrategySetup(strategy, label, X_train, y_train, nn.BCELoss(), False, details)

    if strategy == "weighted_bce":
        pos_weight = n_neg / n_pos
        details["pos_weight"] = pos_weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        return StrategySetup(strategy, label, X_train, y_train, criterion, True, details)

    if strategy == "oversampling":
        from imblearn.over_sampling import RandomOverSampler

        sampler = RandomOverSampler(random_state=seed)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        details.update({"n_rows_after": int(len(y_res)), "n_pos_after": int((y_res == 1).sum())})
        return StrategySetup(strategy, label, np.asarray(X_res, dtype=np.float32),
                             np.asarray(y_res).astype(int), nn.BCELoss(), False, details)

    # focal_loss
    details.update({"alpha": focal_alpha, "gamma": focal_gamma})
    return StrategySetup(strategy, label, X_train, y_train,
                         FocalLoss(alpha=focal_alpha, gamma=focal_gamma), True, details)
