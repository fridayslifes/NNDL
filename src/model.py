r"""
PyTorch Multi-Layer Perceptron for churn prediction (framework component).

Architecture (Part 4 of the brief)
----------------------------------
::

    input (D_in = 40 encoded features)
      → Linear(D_in, 64) → BatchNorm1d(64) → ReLU → Dropout(0.3)
      → Linear(64, 32)   → BatchNorm1d(32) → ReLU → Dropout(0.3)
      → Linear(32, 1)    → Sigmoid                      → P(churn) ∈ (0, 1)

Why each component?
~~~~~~~~~~~~~~~~~~~
* **Linear** — an affine map :math:`h = Wx + b`; stacking two hidden layers lets
  the network learn non-linear interactions (e.g. *fibre optic × month-to-month*)
  that a single linear model cannot represent.
* **BatchNorm1d** — normalises each hidden unit over the mini-batch,
  :math:`\hat{h} = \gamma\,\frac{h - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta`.
  This stabilises the distribution of activations, allows larger learning
  rates and acts as a mild regulariser. At inference time (``model.eval()``)
  the running mean/variance collected during training replace the batch
  statistics, so a single customer can be scored deterministically.
  (The Linear bias before BatchNorm is redundant — BN subtracts the mean —
  but it is harmless and matches the architecture in the brief.)
* **ReLU** — :math:`\max(0, h)`. Its derivative is exactly 1 for positive inputs,
  so gradients do not shrink as they flow backwards (no vanishing gradient,
  unlike sigmoid whose derivative is at most 0.25 and ≈0 in the tails).
* **Dropout(0.3)** — during training each hidden unit is zeroed with probability
  0.3 and the survivors are scaled by 1/0.7. This prevents co-adaptation of
  units (an implicit ensemble) and reduces over-fitting. Disabled in ``eval()``.
* **Sigmoid output** — squashes the logit to a probability in (0, 1), which is
  what BCE expects and what the cost-based threshold sweep operates on.

The model exposes both ``logits(x)`` (pre-sigmoid scores, needed by
``BCEWithLogitsLoss`` and focal loss for numerical stability) and
``forward(x)`` (probabilities, used by ``BCELoss``, SHAP and the web app).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from . import config


class ChurnMLP(nn.Module):
    """
    Feed-forward network: [Linear → BatchNorm → ReLU → Dropout] × len(hidden_dims) → Linear → Sigmoid.

    Parameters
    ----------
    input_dim : int
        Number of encoded input features D_in.
    hidden_dims : sequence of int, default (64, 32)
        Width of each hidden layer.
    dropout : float, default 0.3
        Drop probability p ∈ [0, 1) used after every hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = config.HIDDEN_DIMS,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(h <= 0 for h in hidden_dims):
            raise ValueError("hidden_dims must be a non-empty sequence of positive ints")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout_p = float(dropout)

        layers: list[nn.Module] = []
        prev = self.input_dim
        for width in self.hidden_dims:
            layers += [
                nn.Linear(prev, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),            # not in-place: SHAP's DeepExplainer needs the input kept
                nn.Dropout(dropout),
            ]
            prev = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)       # produces the logit z
        self.sigmoid = nn.Sigmoid()

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-activation output z (shape ``(N, 1)``); P(churn) = σ(z)."""
        return self.head(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Churn probability σ(z), shape ``(N, 1)``."""
        return self.sigmoid(self.logits(x))


class CalibratedChurnModel(nn.Module):
    r"""
    Wrap a trained :class:`ChurnMLP` with a fixed Platt-scaling layer.

    Output: :math:`p_{cal} = \sigma(a \cdot z + b)` where z is the MLP logit and
    (a, b) were fitted on the validation set (see ``src/calibration.py``).
    Implemented with a frozen ``nn.Linear(1, 1)`` + ``nn.Sigmoid`` so SHAP's
    DeepExplainer can propagate through it like any other layer.
    Because a > 0 the mapping is monotonic: ranking (PR-AUC, ROC-AUC) is unchanged.
    """

    def __init__(self, base: ChurnMLP, a: float = 1.0, b: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.platt = nn.Linear(1, 1)
        with torch.no_grad():
            self.platt.weight.fill_(float(a))
            self.platt.bias.fill_(float(b))
        for param in self.platt.parameters():
            param.requires_grad_(False)
        self.out = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calibrated churn probability, shape ``(N, 1)``."""
        return self.out(self.platt(self.base.logits(x)))


def build_model(input_dim: int, model_config: dict | None = None) -> ChurnMLP:
    """Instantiate a :class:`ChurnMLP` using hyper-parameters from ``model_config`` (or defaults)."""
    model_config = model_config or {}
    return ChurnMLP(
        input_dim=input_dim,
        hidden_dims=tuple(model_config.get("hidden_dims", config.HIDDEN_DIMS)),
        dropout=float(model_config.get("dropout", config.DROPOUT)),
    )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (weights + biases + BatchNorm γ/β)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_weights(model: nn.Module, path: str | Path) -> Path:
    """Save only the ``state_dict`` (weights + BN running stats) — portable and safe to load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_model(
    path: str | Path,
    input_dim: int,
    model_config: dict | None = None,
    device: torch.device | str = "cpu",
) -> ChurnMLP:
    """
    Rebuild the architecture and load saved weights; returns the model in ``eval()`` mode.

    ``weights_only=True`` tells ``torch.load`` to refuse arbitrary pickled
    objects, which is the safe way to load checkpoints.

    Raises
    ------
    FileNotFoundError
        If the checkpoint does not exist.
    RuntimeError
        If the checkpoint's tensor shapes do not match ``input_dim``/hidden sizes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found: {path}")
    model = build_model(input_dim, model_config)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only argument
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_logits(
    model: ChurnMLP, X: np.ndarray, device: torch.device | str = "cpu", batch_size: int = 2048
) -> np.ndarray:
    """
    Pre-sigmoid scores z for every row of ``X`` (shape ``(n,)``).

    The model is put in ``eval()`` mode so Dropout is off and BatchNorm uses its
    running statistics — otherwise predictions would be random and batch-dependent.
    """
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    outputs = []
    for start in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        outputs.append(model.logits(xb).squeeze(1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def predict_proba(
    model: ChurnMLP, X: np.ndarray, device: torch.device | str = "cpu", batch_size: int = 2048
) -> np.ndarray:
    """Churn probabilities σ(z) for every row of ``X`` (shape ``(n,)``)."""
    z = predict_logits(model, X, device=device, batch_size=batch_size)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
