r"""
Focal Loss implemented from scratch in PyTorch (no external focal-loss package).

Reference: Lin, Goyal, Girshick, He & Dollár (2017), *Focal Loss for Dense
Object Detection*. We deliberately do **not** use
``torchvision.ops.sigmoid_focal_loss`` — every term is written out below.

Definition
----------
Let p = σ(z) be the predicted churn probability and y ∈ {0, 1} the label. Define

.. math::

    p_t = \begin{cases} p & y = 1 \\ 1 - p & y = 0 \end{cases}
    \qquad
    \alpha_t = \begin{cases} \alpha & y = 1 \\ 1 - \alpha & y = 0 \end{cases}

.. math::

    FL(p_t) = -\alpha_t\,(1 - p_t)^{\gamma}\,\log(p_t)

How it differs from BCE (viva Q5)
---------------------------------
Standard BCE is :math:`-\log(p_t)`. Focal loss multiplies it by two factors:

* **Modulating factor** :math:`(1 - p_t)^\gamma`. For an *easy* example the
  model already gets right (p_t = 0.95) and γ = 2, the factor is
  (0.05)² = 0.0025 — its loss is cut by 400×. For a *hard* example
  (p_t = 0.3) the factor is 0.49 — barely reduced. Training therefore focuses
  on the hard, uncertain customers instead of the thousands of obvious
  loyal ones that dominate an imbalanced dataset.
* **Class weight** :math:`\alpha_t`. With α = 0.75 each churner (minority
  class) counts 3× as much as a non-churner (0.75 vs 0.25).

With γ = 0 and α = 0.5 focal loss equals exactly 0.5 × BCE (a unit test checks this).

Numerical stability
-------------------
We work from **logits** and compute :math:`\log p = \log\sigma(z)` with
``F.logsigmoid`` (implemented as :math:`-\mathrm{softplus}(-z)`), which never
evaluates log(0) even when σ(z) rounds to exactly 0 or 1 in float32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """
    Binary focal loss :math:`-\\alpha_t (1-p_t)^\\gamma \\log p_t`.

    Parameters
    ----------
    alpha : float, default 0.75
        Weight of the positive (churn) class in [0, 1]; negatives get 1 − α.
    gamma : float, default 2.0
        Focusing parameter γ ≥ 0. γ = 0 removes the modulating factor.
    reduction : {"mean", "sum", "none"}, default "mean"
    from_logits : bool, default True
        ``True`` → ``inputs`` are raw logits z (recommended, numerically stable);
        ``False`` → ``inputs`` are probabilities in [0, 1].
    eps : float, default 1e-7
        Clamp used only when ``from_logits=False`` to avoid log(0).
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
        from_logits: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss.

        Parameters
        ----------
        inputs : Tensor, shape (N,) or (N, 1)
            Logits (default) or probabilities.
        targets : Tensor, same number of elements, values in {0, 1}

        Returns
        -------
        Tensor
            Scalar for "mean"/"sum", per-sample losses for "none".
        """
        targets = targets.to(dtype=inputs.dtype).reshape(inputs.shape)

        if self.from_logits:
            log_p = F.logsigmoid(inputs)          # log σ(z)
            log_1m_p = F.logsigmoid(-inputs)      # log(1 − σ(z)) = log σ(−z)
        else:
            p = inputs.clamp(self.eps, 1.0 - self.eps)
            log_p, log_1m_p = torch.log(p), torch.log1p(-p)

        # log p_t selects log p for churners and log(1 − p) for non-churners.
        log_p_t = targets * log_p + (1.0 - targets) * log_1m_p
        p_t = torch.exp(log_p_t)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        modulating = (1.0 - p_t).pow(self.gamma)

        loss = -alpha_t * modulating * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction!r}, from_logits={self.from_logits}"
