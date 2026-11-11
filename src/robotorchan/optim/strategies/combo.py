"""Sequential binary COMBO-style Bayesian optimization."""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.acquisition.analytic import UpperConfidenceBound
from torch import Tensor

from robotorchan.models.standard.combo import COMBOSurrogate


def propose_combo_candidate(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    beta: float = 0.2,
    maximize: bool = True,
    max_dimensions: int = 16,
    fit_model: bool = True,
) -> Tensor:
    """Fit graph-diffusion GP and maximize UCB over unobserved binary points.

    Exact enumeration is limited to at most 20 dimensions. Returns [1, d].
    """
    if train_X.ndim != 2 or train_X.shape[1] < 1:
        raise ValueError("train_X must have shape [n, d].")
    d = train_X.shape[1]
    if max_dimensions < 1 or max_dimensions > 20 or d > max_dimensions:
        raise ValueError("Binary dimension exceeds the configured enumeration limit.")
    if beta < 0:
        raise ValueError("beta must be nonnegative.")
    model = COMBOSurrogate(train_X, train_Y)
    if fit_model:
        model.fit()
    model.eval()
    indices = torch.arange(2**d, device=train_X.device)
    bits = torch.arange(d, device=train_X.device)
    candidates = ((indices[:, None] >> bits) & 1).to(dtype=train_X.dtype)
    unseen = ~(candidates[:, None, :] == train_X[None, :, :]).all(dim=-1).any(dim=-1)
    if not bool(unseen.any()):
        raise ValueError("The binary search space is exhausted.")
    candidates = candidates[unseen]
    acquisition = UpperConfidenceBound(model, beta=beta, maximize=maximize)
    with torch.no_grad():
        values = acquisition(candidates.unsqueeze(-2))
    return candidates[torch.argmax(values)].unsqueeze(0)


def run_combo_sequential(
    train_X: Tensor,
    train_Y: Tensor,
    objective: Callable[[Tensor], Tensor],
    *,
    steps: int,
    beta: float = 0.2,
    maximize: bool = True,
    max_dimensions: int = 16,
    fit_model: bool = True,
) -> tuple[Tensor, Tensor]:
    """Propose, evaluate, append, and refit COMBO GP at each iteration.

    The objective accepts [1, d] and must return a finite [1, 1] tensor.
    """
    if steps < 0:
        raise ValueError("steps must be nonnegative.")
    X, Y = train_X.detach().clone(), train_Y.detach().clone()
    for _ in range(steps):
        candidate = propose_combo_candidate(
            X,
            Y,
            beta=beta,
            maximize=maximize,
            max_dimensions=max_dimensions,
            fit_model=fit_model,
        )
        value = objective(candidate)
        if value.shape != (1, 1):
            raise ValueError("objective must return a tensor of shape [1, 1].")
        if value.dtype != Y.dtype or value.device != Y.device:
            raise ValueError("objective output must match train_Y dtype and device.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("objective must return a finite observation.")
        X = torch.cat((X, candidate), dim=0)
        Y = torch.cat((Y, value.detach()), dim=0)
    return X, Y
