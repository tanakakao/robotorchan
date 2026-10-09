"""Budget-aware and probabilistic benchmark evaluation metrics."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.runner import BenchmarkTrajectory


def _validate_curve(curve: Tensor) -> None:
    if curve.ndim != 1 or curve.numel() == 0 or not curve.is_floating_point():
        raise ValueError("curve must be a nonempty floating-point vector.")
    if torch.isnan(curve).any():
        raise ValueError("curve must not contain NaN.")


def area_under_curve(curve: Tensor, budgets: Tensor | None = None) -> Tensor:
    """Trapezoidal area with optional strictly increasing evaluation budgets."""
    _validate_curve(curve)
    if budgets is None:
        budgets = torch.arange(curve.numel(), dtype=curve.dtype, device=curve.device)
    if budgets.shape != curve.shape or budgets.device != curve.device:
        raise ValueError("budgets must match curve shape and device.")
    if not torch.isfinite(budgets).all() or not (budgets[1:] > budgets[:-1]).all():
        raise ValueError("budgets must be finite and strictly increasing.")
    if curve.numel() == 1:
        return curve.new_zeros(())
    return ((curve[1:] + curve[:-1]) * 0.5 * (budgets[1:] - budgets[:-1])).sum()


def best_feasible_value_curve(trajectory: BenchmarkTrajectory, *, maximize: bool = True) -> Tensor:
    """Running best feasible truth value; NaN until a feasible point exists."""
    if trajectory.Y_truth.ndim != 2 or trajectory.Y_truth.shape[1] != 1:
        raise ValueError("Single-objective truth with shape (n, 1) is required.")
    if trajectory.constraints.shape[0] != trajectory.Y_truth.shape[0]:
        raise ValueError("Constraint and truth history lengths must agree.")
    values = trajectory.Y_truth.squeeze(-1)
    feasible = (trajectory.constraints >= 0).all(dim=-1)
    oriented = values if maximize else -values
    best = oriented.masked_fill(~feasible, -torch.inf).cummax(dim=0).values
    return torch.where(torch.isneginf(best), torch.nan, best if maximize else -best)


def first_feasible_evaluation(constraints: Tensor) -> int | None:
    """One-based index of the first jointly feasible evaluation, or None."""
    if constraints.ndim != 2 or constraints.shape[0] == 0:
        raise ValueError("constraints must have shape (n, c), n > 0.")
    feasible = (constraints >= 0).all(dim=-1)
    indices = feasible.nonzero(as_tuple=False)
    return int(indices[0, 0]) + 1 if indices.numel() else None


def log_loss(labels: Tensor, probabilities: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Binary negative log-likelihood from predictive class probabilities."""
    if labels.shape != probabilities.shape or labels.numel() == 0:
        raise ValueError("Labels and probabilities must have matching nonempty shapes.")
    if not labels.is_floating_point() or not probabilities.is_floating_point():
        raise ValueError("Labels and probabilities must be floating tensors.")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Labels must be binary 0/1.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must lie within [0, 1].")
    if not 0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5).")
    p = probabilities.clamp(min=eps, max=1 - eps)
    return -(labels * p.log() + (1 - labels) * torch.log1p(-p)).mean()


def probability_calibration_error(
    labels: Tensor, probabilities: Tensor, *, n_bins: int = 10
) -> Tensor:
    """Expected calibration error for binary positive-class probabilities."""
    if labels.shape != probabilities.shape or labels.numel() == 0:
        raise ValueError("Labels and probabilities must have matching nonempty shapes.")
    if not labels.is_floating_point() or not probabilities.is_floating_point():
        raise ValueError("Labels and probabilities must be floating tensors.")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Labels must be binary 0/1.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must lie within [0, 1].")
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    p = probabilities.flatten()
    y = labels.flatten().to(dtype=p.dtype)
    bins = torch.clamp((p * n_bins).long(), max=n_bins - 1)
    error = p.new_zeros(())
    for index in range(n_bins):
        selected = bins == index
        if selected.any():
            error = error + selected.to(p.dtype).mean() * (
                p[selected].mean() - y[selected].mean()
            ).abs()
    return error
