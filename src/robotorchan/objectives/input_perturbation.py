"""BoTorch-compatible objectives for decision-time input perturbations."""

from __future__ import annotations

import torch
from botorch.acquisition.risk_measures import (
    CVaR,
    Expectation,
    RiskMeasureMCObjective,
    VaR,
    WorstCase,
)

from torch import Tensor

from .risk import RiskType


def make_input_perturbation_objective(
    risk_type: RiskType,
    *,
    n_w: int,
    alpha: float = 0.9,
) -> RiskMeasureMCObjective:
    """Build an MC objective that groups n_w perturbed scenarios per candidate."""
    if n_w < 1:
        raise ValueError("n_w must be positive.")
    if risk_type == "expectation":
        return Expectation(n_w=n_w)
    if risk_type == "worst_case":
        return WorstCase(n_w=n_w)
    if risk_type == "var":
        return VaR(alpha=alpha, n_w=n_w)
    if risk_type == "cvar":
        return CVaR(alpha=alpha, n_w=n_w)
    if risk_type == "mean_variance":
        raise ValueError("mean_variance is not yet certified for the InputPerturbation MC path.")
    if risk_type == "sn_ratio":
        raise ValueError("sn_ratio is not yet certified for the InputPerturbation MC path.")
    raise ValueError(f"Unsupported risk_type: {risk_type}")


def make_protected_perturbation_set(
    perturbations: Tensor,
    *,
    input_dim: int,
    protected_dims: list[int] | tuple[int, ...] = (),
) -> Tensor:
    """Expand design perturbations while keeping structural dimensions fixed."""
    if input_dim < 1:
        raise ValueError("input_dim must be positive.")
    normalized = tuple(dim % input_dim for dim in protected_dims)
    if len(set(normalized)) != len(normalized):
        raise ValueError("protected_dims must be unique after normalizing negative indices.")
    free_dims = [dim for dim in range(input_dim) if dim not in normalized]
    if perturbations.ndim != 2 or perturbations.shape[-1] != len(free_dims):
        raise ValueError("perturbations must have one column per unprotected input dimension.")
    expanded = torch.zeros(
        perturbations.shape[0], input_dim, dtype=perturbations.dtype, device=perturbations.device
    )
    expanded[:, free_dims] = perturbations
    return expanded
