"""Ground-truth boundary and classification active-learning benchmarks."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _dummy_objective(X: Tensor) -> Tensor:
    """Return a neutral objective; active-learning scores are separate."""
    return X.new_zeros((*X.shape[:-1], 1))


def circle_margin(X: Tensor) -> Tensor:
    """Signed distance-like level set; positive values indicate class 1."""
    return (0.16 - (X[..., 0] - 0.5).square() - (X[..., 1] - 0.5).square()).unsqueeze(-1)


def wave_margin(X: Tensor) -> Tensor:
    """Nonlinear signed level set with both classes in the unit square."""
    return (X[..., 1] - 0.5 - 0.2 * torch.sin(4.0 * torch.pi * X[..., 0])).unsqueeze(-1)


def active_learning_labels(X: Tensor, *, boundary: str) -> Tensor:
    """Convert signed boundary truth to canonical binary labels."""
    margins = {"circle": circle_margin, "wave": wave_margin}
    try:
        evaluate = margins[boundary]
    except KeyError as error:
        raise ValueError(f"Unknown boundary: {boundary}") from error
    return (evaluate(X) >= 0).to(dtype=X.dtype)


def circle_boundary() -> BenchmarkProblem:
    """Discover a circular classification boundary on the unit square."""
    return BenchmarkProblem(
        name="circle_boundary",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_dummy_objective,
        directions=("maximize",),
        variable_types=("continuous", "continuous"),
        constraints=circle_margin,
        n_constraints=1,
    )


def wave_boundary() -> BenchmarkProblem:
    """Discover a sinusoidal classification boundary on the unit square."""
    return BenchmarkProblem(
        name="wave_boundary",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_dummy_objective,
        directions=("maximize",),
        variable_types=("continuous", "continuous"),
        constraints=wave_margin,
        n_constraints=1,
    )


def boundary_mae(true_margin: Tensor, predicted_margin: Tensor) -> Tensor:
    """Mean absolute signed-margin error on a fixed reference set."""
    if true_margin.shape != predicted_margin.shape or true_margin.numel() == 0:
        raise ValueError("Margins must have matching nonempty shapes.")
    if not torch.isfinite(true_margin).all() or not torch.isfinite(predicted_margin).all():
        raise ValueError("Margins must be finite.")
    return (true_margin - predicted_margin).abs().mean()


def classification_accuracy(true_labels: Tensor, probabilities: Tensor) -> Tensor:
    """Binary accuracy from predictive class probabilities at threshold 0.5."""
    if true_labels.shape != probabilities.shape or true_labels.numel() == 0:
        raise ValueError("Labels and probabilities must have matching nonempty shapes.")
    if not torch.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    if not ((true_labels == 0) | (true_labels == 1)).all():
        raise ValueError("Labels must be binary 0/1.")
    return ((probabilities >= 0.5) == true_labels.bool()).to(probabilities.dtype).mean()


def brier_score(true_labels: Tensor, probabilities: Tensor) -> Tensor:
    """Mean squared probability error against observed binary labels."""
    if true_labels.shape != probabilities.shape or true_labels.numel() == 0:
        raise ValueError("Labels and probabilities must have matching nonempty shapes.")
    if not torch.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    if not ((true_labels == 0) | (true_labels == 1)).all():
        raise ValueError("Labels must be binary 0/1.")
    return (probabilities - true_labels.to(probabilities.dtype)).square().mean()


def register_active_learning_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register both classification boundary benchmark problems."""
    registry.register("circle_boundary", circle_boundary)
    registry.register("wave_boundary", wave_boundary)
