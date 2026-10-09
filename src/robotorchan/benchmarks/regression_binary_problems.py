"""Single regression objective with a binary Pass/Fail outcome constraint."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _strength(X: Tensor) -> Tensor:
    return (1.0 - (X[..., 0] - 0.8).square() - (X[..., 1] - 0.3).square()).unsqueeze(-1)


def _pass_margin(X: Tensor) -> Tensor:
    return (X[..., 1] - 0.25 - 0.5 * X[..., 0]).unsqueeze(-1)


def strength_pass_labels(X: Tensor) -> Tensor:
    """Return canonical binary labels with Pass=1, Fail=0.

    The boundary is feasible, consistent with the benchmark g(X)>=0 rule.
    """
    return (_pass_margin(X) >= 0).to(dtype=X.dtype)


def strength_pass() -> BenchmarkProblem:
    """Maximize continuous strength under a binary Pass feasibility requirement.

    The deterministic signed margin is used for evaluation metrics only.
    Classification models must be trained on strength_pass_labels(X), not
    on the signed margin. The constrained optimum is (0.66, 0.58), with
    strength 0.902.
    """
    return BenchmarkProblem(
        name="strength_pass",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        objective=_strength,
        directions=("maximize",),
        variable_types=("continuous", "continuous"),
        constraints=_pass_margin,
        n_constraints=1,
        optimal_value=torch.tensor([0.902], dtype=torch.double),
    )


def register_regression_binary_problems(registry: BenchmarkProblemRegistry) -> None:
    """Register the regression objective plus binary constraint benchmark."""
    registry.register("strength_pass", strength_pass)
