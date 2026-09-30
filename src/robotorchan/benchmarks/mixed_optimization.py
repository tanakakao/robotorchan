"""Mixed-variable constrained benchmark scenarios for optimizer regression studies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.variable_space import MixedVariableSpace


@dataclass(frozen=True)
class MixedBenchmarkScenario:
    """Definition of one mixed-variable optimizer benchmark scenario."""

    name: str
    bounds: Tensor
    q: int
    variable_space: MixedVariableSpace
    constraints: CandidateConstraints | None = None
    fixed_features: dict[int, float] | None = None


class MixedInteractionAcquisition(AcquisitionFunction):
    """Synthetic acquisition with continuous, integer, and categorical interactions."""

    def __init__(self, category_targets: dict[float, float] | None = None) -> None:
        torch.nn.Module.__init__(self)
        self.category_targets = category_targets or {0.0: 0.2, 1.0: 0.8, 2.0: 0.5}

    def forward(self, X: Tensor) -> Tensor:
        continuous = X[..., 0]
        integer = X[..., 1] if X.shape[-1] > 1 else torch.zeros_like(continuous)
        if X.shape[-1] > 2:
            categorical = X[..., 2]
            target = torch.zeros_like(categorical)
            for category, value in self.category_targets.items():
                target = torch.where(categorical == category, value, target)
        else:
            target = torch.full_like(continuous, 0.65)
        point_value = -((continuous - target) ** 2) - 0.04 * ((integer - 2.0) ** 2)
        return point_value.sum(dim=-1)


def mixed_benchmark_scenarios(
    *, dtype: torch.dtype = torch.double
) -> dict[str, MixedBenchmarkScenario]:
    """Return canonical C+I, C+Cat, C+I+Cat, and constrained mixed scenarios."""
    ci_bounds = torch.tensor([[0.0, 0.0], [1.0, 4.0]], dtype=dtype)
    cic_bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 4.0, 2.0]], dtype=dtype)
    ccat_bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=dtype)
    constrained = CandidateConstraints(
        nonlinear_inequality_constraints=(
            (lambda x: 0.65 + 0.08 * x[..., 1] - 0.12 * x[..., 2] - x[..., 0], True),
        )
    )
    return {
        "continuous_integer": MixedBenchmarkScenario(
            "continuous_integer", ci_bounds, 1, MixedVariableSpace(ci_bounds, integer_dims=(1,))
        ),
        "continuous_categorical": MixedBenchmarkScenario(
            "continuous_categorical",
            ccat_bounds,
            1,
            MixedVariableSpace(ccat_bounds, categorical_values={1: [0.0, 1.0, 2.0]}),
        ),
        "full_mixed": MixedBenchmarkScenario(
            "full_mixed",
            cic_bounds,
            1,
            MixedVariableSpace(
                cic_bounds, integer_dims=(1,), categorical_values={2: [0.0, 1.0, 2.0]}
            ),
        ),
        "constrained_full_mixed": MixedBenchmarkScenario(
            "constrained_full_mixed",
            cic_bounds,
            1,
            MixedVariableSpace(
                cic_bounds, integer_dims=(1,), categorical_values={2: [0.0, 1.0, 2.0]}
            ),
            constraints=constrained,
        ),
    }
