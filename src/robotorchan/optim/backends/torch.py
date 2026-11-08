"""PyTorch acquisition optimizer backend.

This module keeps BoTorch acquisition optimization semantics and swaps only
the numerical candidate generator for ``gen_candidates_torch``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.generation.gen import gen_candidates_torch
from botorch.optim import optimize_acqf as botorch_optimize_acqf
from torch import Tensor

from robotorchan.optim.constraints.contracts import (
    CandidateConstraints,
    LinearConstraint,
    NonlinearConstraint,
    resolve_candidate_constraints,
)
from robotorchan.optim.constraints.evaluation import candidate_constraint_violation
from robotorchan.optim.constraints.validation import validate_candidate_constraints

TorchOptimizerName = Literal["adam", "adamw", "sgd"]

_TORCH_OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def optimize_acqf_torch(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int | None,
    *,
    optimizer: TorchOptimizerName | type[torch.optim.Optimizer] = "adam",
    options: dict[str, Any] | None = None,
    constraints: CandidateConstraints | None = None,
    inequality_constraints: tuple[LinearConstraint, ...] | None = None,
    equality_constraints: tuple[LinearConstraint, ...] | None = None,
    nonlinear_inequality_constraints: tuple[NonlinearConstraint, ...] | None = None,
    nonlinear_equality_constraints: tuple[NonlinearConstraint, ...] | None = None,
    nonlinear_equality_tolerance: float = 1e-6,
    fixed_features: dict[int, float | Tensor] | None = None,
    batch_initial_conditions: Tensor | None = None,
    timeout_sec: float | None = None,
    constraint_mode: Literal["strict", "penalty"] = "strict",
) -> tuple[Tensor, Tensor]:
    """Optimize a BoTorch acquisition function with a ``torch.optim`` backend.

    BoTorch remains responsible for restart initialization and best-candidate
    selection. ``gen_candidates_torch`` performs differentiable local optimization.

    Candidate constraints use a differentiable penalty during local search and
    either reject infeasible final candidates (strict) or return them (penalty).
    Nonlinear callables must be differentiable for gradient-based search.
    """
    if constraint_mode not in {"strict", "penalty"}:
        raise ValueError("constraint_mode must be either strict or penalty.")
    if q < 1:
        raise ValueError("q must be at least 1.")
    if num_restarts < 1:
        raise ValueError("num_restarts must be at least 1.")
    candidate_constraints = resolve_candidate_constraints(
        constraints,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        nonlinear_equality_constraints=nonlinear_equality_constraints,
        nonlinear_equality_tolerance=nonlinear_equality_tolerance,
    )
    validate_candidate_constraints(candidate_constraints, bounds=bounds, q=q)
    optimizer_class = _resolve_torch_optimizer(optimizer)
    gen_candidates = partial(
        gen_candidates_torch,
        optimizer=optimizer_class,
        timeout_sec=timeout_sec,
    )
    search_function = (
        _PenalizedAcquisition(acq_function, candidate_constraints)
        if (
            candidate_constraints.inequality_constraints
            or candidate_constraints.equality_constraints
            or candidate_constraints.nonlinear_inequality_constraints
        )
        else acq_function
    )
    candidate, value = botorch_optimize_acqf(
        acq_function=search_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=None if options is None else dict(options),
        fixed_features=fixed_features,
        batch_initial_conditions=batch_initial_conditions,
        gen_candidates=gen_candidates,
    )
    if (
        candidate_constraints.inequality_constraints
        or candidate_constraints.equality_constraints
        or candidate_constraints.nonlinear_inequality_constraints
    ):
        violation = candidate_constraint_violation(candidate, candidate_constraints)
        if constraint_mode == "strict" and bool((violation > 0).any()):
            raise ValueError("PyTorch optimizer found no feasible candidate.")
        with torch.no_grad():
            value = acq_function(candidate)
    return candidate, value


class _PenalizedAcquisition(AcquisitionFunction):
    """Apply a differentiable candidate-constraint penalty during torch search."""

    def __init__(self, acquisition: AcquisitionFunction, constraints: CandidateConstraints) -> None:
        super().__init__(model=getattr(acquisition, "model", None))
        self.acquisition = acquisition
        self.constraints = constraints

    def forward(self, X: Tensor) -> Tensor:
        value = self.acquisition(X)
        violation = candidate_constraint_violation(X, self.constraints)
        return value - 1e4 * violation


def _resolve_torch_optimizer(
    optimizer: TorchOptimizerName | type[torch.optim.Optimizer],
) -> type[torch.optim.Optimizer]:
    if isinstance(optimizer, str):
        try:
            return _TORCH_OPTIMIZERS[optimizer.lower()]
        except KeyError as error:
            supported = ", ".join(sorted(_TORCH_OPTIMIZERS))
            raise ValueError(
                f"Unknown torch optimizer {optimizer!r}. Supported optimizers: {supported}."
            ) from error
    if isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
        return optimizer
    raise TypeError("optimizer must be a supported name or torch.optim.Optimizer subclass.")
