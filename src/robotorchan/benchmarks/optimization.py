"""Benchmark utilities for acquisition optimizer backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.constraints.evaluation import (
    candidate_constraint_violation,
    candidate_is_feasible,
)

Optimizer = Callable[..., tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class BenchmarkResult:
    """One measured optimizer run."""

    name: str
    candidate: Tensor
    acquisition_value: Tensor
    wall_time_seconds: float
    acquisition_evaluations: int
    feasible: bool | None
    constraint_violation: float | None
    seed: int | None


class CountingAcquisition(AcquisitionFunction):
    """Transparent acquisition wrapper that counts evaluated q-batches."""

    def __init__(self, acquisition: AcquisitionFunction) -> None:
        torch.nn.Module.__init__(self)
        self.acquisition = acquisition
        self.evaluations = 0

    def forward(self, X: Tensor) -> Tensor:
        batch_shape = X.shape[:-2]
        evaluations = 1
        for size in batch_shape:
            evaluations *= size
        self.evaluations += evaluations
        return self.acquisition(X)


def benchmark_optimizer(
    name: str,
    optimizer: Optimizer,
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    *,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    constraints: CandidateConstraints | None = None,
    seed: int | None = None,
    equality_tolerance: float = 1e-6,
) -> BenchmarkResult:
    """Run one optimizer with common timing, evaluation, and feasibility metrics."""
    kwargs = dict(optimizer_kwargs or {})
    if constraints is not None:
        kwargs.setdefault("constraints", constraints)
    if seed is not None:
        kwargs.setdefault("seed", seed)

    counted = CountingAcquisition(acq_function)
    start = perf_counter()
    candidate, _ = optimizer(counted, bounds, q, **kwargs)
    elapsed = perf_counter() - start

    with torch.no_grad():
        raw_value = acq_function(candidate.unsqueeze(0)).reshape(())
    feasible = None
    constraint_violation = None
    if constraints is not None:
        batched_candidate = candidate.unsqueeze(0)
        feasible = bool(
            candidate_is_feasible(
                batched_candidate,
                constraints,
                equality_tolerance=equality_tolerance,
            ).item()
        )
        constraint_violation = float(
            candidate_constraint_violation(
                batched_candidate,
                constraints,
                equality_tolerance=equality_tolerance,
            ).item()
        )
    return BenchmarkResult(
        name=name,
        candidate=candidate,
        acquisition_value=raw_value,
        wall_time_seconds=elapsed,
        acquisition_evaluations=counted.evaluations,
        feasible=feasible,
        constraint_violation=constraint_violation,
        seed=seed,
    )


def benchmark_optimizers(
    optimizers: Mapping[str, tuple[Optimizer, Mapping[str, Any]]],
    acq_factory: Callable[[], AcquisitionFunction],
    bounds: Tensor,
    q: int,
    *,
    seeds: tuple[int, ...] = (0,),
    constraints: CandidateConstraints | None = None,
) -> list[BenchmarkResult]:
    """Benchmark multiple optimizers and seeds with fresh acquisition instances."""
    results: list[BenchmarkResult] = []
    for name, (optimizer, kwargs) in optimizers.items():
        for seed in seeds:
            results.append(
                benchmark_optimizer(
                    name,
                    optimizer,
                    acq_factory(),
                    bounds,
                    q,
                    optimizer_kwargs=kwargs,
                    constraints=constraints,
                    seed=seed,
                )
            )
    return results
