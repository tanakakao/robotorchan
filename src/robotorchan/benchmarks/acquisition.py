"""Benchmark utilities for acquisition-function evaluation cost."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor


@dataclass(frozen=True)
class AcquisitionBenchmarkResult:
    """One measured acquisition-function evaluation."""

    name: str
    batch_size: int
    q: int
    input_dim: int
    wall_time_seconds: float
    value_mean: float
    value_std: float


def benchmark_acquisition(
    name: str,
    acquisition: AcquisitionFunction,
    X: Tensor,
) -> AcquisitionBenchmarkResult:
    """Evaluate one acquisition on a fixed candidate batch."""
    if X.ndim < 3:
        raise ValueError("X must have shape [..., q, d].")
    batch_size = int(torch.tensor(X.shape[:-2]).prod().item())
    start = perf_counter()
    with torch.no_grad():
        values = acquisition(X)
    elapsed = perf_counter() - start
    flat_values = values.detach().reshape(-1).double()
    if not torch.isfinite(flat_values).all():
        raise ValueError("Acquisition benchmark produced non-finite values.")
    return AcquisitionBenchmarkResult(
        name=name,
        batch_size=batch_size,
        q=X.shape[-2],
        input_dim=X.shape[-1],
        wall_time_seconds=elapsed,
        value_mean=float(flat_values.mean()),
        value_std=float(flat_values.std(unbiased=False)),
    )
