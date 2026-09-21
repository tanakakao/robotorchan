"""Deterministic predictive benchmark for non-GP surrogate models.

Run manually with:
    python benchmarks/non_gp_surrogates.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor

from robotorchan.models import (
    ExtraTreesSurrogate,
    GradientBoostingSurrogate,
    HistGradientBoostingSurrogate,
    RandomForestSurrogate,
)


@dataclass(frozen=True)
class BenchmarkResult:
    """One surrogate benchmark result."""

    model: str
    rmse: float
    mean_posterior_std: float
    fit_seconds: float
    posterior_seconds: float


def _target(X: Tensor) -> Tensor:
    return (
        torch.sin(2.0 * torch.pi * X[:, :1])
        + 0.35 * torch.cos(4.0 * torch.pi * X[:, 1:2])
        + 0.2 * X[:, 2:3].square()
    )


def make_data(
    *,
    n_train: int = 96,
    n_test: int = 128,
    d: int = 6,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a reproducible nonlinear regression benchmark."""
    generator = torch.Generator().manual_seed(seed)
    train_X = torch.rand(n_train, d, generator=generator, dtype=torch.double)
    test_X = torch.rand(n_test, d, generator=generator, dtype=torch.double)
    noise = 0.03 * torch.randn(n_train, 1, generator=generator, dtype=torch.double)
    return train_X, _target(train_X) + noise, test_X, _target(test_X)


def run_benchmark(seed: int = 0) -> list[BenchmarkResult]:
    """Fit supported non-GP surrogates and collect predictive diagnostics."""
    train_X, train_Y, test_X, test_Y = make_data(seed=seed)
    factories: dict[str, Callable[[], object]] = {
        "RandomForest": lambda: RandomForestSurrogate(
            train_X, train_Y, n_estimators=64, random_state=seed
        ),
        "ExtraTrees": lambda: ExtraTreesSurrogate(
            train_X, train_Y, n_estimators=64, random_state=seed
        ),
        "GradientBoosting": lambda: GradientBoostingSurrogate(
            train_X, train_Y, n_members=8, random_state=seed, n_estimators=60
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingSurrogate(
            train_X, train_Y, n_members=8, random_state=seed, max_iter=60
        ),
    }
    results = []
    for name, factory in factories.items():
        model = factory()
        start = perf_counter()
        model.fit()
        fit_seconds = perf_counter() - start
        start = perf_counter()
        posterior = model.posterior(test_X)
        posterior_seconds = perf_counter() - start
        prediction = posterior.mean
        rmse = torch.sqrt(torch.mean((prediction - test_Y).square())).item()
        results.append(
            BenchmarkResult(
                model=name,
                rmse=rmse,
                mean_posterior_std=posterior.variance.sqrt().mean().item(),
                fit_seconds=fit_seconds,
                posterior_seconds=posterior_seconds,
            )
        )
    return results


if __name__ == "__main__":
    for result in run_benchmark():
        print(
            f"{result.model:24s} RMSE={result.rmse:.4f} "
            f"posterior_std={result.mean_posterior_std:.4f} "
            f"fit={result.fit_seconds:.3f}s posterior={result.posterior_seconds:.3f}s"
        )
