"""Predictive benchmark for high-dimensional multi-task surrogate models."""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from botorch.fit import fit_gpytorch_mll
from torch import Tensor

from robotorchan.models import (
    PCAMultiTaskGP,
    PLSMultiTaskGP,
    RandomProjectionMultiTaskGP,
    MultiTaskGP,
)


@dataclass
class BenchmarkResult:
    """One multi-task surrogate benchmark result."""

    model: str
    rmse: float
    train_seconds: float
    posterior_seconds: float


def make_synthetic_data(
    n_train_per_task: int,
    n_test_per_task: int,
    input_dim: int,
    *,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a long-format two-task problem with a shared low-D signal."""
    if input_dim < 5:
        raise ValueError("input_dim must be at least 5")
    generator = torch.Generator().manual_seed(seed)
    n_total = n_train_per_task + n_test_per_task
    data_X = torch.rand(n_total, input_dim, generator=generator, dtype=torch.double)
    shared = (
        torch.sin(2.0 * math.pi * data_X[:, 0])
        + 0.8 * (data_X[:, 1] - 0.5).square()
        - 0.6 * data_X[:, 2]
        + 0.4 * data_X[:, 3] * data_X[:, 4]
    )
    task0 = torch.zeros(n_total, 1, dtype=torch.double)
    task1 = torch.ones(n_total, 1, dtype=torch.double)
    X0 = torch.cat((data_X, task0), dim=-1)
    X1 = torch.cat((data_X, task1), dim=-1)
    Y0 = shared.unsqueeze(-1)
    Y1 = (0.7 * shared + 0.3 * data_X[:, 0] + 0.2).unsqueeze(-1)
    train_X = torch.cat((X0[:n_train_per_task], X1[:n_train_per_task]))
    train_Y = torch.cat((Y0[:n_train_per_task], Y1[:n_train_per_task]))
    test_X = torch.cat((X0[n_train_per_task:], X1[n_train_per_task:]))
    test_Y = torch.cat((Y0[n_train_per_task:], Y1[n_train_per_task:]))
    return train_X, train_Y, test_X, test_Y


def model_factories(latent_dim: int) -> dict[str, Callable[[Tensor, Tensor], object]]:
    """Return comparable long-format multi-task model constructors."""
    return {
        "MultiTaskGP": lambda X, Y: MultiTaskGP(X, Y, task_feature=-1),
        "PCAMultiTaskGP": lambda X, Y: PCAMultiTaskGP(
            X, Y, task_feature=-1, n_components=latent_dim
        ),
        "PLSMultiTaskGP": lambda X, Y: PLSMultiTaskGP(
            X, Y, task_feature=-1, n_components=latent_dim
        ),
        "RandomProjectionMultiTaskGP": lambda X, Y: RandomProjectionMultiTaskGP(
            X, Y, task_feature=-1, n_components=latent_dim, random_state=0
        ),
    }


def run_benchmark(
    *,
    n_train_per_task: int = 24,
    n_test_per_task: int = 48,
    input_dim: int = 40,
    latent_dim: int = 5,
    seed: int = 0,
) -> list[BenchmarkResult]:
    """Fit models and compare predictive accuracy plus runtime."""
    train_X, train_Y, test_X, test_Y = make_synthetic_data(
        n_train_per_task,
        n_test_per_task,
        input_dim,
        seed=seed,
    )
    results: list[BenchmarkResult] = []
    for name, factory in model_factories(latent_dim).items():
        start = time.perf_counter()
        model = factory(train_X, train_Y)
        fit_gpytorch_mll(model.make_mll())
        train_seconds = time.perf_counter() - start
        model.eval()
        model.likelihood.eval()
        start = time.perf_counter()
        with torch.no_grad():
            mean = model.posterior(test_X).mean
        posterior_seconds = time.perf_counter() - start
        rmse = torch.sqrt(torch.mean((mean - test_Y).square()))
        results.append(
            BenchmarkResult(
                model=name,
                rmse=float(rmse),
                train_seconds=train_seconds,
                posterior_seconds=posterior_seconds,
            )
        )
    return results


def write_csv(results: list[BenchmarkResult], path: Path) -> None:
    """Write benchmark results to CSV."""
    if not results:
        raise ValueError("results must contain at least one row")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train-per-task", type=int, default=24)
    parser.add_argument("--n-test-per-task", type=int, default=48)
    parser.add_argument("--input-dim", type=int, default=40)
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/high_dimensional_multitask.csv"),
    )
    args = parser.parse_args()
    results = run_benchmark(
        n_train_per_task=args.n_train_per_task,
        n_test_per_task=args.n_test_per_task,
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
    write_csv(results, args.output)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
