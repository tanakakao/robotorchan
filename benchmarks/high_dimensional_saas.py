"""Sparse-prior baselines for the high-dimensional input benchmark."""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from torch import Tensor

from robotorchan.models import (
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
    SaasFullyBayesianSingleTaskGP,
)


@dataclass
class SaasBenchmarkResult:
    """One sparse-prior surrogate benchmark result."""

    model: str
    rmse: float
    nll: float
    train_seconds: float
    posterior_seconds: float


def make_sparse_synthetic_data(
    n_train: int,
    n_test: int,
    input_dim: int,
    *,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a high-D problem whose response depends on five coordinates."""
    if input_dim < 5:
        raise ValueError("input_dim must be at least 5")
    generator = torch.Generator().manual_seed(seed)
    X = torch.rand(n_train + n_test, input_dim, generator=generator, dtype=torch.double)
    y = (
        torch.sin(2.0 * math.pi * X[:, 0])
        + 0.8 * (X[:, 1] - 0.5).square()
        - 0.6 * X[:, 2]
        + 0.4 * X[:, 3] * X[:, 4]
    ).unsqueeze(-1)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def gaussian_nll(mean: Tensor, variance: Tensor, target: Tensor) -> Tensor:
    """Return mean Gaussian negative log likelihood."""
    variance = variance.clamp_min(1e-10)
    return 0.5 * (torch.log(2.0 * torch.pi * variance) + (target - mean).square() / variance).mean()


def map_saas_factories(num_taus: int = 4) -> dict[str, object]:
    """Return inexpensive MAP-SAAS constructors."""
    return {
        "AdditiveMapSaasSingleTaskGP": lambda X, Y: AdditiveMapSaasSingleTaskGP(
            X, Y, num_taus=num_taus
        ),
        "EnsembleMapSaasSingleTaskGP": lambda X, Y: EnsembleMapSaasSingleTaskGP(
            X, Y, num_taus=num_taus
        ),
    }


def _evaluate(
    name: str,
    model: object,
    test_X: Tensor,
    test_Y: Tensor,
    train_seconds: float,
) -> SaasBenchmarkResult:
    model.eval()
    model.likelihood.eval()
    start = time.perf_counter()
    with torch.no_grad():
        posterior = model.posterior(test_X)
        mean = posterior.mean
        variance = posterior.variance
    posterior_seconds = time.perf_counter() - start
    return SaasBenchmarkResult(
        model=name,
        rmse=float(torch.sqrt(torch.mean((mean - test_Y).square()))),
        nll=float(gaussian_nll(mean, variance, test_Y)),
        train_seconds=train_seconds,
        posterior_seconds=posterior_seconds,
    )


def run_map_saas_benchmark(
    *,
    n_train: int = 64,
    n_test: int = 128,
    input_dim: int = 40,
    num_taus: int = 4,
    seed: int = 0,
) -> list[SaasBenchmarkResult]:
    """Benchmark MAP-SAAS models using ordinary exact-GP fitting."""
    train_X, train_Y, test_X, test_Y = make_sparse_synthetic_data(
        n_train, n_test, input_dim, seed=seed
    )
    results = []
    for name, factory in map_saas_factories(num_taus).items():
        start = time.perf_counter()
        model = factory(train_X, train_Y)
        fit_gpytorch_mll(model.make_mll())
        train_seconds = time.perf_counter() - start
        results.append(_evaluate(name, model, test_X, test_Y, train_seconds))
    return results


def run_fully_bayesian_saas_benchmark(
    *,
    n_train: int = 64,
    n_test: int = 128,
    input_dim: int = 40,
    warmup_steps: int = 128,
    num_samples: int = 64,
    thinning: int = 4,
    seed: int = 0,
) -> SaasBenchmarkResult:
    """Benchmark fully Bayesian SAAS with an explicitly controlled NUTS budget."""
    train_X, train_Y, test_X, test_Y = make_sparse_synthetic_data(
        n_train, n_test, input_dim, seed=seed
    )
    model = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
    start = time.perf_counter()
    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=warmup_steps,
        num_samples=num_samples,
        thinning=thinning,
        disable_progbar=True,
    )
    train_seconds = time.perf_counter() - start
    return _evaluate("SaasFullyBayesianSingleTaskGP", model, test_X, test_Y, train_seconds)


def write_csv(results: list[SaasBenchmarkResult], path: Path) -> None:
    """Write benchmark results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=40)
    parser.add_argument("--num-taus", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fully-bayesian", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--thinning", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/high_dimensional_saas.csv"),
    )
    args = parser.parse_args()
    results = run_map_saas_benchmark(
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        num_taus=args.num_taus,
        seed=args.seed,
    )
    if args.fully_bayesian:
        results.append(
            run_fully_bayesian_saas_benchmark(
                n_train=args.n_train,
                n_test=args.n_test,
                input_dim=args.input_dim,
                warmup_steps=args.warmup_steps,
                num_samples=args.num_samples,
                thinning=args.thinning,
                seed=args.seed,
            )
        )
    write_csv(results, args.output)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
