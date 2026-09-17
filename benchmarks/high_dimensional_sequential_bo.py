"""Benchmark sequential continuous BO search strategies in high dimensions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.fit import fit_gpytorch_mll
from torch import Tensor

from robotorchan.models import PCAGP, RandomProjectionGP, SingleTaskGP
from robotorchan.optim import (
    LatentSpaceStrategy,
    OriginalSpaceStrategy,
    PCAReconstruction,
    RandomProjectionReconstruction,
    RandomSearchStrategy,
)


@dataclass(frozen=True)
class SequentialBOResult:
    """One BO iteration for one strategy and seed."""

    strategy: str
    input_dim: int
    seed: int
    iteration: int
    objective_value: float
    best_observed: float
    simple_regret: float
    optimization_time: float


@dataclass(frozen=True)
class SequentialBOAggregateResult:
    """Repeated-seed summary at one BO iteration."""

    strategy: str
    input_dim: int
    iteration: int
    n_seeds: int
    best_observed_mean: float
    best_observed_std: float
    simple_regret_mean: float
    simple_regret_std: float
    optimization_time_mean: float
    optimization_time_std: float


def objective(X: Tensor) -> Tensor:
    """Sparse synthetic objective with optimum zero at active coordinates 0.75."""
    if X.shape[-1] < 5:
        raise ValueError("input dimension must be at least 5")
    return -((X[..., :5] - 0.75).square().sum(dim=-1, keepdim=True))


def make_initial_data(input_dim: int, *, n_train: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """Generate common initial observations without sharing search RNG streams."""
    if input_dim < 5:
        raise ValueError("input_dim must be at least 5")
    if n_train < 2:
        raise ValueError("n_train must be at least 2")
    generator = torch.Generator().manual_seed(seed)
    train_X = torch.rand(n_train, input_dim, dtype=torch.double, generator=generator)
    train_Y = objective(train_X)
    bounds = torch.stack(
        [torch.zeros(input_dim, dtype=torch.double), torch.ones(input_dim, dtype=torch.double)]
    )
    return train_X, train_Y, bounds


def _fit_model(strategy: str, train_X: Tensor, train_Y: Tensor, latent_dim: int):
    if strategy == "LatentPCA":
        model = PCAGP(train_X, train_Y, n_components=latent_dim)
    elif strategy == "LatentRandomProjection":
        model = RandomProjectionGP(train_X, train_Y, n_components=latent_dim, random_state=17)
    else:
        model = SingleTaskGP(train_X, train_Y)
    fit_gpytorch_mll(model.make_mll())
    model.eval()
    return model


def _make_strategy(
    name: str,
    model,
    bounds: Tensor,
    *,
    random_samples: int,
    num_restarts: int,
    raw_samples: int,
    search_seed: int,
):
    if name == "OriginalSpace":
        return OriginalSpaceStrategy(bounds, num_restarts=num_restarts, raw_samples=raw_samples)
    if name == "RandomSearch":
        return RandomSearchStrategy(bounds, num_samples=random_samples, seed=search_seed)
    if name == "LatentPCA":
        return LatentSpaceStrategy(
            bounds,
            PCAReconstruction(model.input_reducer),
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    if name == "LatentRandomProjection":
        return LatentSpaceStrategy(
            bounds,
            RandomProjectionReconstruction(model.input_reducer),
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    raise ValueError(f"Unknown strategy: {name}")


def run_strategy(
    name: str,
    input_dim: int,
    *,
    n_train: int = 24,
    n_iterations: int = 5,
    latent_dim: int = 5,
    random_samples: int = 4096,
    num_restarts: int = 10,
    raw_samples: int = 512,
    seed: int = 0,
) -> list[SequentialBOResult]:
    """Run a sequential BO trajectory from common initial observations."""
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if latent_dim < 1 or latent_dim > input_dim:
        raise ValueError("latent_dim must be between 1 and input_dim")

    train_X, train_Y, bounds = make_initial_data(input_dim, n_train=n_train, seed=seed)
    optimum = 0.0
    results: list[SequentialBOResult] = []

    for iteration in range(1, n_iterations + 1):
        model = _fit_model(name, train_X, train_Y, latent_dim)
        acquisition = PosteriorMean(model)
        # Deliberately separate search RNG from initial-data RNG and BO iterations.
        search_seed = seed * 100_000 + iteration * 1_009 + 73
        strategy = _make_strategy(
            name,
            model,
            bounds,
            random_samples=random_samples,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            search_seed=search_seed,
        )
        start = perf_counter()
        search_result = strategy.optimize(acquisition, q=1)
        elapsed = perf_counter() - start
        candidate = search_result.candidates.detach()
        with torch.no_grad():
            candidate_Y = objective(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, candidate_Y], dim=0)
        best_observed = float(train_Y.max())
        objective_value = float(candidate_Y.squeeze())
        results.append(
            SequentialBOResult(
                strategy=name,
                input_dim=input_dim,
                seed=seed,
                iteration=iteration,
                objective_value=objective_value,
                best_observed=best_observed,
                simple_regret=max(0.0, optimum - best_observed),
                optimization_time=elapsed,
            )
        )
    return results


def run_dimension(input_dim: int, *, seed: int = 0, **kwargs: object) -> list[SequentialBOResult]:
    """Compare all strategies from identical initial observations."""
    results: list[SequentialBOResult] = []
    for name in ("OriginalSpace", "RandomSearch", "LatentPCA", "LatentRandomProjection"):
        results.extend(run_strategy(name, input_dim, seed=seed, **kwargs))
    return results


def run_benchmark(
    input_dims: Sequence[int], seeds: Sequence[int], **kwargs: object
) -> list[SequentialBOResult]:
    """Run dimensions and repeated independent seeds."""
    if not input_dims:
        raise ValueError("input_dims must contain at least one value")
    if not seeds:
        raise ValueError("seeds must contain at least one value")
    results: list[SequentialBOResult] = []
    for input_dim in input_dims:
        for seed in seeds:
            results.extend(run_dimension(int(input_dim), seed=int(seed), **kwargs))
    return results


def aggregate_results(results: Sequence[SequentialBOResult]) -> list[SequentialBOAggregateResult]:
    """Aggregate repeated runs by strategy, dimension, and BO iteration."""
    if not results:
        raise ValueError("results must contain at least one row")
    groups: dict[tuple[str, int, int], list[SequentialBOResult]] = defaultdict(list)
    for result in results:
        groups[(result.strategy, result.input_dim, result.iteration)].append(result)

    summaries: list[SequentialBOAggregateResult] = []
    for (strategy, input_dim, iteration), rows in sorted(groups.items()):
        best = torch.tensor([row.best_observed for row in rows], dtype=torch.double)
        regret = torch.tensor([row.simple_regret for row in rows], dtype=torch.double)
        timing = torch.tensor([row.optimization_time for row in rows], dtype=torch.double)
        summaries.append(
            SequentialBOAggregateResult(
                strategy=strategy,
                input_dim=input_dim,
                iteration=iteration,
                n_seeds=len(rows),
                best_observed_mean=float(best.mean()),
                best_observed_std=float(best.std(unbiased=False)),
                simple_regret_mean=float(regret.mean()),
                simple_regret_std=float(regret.std(unbiased=False)),
                optimization_time_mean=float(timing.mean()),
                optimization_time_std=float(timing.std(unbiased=False)),
            )
        )
    return summaries


def write_csv(results: Sequence[object], path: Path) -> None:
    """Write dataclass rows to CSV."""
    if not results:
        raise ValueError("results must contain at least one row")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dims", type=parse_int_list, default=[20, 50, 100, 200])
    parser.add_argument("--seeds", type=parse_int_list, default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--random-samples", type=int, default=4096)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--raw-samples", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/high_dimensional_sequential_bo.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmark_results/high_dimensional_sequential_bo_summary.csv"),
    )
    args = parser.parse_args()
    results = run_benchmark(
        args.input_dims,
        args.seeds,
        n_train=args.n_train,
        n_iterations=args.n_iterations,
        latent_dim=args.latent_dim,
        random_samples=args.random_samples,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
    )
    summaries = aggregate_results(results)
    write_csv(results, args.output)
    write_csv(summaries, args.summary_output)
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
