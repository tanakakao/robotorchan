"""Benchmark continuous high-dimensional acquisition-function search strategies."""

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

from robotorchan.models import SingleTaskGP
from robotorchan.optim import (
    OriginalSpaceStrategy,
    RandomSearchStrategy,
    REMBOStrategy,
    SearchStrategy,
)

_SEARCH_SEED_OFFSET = 1_000_003
_EMBEDDING_SEED_OFFSET = 2_000_033
_MAX_TORCH_SEED = 2**63 - 1


@dataclass(frozen=True)
class AcqfOptimizationResult:
    """One strategy run for a fixed fitted acquisition function."""

    strategy: str
    input_dim: int
    seed: int
    acquisition_value: float
    objective_value: float
    simple_regret: float
    optimization_time: float


@dataclass(frozen=True)
class AcqfOptimizationAggregateResult:
    """Repeated-seed summary for one strategy and input dimension."""

    strategy: str
    input_dim: int
    n_seeds: int
    acquisition_value_mean: float
    acquisition_value_std: float
    simple_regret_mean: float
    simple_regret_std: float
    optimization_time_mean: float
    optimization_time_std: float


def objective(X: Tensor) -> Tensor:
    """Evaluate a sparse synthetic objective with optimum zero at X=0.75."""
    if X.shape[-1] < 5:
        raise ValueError("input dimension must be at least 5")
    active = X[..., :5]
    return -((active - 0.75).square().sum(dim=-1, keepdim=True))


def _search_seed(problem_seed: int) -> int:
    """Derive a deterministic random-search seed independent from the problem RNG."""
    return (problem_seed + _SEARCH_SEED_OFFSET) % _MAX_TORCH_SEED


def _embedding_seed(problem_seed: int) -> int:
    """Derive a deterministic embedding seed independent from other RNG streams."""
    return (problem_seed + _EMBEDDING_SEED_OFFSET) % _MAX_TORCH_SEED


def make_problem(
    input_dim: int,
    *,
    n_train: int,
    seed: int,
) -> tuple[SingleTaskGP, Tensor, Tensor]:
    """Create and fit one surrogate shared by all search strategies."""
    if input_dim < 5:
        raise ValueError("input_dim must be at least 5")
    if n_train < 2:
        raise ValueError("n_train must be at least 2")

    generator = torch.Generator().manual_seed(seed)
    train_X = torch.rand(n_train, input_dim, dtype=torch.double, generator=generator)
    train_Y = objective(train_X)
    model = SingleTaskGP(train_X, train_Y)
    fit_gpytorch_mll(model.make_mll())
    model.eval()
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    optimum_X = torch.full((1, input_dim), 0.75, dtype=torch.double)
    return model, bounds, optimum_X


def strategy_factories(
    bounds: Tensor,
    *,
    embedding_dim: int,
    random_samples: int,
    num_restarts: int,
    raw_samples: int,
    seed: int,
) -> dict[str, SearchStrategy]:
    """Build strategies with shared public bounds and independent RNG streams."""
    if embedding_dim < 1 or embedding_dim > bounds.shape[-1]:
        raise ValueError("embedding_dim must be between 1 and input_dim")
    return {
        "OriginalSpace": OriginalSpaceStrategy(
            bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        ),
        "RandomSearch": RandomSearchStrategy(
            bounds,
            num_samples=random_samples,
            seed=_search_seed(seed),
        ),
        "REMBO": REMBOStrategy(
            bounds,
            embedding_dim=embedding_dim,
            seed=_embedding_seed(seed),
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        ),
    }


def run_dimension(
    input_dim: int,
    *,
    n_train: int = 24,
    embedding_dim: int = 5,
    random_samples: int = 4096,
    num_restarts: int = 10,
    raw_samples: int = 512,
    seed: int = 0,
) -> list[AcqfOptimizationResult]:
    """Compare strategies against the same fitted model and acquisition function."""
    model, bounds, optimum_X = make_problem(input_dim, n_train=n_train, seed=seed)
    acquisition = PosteriorMean(model)
    optimum = float(objective(optimum_X).squeeze())
    results: list[AcqfOptimizationResult] = []

    for name, strategy in strategy_factories(
        bounds,
        embedding_dim=embedding_dim,
        random_samples=random_samples,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        seed=seed,
    ).items():
        start = perf_counter()
        search_result = strategy.optimize(acquisition, q=1)
        elapsed = perf_counter() - start
        candidate = search_result.candidates
        with torch.no_grad():
            acq_value = float(acquisition(candidate.unsqueeze(-2)).squeeze())
            objective_value = float(objective(candidate).squeeze())
        results.append(
            AcqfOptimizationResult(
                strategy=name,
                input_dim=input_dim,
                seed=seed,
                acquisition_value=acq_value,
                objective_value=objective_value,
                simple_regret=max(0.0, optimum - objective_value),
                optimization_time=elapsed,
            )
        )
    return results


def run_benchmark(
    input_dims: Sequence[int],
    seeds: Sequence[int],
    **kwargs: object,
) -> list[AcqfOptimizationResult]:
    """Run all requested dimensions and independent seeds."""
    if not input_dims:
        raise ValueError("input_dims must contain at least one value")
    if not seeds:
        raise ValueError("seeds must contain at least one value")
    results: list[AcqfOptimizationResult] = []
    for input_dim in input_dims:
        for seed in seeds:
            results.extend(run_dimension(int(input_dim), seed=int(seed), **kwargs))
    return results


def aggregate_results(
    results: Sequence[AcqfOptimizationResult],
) -> list[AcqfOptimizationAggregateResult]:
    """Aggregate repeated runs by strategy and input dimension."""
    if not results:
        raise ValueError("results must contain at least one row")
    groups: dict[tuple[str, int], list[AcqfOptimizationResult]] = defaultdict(list)
    for result in results:
        groups[(result.strategy, result.input_dim)].append(result)

    summaries: list[AcqfOptimizationAggregateResult] = []
    for (strategy, input_dim), rows in sorted(groups.items()):
        acq = torch.tensor([row.acquisition_value for row in rows], dtype=torch.double)
        regret = torch.tensor([row.simple_regret for row in rows], dtype=torch.double)
        timing = torch.tensor([row.optimization_time for row in rows], dtype=torch.double)
        summaries.append(
            AcqfOptimizationAggregateResult(
                strategy=strategy,
                input_dim=input_dim,
                n_seeds=len(rows),
                acquisition_value_mean=float(acq.mean()),
                acquisition_value_std=float(acq.std(unbiased=False)),
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
    """Parse a comma-separated integer list."""
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dims", type=parse_int_list, default=[20, 50, 100, 200])
    parser.add_argument("--seeds", type=parse_int_list, default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--embedding-dim", type=int, default=5)
    parser.add_argument("--random-samples", type=int, default=4096)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--raw-samples", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/high_dimensional_acqf_optimization.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmark_results/high_dimensional_acqf_optimization_summary.csv"),
    )
    args = parser.parse_args()
    results = run_benchmark(
        args.input_dims,
        args.seeds,
        n_train=args.n_train,
        embedding_dim=args.embedding_dim,
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
