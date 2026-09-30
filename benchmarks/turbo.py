"""Reproducible global-BO versus TuRBO benchmark on standard synthetic functions."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from torch import Tensor

from robotorchan.models import SingleTaskGP
from robotorchan.optim import OriginalSpaceStrategy, TuRBOState, TuRBOStrategy


@dataclass(frozen=True)
class TurboBenchmarkRow:
    """One optimization step from a reproducible benchmark trajectory."""

    function: str
    method: str
    input_dim: int
    seed: int
    iteration: int
    n_observations: int
    best_observed: float
    simple_regret: float
    trust_region_length: float | None


def ackley(X: Tensor) -> Tensor:
    """Return the negated Ackley objective on [0, 1]^d."""
    x = 10.0 * X - 5.0
    term1 = -20.0 * torch.exp(-0.2 * torch.sqrt(x.square().mean(dim=-1)))
    term2 = -torch.exp(torch.cos(2.0 * math.pi * x).mean(dim=-1))
    return -(term1 + term2 + 20.0 + math.e).unsqueeze(-1)


def rosenbrock(X: Tensor) -> Tensor:
    """Return the negated Rosenbrock objective on [0, 1]^d."""
    x = 4.0 * X - 2.0
    value = 100.0 * (x[..., 1:] - x[..., :-1].square()).square()
    value = value + (1.0 - x[..., :-1]).square()
    return -value.sum(dim=-1, keepdim=True)


def levy(X: Tensor) -> Tensor:
    """Return the negated Levy objective on [0, 1]^d."""
    x = 20.0 * X - 10.0
    w = 1.0 + (x - 1.0) / 4.0
    first = torch.sin(math.pi * w[..., 0]).square()
    middle = (w[..., :-1] - 1.0).square()
    middle *= 1.0 + 10.0 * torch.sin(math.pi * w[..., :-1] + 1.0).square()
    last = (w[..., -1] - 1.0).square()
    last *= 1.0 + torch.sin(2.0 * math.pi * w[..., -1]).square()
    return -(first + middle.sum(dim=-1) + last).unsqueeze(-1)


OBJECTIVES: dict[str, Callable[[Tensor], Tensor]] = {
    "ackley": ackley,
    "rosenbrock": rosenbrock,
    "levy": levy,
}


def _fit_model(train_X: Tensor, train_Y: Tensor) -> SingleTaskGP:
    model = SingleTaskGP(train_X, train_Y)
    fit_gpytorch_mll(model.make_mll())
    return model


def run_trajectory(
    function: str,
    method: str,
    *,
    input_dim: int,
    n_initial: int,
    n_iterations: int,
    seed: int,
    num_restarts: int = 4,
    raw_samples: int = 64,
) -> list[TurboBenchmarkRow]:
    """Run one global-BO or TuRBO trajectory from a shared Sobol initialization."""
    if function not in OBJECTIVES:
        raise ValueError(f"unknown function: {function}")
    if method not in {"global", "turbo"}:
        raise ValueError("method must be 'global' or 'turbo'.")
    if input_dim < 2:
        raise ValueError("input_dim must be at least 2.")
    if n_initial < 2:
        raise ValueError("n_initial must be at least 2.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1.")

    objective = OBJECTIVES[function]
    bounds = torch.stack(
        [torch.zeros(input_dim, dtype=torch.double), torch.ones(input_dim, dtype=torch.double)]
    )
    sobol = torch.quasirandom.SobolEngine(input_dim, scramble=True, seed=seed)
    train_X = sobol.draw(n_initial).to(dtype=torch.double)
    train_Y = objective(train_X)
    best_index = int(train_Y.squeeze(-1).argmax())
    strategy: OriginalSpaceStrategy | TuRBOStrategy
    if method == "global":
        strategy = OriginalSpaceStrategy(
            bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    else:
        best_value = float(train_Y[best_index].item())
        strategy = TuRBOStrategy(
            bounds,
            center=train_X[best_index],
            state=TuRBOState(
                dim=input_dim,
                best_value=best_value,
                observed_best_value=best_value,
            ),
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            seed=seed,
        )

    rows: list[TurboBenchmarkRow] = []
    for iteration in range(1, n_iterations + 1):
        model = _fit_model(train_X, train_Y)
        acquisition = LogExpectedImprovement(model, best_f=float(train_Y.max().item()))
        result = strategy.optimize(acquisition)
        new_X = result.candidates.detach()
        new_Y = objective(new_X)
        train_X = torch.cat([train_X, new_X], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)
        if isinstance(strategy, TuRBOStrategy):
            strategy.update_state(new_Y, candidates=new_X)
            if strategy.state.restart_triggered:
                strategy.restart()
            trust_length: float | None = strategy.state.length
        else:
            trust_length = None
        best = float(train_Y.max().item())
        rows.append(
            TurboBenchmarkRow(
                function=function,
                method=method,
                input_dim=input_dim,
                seed=seed,
                iteration=iteration,
                n_observations=train_X.shape[0],
                best_observed=best,
                simple_regret=max(0.0, -best),
                trust_region_length=trust_length,
            )
        )
    return rows


def run_benchmark(
    *,
    functions: Sequence[str] = ("ackley", "rosenbrock", "levy"),
    dimensions: Sequence[int] = (20, 50, 100),
    seeds: Sequence[int] = (0, 1, 2),
    n_initial: int = 20,
    n_iterations: int = 20,
    num_restarts: int = 4,
    raw_samples: int = 64,
) -> list[TurboBenchmarkRow]:
    """Run matched global-BO and TuRBO trajectories across benchmark settings."""
    rows: list[TurboBenchmarkRow] = []
    for function in functions:
        for input_dim in dimensions:
            for seed in seeds:
                for method in ("global", "turbo"):
                    rows.extend(
                        run_trajectory(
                            function,
                            method,
                            input_dim=input_dim,
                            n_initial=n_initial,
                            n_iterations=n_iterations,
                            seed=seed,
                            num_restarts=num_restarts,
                            raw_samples=raw_samples,
                        )
                    )
    return rows


def write_csv(rows: Sequence[TurboBenchmarkRow], path: Path) -> None:
    """Write benchmark trajectories to CSV."""
    if not rows:
        raise ValueError("rows must contain at least one benchmark result.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=_parse_ints, default=[20, 50, 100])
    parser.add_argument("--seeds", type=_parse_ints, default=[0, 1, 2])
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument("--raw-samples", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/turbo.csv"),
    )
    args = parser.parse_args()
    rows = run_benchmark(
        dimensions=args.dimensions,
        seeds=args.seeds,
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
    )
    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
