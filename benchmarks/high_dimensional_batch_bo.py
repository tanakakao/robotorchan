"""Benchmark joint q-batch search strategies in high-dimensional BO."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from torch import Tensor

from robotorchan.models import SingleTaskGP
from robotorchan.optim import (
    BAxUSState,
    BAxUSStrategy,
    BAxUSThompsonSamplingStrategy,
    HeSBOStrategy,
    OriginalSpaceStrategy,
    RandomSearchStrategy,
    REMBOStrategy,
    TuRBOState,
    TuRBOStrategy,
)

STRATEGY_NAMES = (
    "OriginalSpace",
    "RandomSearch",
    "REMBO",
    "HeSBO",
    "TuRBO",
    "BAxUS",
    "BAxUSTS",
)


@dataclass(frozen=True)
class BatchBOResult:
    strategy: str
    input_dim: int
    seed: int
    iteration: int
    q: int
    batch_best: float
    best_observed: float
    simple_regret: float
    optimization_time: float


def objective(X: Tensor) -> Tensor:
    """Synthetic maximization objective with five active dimensions."""
    if X.shape[-1] < 5:
        raise ValueError("input dimension must be at least 5")
    return -((X[..., :5] - 0.75).square().sum(dim=-1, keepdim=True))


def make_initial_data(input_dim: int, *, n_train: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
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


def _fit_model(train_X: Tensor, train_Y: Tensor) -> SingleTaskGP:
    model = SingleTaskGP(train_X, train_Y)
    fit_gpytorch_mll(model.make_mll())
    model.eval()
    return model


def _make_acquisition(model: SingleTaskGP, train_Y: Tensor) -> qLogExpectedImprovement:
    return qLogExpectedImprovement(model=model, best_f=train_Y.max())


def _initial_incumbent(train_X: Tensor, train_Y: Tensor) -> tuple[Tensor, float]:
    best_index = train_Y.reshape(-1).argmax()
    return train_X[best_index].detach().clone(), float(train_Y.reshape(-1)[best_index])


def _make_strategy(
    name: str,
    bounds: Tensor,
    train_X: Tensor,
    train_Y: Tensor,
    *,
    random_samples: int,
    embedding_dim: int,
    ts_candidates: int | None,
    num_restarts: int,
    raw_samples: int,
    search_seed: int,
    eval_budget: int,
):
    if name == "OriginalSpace":
        return OriginalSpaceStrategy(bounds, num_restarts=num_restarts, raw_samples=raw_samples)
    if name == "RandomSearch":
        return RandomSearchStrategy(bounds, num_samples=random_samples, seed=search_seed)
    if name == "REMBO":
        return REMBOStrategy(
            bounds,
            embedding_dim=embedding_dim,
            seed=search_seed,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    if name == "HeSBO":
        return HeSBOStrategy(
            bounds,
            embedding_dim=embedding_dim,
            seed=search_seed,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    center, best_value = _initial_incumbent(train_X, train_Y)
    if name == "TuRBO":
        return TuRBOStrategy(
            bounds,
            center=center,
            state=TuRBOState(best_value=best_value),
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    if name in {"BAxUS", "BAxUSTS"}:
        state = BAxUSState(dim=bounds.shape[-1], eval_budget=eval_budget, best_value=best_value)
        if name == "BAxUSTS":
            return BAxUSThompsonSamplingStrategy(
                bounds,
                state=state,
                seed=search_seed,
                n_candidates=ts_candidates,
            )
        return BAxUSStrategy(
            bounds,
            state=state,
            seed=search_seed,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    raise ValueError(f"Unknown strategy: {name}")


def _update_stateful_strategy(
    strategy, candidates: Tensor, values: Tensor, *, search_metadata: dict
) -> None:
    if isinstance(strategy, TuRBOStrategy):
        strategy.update_state(values, candidates=candidates)
    elif isinstance(strategy, BAxUSStrategy):
        state = strategy.update_state(
            values,
            target_candidates=search_metadata["target_candidates"],
        )
        if state.restart_triggered and strategy.target_dim < strategy.input_dim:
            strategy.expand_subspace()


def run_strategy(
    name: str,
    input_dim: int,
    *,
    q: int = 3,
    n_train: int = 24,
    n_iterations: int = 5,
    random_samples: int = 1024,
    embedding_dim: int = 5,
    ts_candidates: int | None = None,
    num_restarts: int = 10,
    raw_samples: int = 512,
    seed: int = 0,
) -> list[BatchBOResult]:
    if name not in STRATEGY_NAMES:
        raise ValueError(f"Unknown strategy: {name}")
    if q < 2:
        raise ValueError("q must be at least 2 for the batch benchmark")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if random_samples < 1:
        raise ValueError("random_samples must be at least 1")
    if embedding_dim < 1 or embedding_dim > input_dim:
        raise ValueError("embedding_dim must be between 1 and input_dim")
    if ts_candidates is not None and ts_candidates < q:
        raise ValueError("ts_candidates must be at least q")

    train_X, train_Y, bounds = make_initial_data(input_dim, n_train=n_train, seed=seed)
    strategy_seed = seed * 100_000 + 73
    strategy = _make_strategy(
        name,
        bounds,
        train_X,
        train_Y,
        random_samples=random_samples,
        embedding_dim=embedding_dim,
        ts_candidates=ts_candidates,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        search_seed=strategy_seed,
        eval_budget=n_iterations * q,
    )
    results: list[BatchBOResult] = []
    for iteration in range(1, n_iterations + 1):
        model = _fit_model(train_X, train_Y)
        acquisition = _make_acquisition(model, train_Y)
        if name == "RandomSearch":
            search_seed = seed * 100_000 + iteration * 1_009 + 73
            strategy = RandomSearchStrategy(bounds, num_samples=random_samples, seed=search_seed)
        start = perf_counter()
        search_result = strategy.optimize(acquisition, q=q)
        elapsed = perf_counter() - start
        candidates = search_result.candidates.detach()
        with torch.no_grad():
            candidate_Y = objective(candidates)
        _update_stateful_strategy(
            strategy,
            candidates,
            candidate_Y,
            search_metadata=search_result.metadata,
        )
        train_X = torch.cat([train_X, candidates], dim=0)
        train_Y = torch.cat([train_Y, candidate_Y], dim=0)
        best_observed = float(train_Y.max())
        results.append(
            BatchBOResult(
                strategy=name,
                input_dim=input_dim,
                seed=seed,
                iteration=iteration,
                q=q,
                batch_best=float(candidate_Y.max()),
                best_observed=best_observed,
                simple_regret=max(0.0, -best_observed),
                optimization_time=elapsed,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default="RandomSearch")
    parser.add_argument("--input-dim", type=int, default=50)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--random-samples", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=5)
    parser.add_argument("--ts-candidates", type=int, default=None)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--raw-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = run_strategy(
        args.strategy,
        args.input_dim,
        q=args.q,
        n_train=args.n_train,
        n_iterations=args.n_iterations,
        random_samples=args.random_samples,
        embedding_dim=args.embedding_dim,
        ts_candidates=args.ts_candidates,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
        seed=args.seed,
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
