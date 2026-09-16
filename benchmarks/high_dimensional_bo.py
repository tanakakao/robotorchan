"""Sequential BO benchmark for high-dimensional surrogate models."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

from robotorchan.models import (
    PCAGP,
    PLSGP,
    VAEGP,
    AdditiveMapSaasSingleTaskGP,
    AutoEncoderGP,
    HybridAutoEncoderGP,
    JointEncoderGP,
    JointVAEGP,
    RandomProjectionGP,
    SingleTaskGP,
    SupervisedAutoEncoderGP,
    SupervisedVAEGP,
)


@dataclass
class BOIterationResult:
    """One iteration of a sequential BO trajectory."""

    model: str
    iteration: int
    n_observations: int
    best_observed: float
    simple_regret: float
    selected_value: float
    seed: int = 0


@dataclass
class BOAggregateResult:
    """Repeated-seed summary for one model and BO iteration."""

    model: str
    iteration: int
    n_seeds: int
    simple_regret_mean: float
    simple_regret_std: float
    simple_regret_median: float
    simple_regret_q25: float
    simple_regret_q75: float
    best_observed_mean: float
    best_observed_std: float


@dataclass(frozen=True)
class BOModelSpec:
    """Model constructor and fitting policy for repeated BO fitting."""

    factory: Callable[[Tensor, Tensor], object]
    fit_policy: str = "mll"


def objective(X: Tensor) -> Tensor:
    """Evaluate the sparse five-coordinate synthetic objective."""
    if X.shape[-1] < 5:
        raise ValueError("input dimension must be at least 5")
    return (
        torch.sin(2.0 * math.pi * X[..., 0])
        + 0.8 * (X[..., 1] - 0.5).square()
        - 0.6 * X[..., 2]
        + 0.4 * X[..., 3] * X[..., 4]
    ).unsqueeze(-1)


def model_specs(
    latent_dim: int,
    *,
    neural_epochs: int = 20,
    include_extended: bool = False,
) -> dict[str, BOModelSpec]:
    """Return model constructors together with their repeated-fit policy."""
    specs = {
        "SingleTaskGP": BOModelSpec(lambda X, Y: SingleTaskGP(X, Y)),
        "PCAGP": BOModelSpec(lambda X, Y: PCAGP(X, Y, n_components=latent_dim)),
        "PLSGP": BOModelSpec(lambda X, Y: PLSGP(X, Y, n_components=latent_dim)),
        "RandomProjectionGP": BOModelSpec(
            lambda X, Y: RandomProjectionGP(X, Y, n_components=latent_dim, random_state=0)
        ),
    }
    if not include_extended:
        return specs

    neural = {
        "latent_dim": latent_dim,
        "hidden_dims": (32, 16),
        "epochs": neural_epochs,
        "random_state": 0,
    }
    joint = {"latent_dim": latent_dim, "hidden_dims": (32, 16), "random_state": 0}
    specs.update(
        {
            "AutoEncoderGP": BOModelSpec(lambda X, Y: AutoEncoderGP(X, Y, **neural)),
            "VAEGP": BOModelSpec(lambda X, Y: VAEGP(X, Y, **neural)),
            "SupervisedAutoEncoderGP": BOModelSpec(
                lambda X, Y: SupervisedAutoEncoderGP(X, Y, **neural)
            ),
            "SupervisedVAEGP": BOModelSpec(lambda X, Y: SupervisedVAEGP(X, Y, **neural)),
            "JointEncoderGP": BOModelSpec(lambda X, Y: JointEncoderGP(X, Y, **joint), "joint"),
            "HybridAutoEncoderGP": BOModelSpec(
                lambda X, Y: HybridAutoEncoderGP(X, Y, reconstruction_weight=0.1, **joint),
                "joint",
            ),
            "JointVAEGP": BOModelSpec(
                lambda X, Y: JointVAEGP(X, Y, beta=0.1, reconstruction_weight=0.1, **joint),
                "joint",
            ),
            "AdditiveMapSaasSingleTaskGP": BOModelSpec(
                lambda X, Y: AdditiveMapSaasSingleTaskGP(X, Y, num_taus=2)
            ),
        }
    )
    return specs


def model_factories(latent_dim: int) -> dict[str, Callable[[Tensor, Tensor], object]]:
    """Return the inexpensive core model set for backwards-compatible callers."""
    return {name: spec.factory for name, spec in model_specs(latent_dim).items()}


def fit_model(
    model: object,
    *,
    fit_policy: str,
    joint_steps: int,
    joint_learning_rate: float,
) -> None:
    """Fit a model using its standard MLL or joint training-loss contract."""
    if fit_policy == "mll":
        fit_gpytorch_mll(model.make_mll())
        return
    if fit_policy != "joint":
        raise ValueError(f"Unsupported fit policy: {fit_policy}")
    if joint_steps < 1:
        raise ValueError("joint_steps must be at least 1")
    if joint_learning_rate <= 0:
        raise ValueError("joint_learning_rate must be positive")

    optimizer = torch.optim.Adam(model.parameters(), lr=joint_learning_rate)
    for _ in range(joint_steps):
        optimizer.zero_grad()
        loss = model.training_loss()
        loss.backward()
        optimizer.step()


def select_from_pool(
    model: object,
    train_Y: Tensor,
    candidate_X: Tensor,
    *,
    mc_samples: int,
    seed: int,
) -> int:
    """Select the candidate with maximum qLogEI from a shared finite pool."""
    if candidate_X.shape[-2] == 0:
        raise ValueError("candidate_X must contain at least one candidate")
    if mc_samples < 1:
        raise ValueError("mc_samples must be at least 1")
    model.eval()
    model.likelihood.eval()
    sampler = SobolQMCNormalSampler(torch.Size([mc_samples]), seed=seed)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=float(train_Y.max()),
        sampler=sampler,
    )
    with torch.no_grad():
        values = acquisition(candidate_X.unsqueeze(-2))
    return int(values.argmax())


def run_model_bo(
    name: str,
    factory: Callable[[Tensor, Tensor], object],
    initial_X: Tensor,
    candidate_X: Tensor,
    candidate_Y: Tensor,
    *,
    n_iterations: int,
    mc_samples: int,
    seed: int,
    fit_policy: str = "mll",
    joint_steps: int = 30,
    joint_learning_rate: float = 1e-2,
) -> list[BOIterationResult]:
    """Run sequential pool-based BO for one model."""
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if n_iterations > candidate_X.shape[-2]:
        raise ValueError("n_iterations cannot exceed candidate pool size")

    train_X = initial_X.clone()
    train_Y = objective(train_X)
    pool_X = candidate_X.clone()
    pool_Y = candidate_Y.clone()
    optimum = float(torch.cat([train_Y, pool_Y]).max())
    results: list[BOIterationResult] = []

    for iteration in range(1, n_iterations + 1):
        model = factory(train_X, train_Y)
        fit_model(
            model,
            fit_policy=fit_policy,
            joint_steps=joint_steps,
            joint_learning_rate=joint_learning_rate,
        )
        index = select_from_pool(
            model,
            train_Y,
            pool_X,
            mc_samples=mc_samples,
            seed=seed + iteration,
        )
        new_X = pool_X[index : index + 1]
        new_Y = pool_Y[index : index + 1]
        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])
        keep = torch.ones(pool_X.shape[-2], dtype=torch.bool, device=pool_X.device)
        keep[index] = False
        pool_X = pool_X[keep]
        pool_Y = pool_Y[keep]
        best = float(train_Y.max())
        results.append(
            BOIterationResult(
                model=name,
                iteration=iteration,
                n_observations=train_X.shape[-2],
                best_observed=best,
                simple_regret=max(0.0, optimum - best),
                selected_value=float(new_Y.squeeze()),
                seed=seed,
            )
        )
    return results


def run_benchmark(
    *,
    n_initial: int = 12,
    n_candidates: int = 256,
    input_dim: int = 40,
    latent_dim: int = 5,
    n_iterations: int = 10,
    mc_samples: int = 64,
    seed: int = 0,
    include_extended: bool = False,
    neural_epochs: int = 20,
    joint_steps: int = 30,
    joint_learning_rate: float = 1e-2,
) -> list[BOIterationResult]:
    """Run comparable sequential BO trajectories on a shared candidate pool."""
    if input_dim < 5:
        raise ValueError("input_dim must be at least 5")
    if n_initial < 2:
        raise ValueError("n_initial must be at least 2")
    if n_candidates < n_iterations:
        raise ValueError("n_candidates must be at least n_iterations")
    generator = torch.Generator().manual_seed(seed)
    initial_X = torch.rand(n_initial, input_dim, generator=generator, dtype=torch.double)
    candidate_X = torch.rand(n_candidates, input_dim, generator=generator, dtype=torch.double)
    candidate_Y = objective(candidate_X)

    results: list[BOIterationResult] = []
    specs = model_specs(
        latent_dim,
        neural_epochs=neural_epochs,
        include_extended=include_extended,
    )
    for name, spec in specs.items():
        results.extend(
            run_model_bo(
                name,
                spec.factory,
                initial_X,
                candidate_X,
                candidate_Y,
                n_iterations=n_iterations,
                mc_samples=mc_samples,
                seed=seed,
                fit_policy=spec.fit_policy,
                joint_steps=joint_steps,
                joint_learning_rate=joint_learning_rate,
            )
        )
    return results


def run_repeated_benchmark(
    seeds: Sequence[int],
    **benchmark_kwargs: object,
) -> list[BOIterationResult]:
    """Run the benchmark for multiple independent random seeds."""
    if not seeds:
        raise ValueError("seeds must contain at least one value")
    results: list[BOIterationResult] = []
    for seed in seeds:
        results.extend(run_benchmark(seed=int(seed), **benchmark_kwargs))
    return results


def aggregate_results(results: Sequence[BOIterationResult]) -> list[BOAggregateResult]:
    """Aggregate repeated trajectories by model and iteration."""
    if not results:
        raise ValueError("results must contain at least one trajectory row")
    groups: dict[tuple[str, int], list[BOIterationResult]] = defaultdict(list)
    for result in results:
        groups[(result.model, result.iteration)].append(result)

    summaries: list[BOAggregateResult] = []
    for (model, iteration), rows in sorted(groups.items()):
        regrets = torch.tensor([row.simple_regret for row in rows], dtype=torch.double)
        best = torch.tensor([row.best_observed for row in rows], dtype=torch.double)
        summaries.append(
            BOAggregateResult(
                model=model,
                iteration=iteration,
                n_seeds=len(rows),
                simple_regret_mean=float(regrets.mean()),
                simple_regret_std=float(regrets.std(unbiased=False)),
                simple_regret_median=float(regrets.median()),
                simple_regret_q25=float(torch.quantile(regrets, 0.25)),
                simple_regret_q75=float(torch.quantile(regrets, 0.75)),
                best_observed_mean=float(best.mean()),
                best_observed_std=float(best.std(unbiased=False)),
            )
        )
    return summaries


def write_csv(results: Sequence[object], path: Path) -> None:
    """Write dataclass benchmark rows to CSV."""
    if not results:
        raise ValueError("results must contain at least one row")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(value: str) -> list[int]:
    """Parse a comma-separated seed list."""
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-initial", type=int, default=12)
    parser.add_argument("--n-candidates", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=40)
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--mc-samples", type=int, default=64)
    parser.add_argument("--seeds", type=parse_seeds, default=[0])
    parser.add_argument("--include-extended", action="store_true")
    parser.add_argument("--neural-epochs", type=int, default=20)
    parser.add_argument("--joint-steps", type=int, default=30)
    parser.add_argument("--joint-learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/high_dimensional_bo.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmark_results/high_dimensional_bo_summary.csv"),
    )
    args = parser.parse_args()
    results = run_repeated_benchmark(
        args.seeds,
        n_initial=args.n_initial,
        n_candidates=args.n_candidates,
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        n_iterations=args.n_iterations,
        mc_samples=args.mc_samples,
        include_extended=args.include_extended,
        neural_epochs=args.neural_epochs,
        joint_steps=args.joint_steps,
        joint_learning_rate=args.joint_learning_rate,
    )
    summaries = aggregate_results(results)
    write_csv(results, args.output)
    write_csv(summaries, args.summary_output)
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
