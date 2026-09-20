"""Predictive benchmark for expressive single-output surrogate models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from botorch.fit import fit_gpytorch_mll
from torch import Tensor

from robotorchan.models import (
    InfiniteWidthBNNGP,
    JointEncoderGP,
    SingleTaskGP,
    SpectralMixtureGP,
)
from robotorchan.models.deep_gp import SingleTaskDeepGP

MODEL_NAMES = ("SingleTaskGP", "DKL", "DeepGP", "InfiniteWidthBNNGP", "SpectralMixtureGP")


@dataclass(frozen=True)
class PredictiveBenchmarkResult:
    model: str
    seed: int
    rmse: float
    gaussian_nll: float
    coverage_95: float
    fit_time: float
    posterior_time: float


def objective(X: Tensor) -> Tensor:
    smooth = torch.sin(2.0 * torch.pi * 3.0 * X[..., :1])
    interaction = 0.6 * torch.sin(2.0 * torch.pi * X[..., 1:2] * X[..., 2:3])
    trend = 0.25 * X[..., 2:3]
    return smooth + interaction + trend


def make_data(
    *,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if n_train < 4 or n_test < 1:
        raise ValueError("n_train must be at least 4 and n_test must be positive.")
    generator = torch.Generator().manual_seed(seed)
    train_X = torch.rand(n_train, 3, dtype=torch.double, generator=generator)
    test_X = torch.rand(n_test, 3, dtype=torch.double, generator=generator)
    return train_X, objective(train_X), test_X, objective(test_X)


def make_model(name: str, train_X: Tensor, train_Y: Tensor):
    if name == "SingleTaskGP":
        return SingleTaskGP(train_X, train_Y)
    if name == "DKL":
        return JointEncoderGP(
            train_X,
            train_Y,
            latent_dim=3,
            hidden_dims=(12, 8),
            random_state=17,
        )
    if name == "DeepGP":
        return SingleTaskDeepGP(
            train_X,
            train_Y,
            hidden_dims=(4,),
            num_inducing=min(16, train_X.shape[0]),
            posterior_samples=64,
            random_state=17,
        )
    if name == "InfiniteWidthBNNGP":
        return InfiniteWidthBNNGP(train_X, train_Y, depth=3)
    if name == "SpectralMixtureGP":
        return SpectralMixtureGP(train_X, train_Y, num_mixtures=4)
    raise ValueError(f"Unknown model: {name}")


def fit_model(model, *, steps: int) -> None:
    if isinstance(model, SingleTaskDeepGP):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        model.train()
        for _ in range(steps):
            optimizer.zero_grad()
            loss = model.training_loss(num_likelihood_samples=8)
            loss.backward()
            optimizer.step()
        model.eval()
        return
    fit_gpytorch_mll(model.make_mll())
    model.eval()
    model.likelihood.eval()


def evaluate_model(
    name: str,
    *,
    n_train: int = 40,
    n_test: int = 256,
    seed: int = 0,
    deep_gp_steps: int = 75,
) -> PredictiveBenchmarkResult:
    train_X, train_Y, test_X, test_Y = make_data(
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    model = make_model(name, train_X, train_Y)

    start = perf_counter()
    fit_model(model, steps=deep_gp_steps)
    fit_time = perf_counter() - start

    start = perf_counter()
    with torch.no_grad():
        posterior = model.posterior(test_X)
        mean = posterior.mean
        variance = posterior.variance.clamp_min(1e-10)
    posterior_time = perf_counter() - start

    error = test_Y - mean
    rmse = error.square().mean().sqrt()
    gaussian_nll = 0.5 * (
        torch.log(2.0 * torch.pi * variance) + error.square() / variance
    ).mean()
    standard_deviation = variance.sqrt()
    covered = (error.abs() <= 1.96 * standard_deviation).double().mean()

    return PredictiveBenchmarkResult(
        model=name,
        seed=seed,
        rmse=float(rmse),
        gaussian_nll=float(gaussian_nll),
        coverage_95=float(covered),
        fit_time=fit_time,
        posterior_time=posterior_time,
    )


def run_benchmark(
    *,
    seeds: tuple[int, ...],
    n_train: int,
    n_test: int,
    deep_gp_steps: int,
) -> list[PredictiveBenchmarkResult]:
    return [
        evaluate_model(
            name,
            n_train=n_train,
            n_test=n_test,
            seed=seed,
            deep_gp_steps=deep_gp_steps,
        )
        for seed in seeds
        for name in MODEL_NAMES
    ]


def write_csv(results: list[PredictiveBenchmarkResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--deep-gp-steps", type=int, default=75)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/expressive_predictive.csv"),
    )
    args = parser.parse_args()
    results = run_benchmark(
        seeds=tuple(args.seeds),
        n_train=args.n_train,
        n_test=args.n_test,
        deep_gp_steps=args.deep_gp_steps,
    )
    write_csv(results, args.output)


if __name__ == "__main__":
    main()
