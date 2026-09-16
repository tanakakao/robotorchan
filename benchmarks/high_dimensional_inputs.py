"""Common benchmark for high-dimensional input surrogate models."""

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
    PCAGP,
    PLSGP,
    VAEGP,
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
class BenchmarkResult:
    """One surrogate benchmark result."""

    model: str
    rmse: float
    nll: float
    train_seconds: float
    posterior_seconds: float


def make_synthetic_data(
    n_train: int,
    n_test: int,
    input_dim: int,
    *,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a high-D regression problem with a low-D active subspace."""
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
    return 0.5 * (
        torch.log(2.0 * torch.pi * variance) + (target - mean).square() / variance
    ).mean()


def model_factories(
    latent_dim: int,
    neural_epochs: int,
) -> dict[str, Callable[[Tensor, Tensor], object]]:
    """Return frozen/pretrained benchmark model constructors."""
    neural = {"latent_dim": latent_dim, "hidden_dims": (32, 16), "epochs": neural_epochs}
    return {
        "SingleTaskGP": lambda X, Y: SingleTaskGP(X, Y),
        "PCAGP": lambda X, Y: PCAGP(X, Y, n_components=latent_dim),
        "PLSGP": lambda X, Y: PLSGP(X, Y, n_components=latent_dim),
        "RandomProjectionGP": lambda X, Y: RandomProjectionGP(
            X, Y, n_components=latent_dim, random_state=0
        ),
        "AutoEncoderGP": lambda X, Y: AutoEncoderGP(X, Y, random_state=0, **neural),
        "VAEGP": lambda X, Y: VAEGP(X, Y, random_state=0, **neural),
        "SupervisedAutoEncoderGP": lambda X, Y: SupervisedAutoEncoderGP(
            X, Y, random_state=0, **neural
        ),
        "SupervisedVAEGP": lambda X, Y: SupervisedVAEGP(X, Y, random_state=0, **neural),
    }


def joint_model_factories(
    latent_dim: int,
) -> dict[str, Callable[[Tensor, Tensor], object]]:
    """Return models whose representation is trained jointly with the GP."""
    neural = {"latent_dim": latent_dim, "hidden_dims": (32, 16), "random_state": 0}
    return {
        "JointEncoderGP": lambda X, Y: JointEncoderGP(X, Y, **neural),
        "HybridAutoEncoderGP": lambda X, Y: HybridAutoEncoderGP(
            X, Y, reconstruction_weight=0.1, **neural
        ),
        "JointVAEGP": lambda X, Y: JointVAEGP(
            X, Y, beta=0.1, reconstruction_weight=0.1, **neural
        ),
    }


def _fit_joint_model(model: object, steps: int, learning_rate: float) -> None:
    """Optimize a joint neural-GP model with its model-specific objective."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad()
        if isinstance(model, JointVAEGP):
            loss = model.joint_loss()
        elif isinstance(model, HybridAutoEncoderGP):
            loss = model.hybrid_loss()
        else:
            model.train()
            model.likelihood.train()
            output = model(model.raw_train_X)
            loss = -model.make_mll()(output, model.train_targets)
        loss.backward()
        optimizer.step()


def _evaluate_model(
    name: str,
    model: object,
    test_X: Tensor,
    test_Y: Tensor,
    train_seconds: float,
) -> BenchmarkResult:
    """Evaluate one fitted model with common predictive metrics."""
    model.eval()
    model.likelihood.eval()
    start = time.perf_counter()
    with torch.no_grad():
        posterior = model.posterior(test_X)
        mean = posterior.mean
        variance = posterior.variance
    posterior_seconds = time.perf_counter() - start
    rmse = torch.sqrt(torch.mean((mean - test_Y).square()))
    nll = gaussian_nll(mean, variance, test_Y)
    return BenchmarkResult(
        model=name,
        rmse=float(rmse),
        nll=float(nll),
        train_seconds=train_seconds,
        posterior_seconds=posterior_seconds,
    )


def run_benchmark(
    *,
    n_train: int = 64,
    n_test: int = 128,
    input_dim: int = 40,
    latent_dim: int = 5,
    neural_epochs: int = 50,
    joint_steps: int = 100,
    joint_learning_rate: float = 1e-2,
    seed: int = 0,
) -> list[BenchmarkResult]:
    """Fit frozen and joint models and return common predictive metrics."""
    train_X, train_Y, test_X, test_Y = make_synthetic_data(
        n_train, n_test, input_dim, seed=seed
    )
    results: list[BenchmarkResult] = []
    for name, factory in model_factories(latent_dim, neural_epochs).items():
        start = time.perf_counter()
        model = factory(train_X, train_Y)
        fit_gpytorch_mll(model.make_mll())
        train_seconds = time.perf_counter() - start
        results.append(_evaluate_model(name, model, test_X, test_Y, train_seconds))

    for name, factory in joint_model_factories(latent_dim).items():
        start = time.perf_counter()
        model = factory(train_X, train_Y)
        _fit_joint_model(model, joint_steps, joint_learning_rate)
        train_seconds = time.perf_counter() - start
        results.append(_evaluate_model(name, model, test_X, test_Y, train_seconds))
    return results


def write_csv(results: list[BenchmarkResult], path: Path) -> None:
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
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--neural-epochs", type=int, default=50)
    parser.add_argument("--joint-steps", type=int, default=100)
    parser.add_argument("--joint-learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/high_dimensional_inputs.csv"),
    )
    args = parser.parse_args()
    results = run_benchmark(
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        neural_epochs=args.neural_epochs,
        joint_steps=args.joint_steps,
        joint_learning_rate=args.joint_learning_rate,
        seed=args.seed,
    )
    write_csv(results, args.output)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
