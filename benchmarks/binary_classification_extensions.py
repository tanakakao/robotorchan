"""Reproducible binary GP comparison on held-out synthetic labels.

Run: python benchmarks/binary_classification_extensions.py
This is a small illustrative comparison, not a claim of model superiority.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    InfiniteWidthBNNBinaryGPClassifier,
    SpectralMixtureBinaryGPClassifier,
)


def run(seed: int, epochs: int) -> list[dict[str, float | int | str]]:
    torch.manual_seed(seed)
    X = torch.rand(80, 1, dtype=torch.double)
    probabilities = torch.sigmoid(12 * (X[:, 0] - 0.5))
    Y = torch.bernoulli(probabilities)
    train_X, test_X = X[:48], X[48:]
    train_Y, test_Y = Y[:48], Y[48:]
    factories = {
        "standard": lambda: BinarySingleTaskGPClassifier(
            train_X, train_Y, inducing_points=12
        ),
        "spectral_mixture": lambda: SpectralMixtureBinaryGPClassifier(
            train_X, train_Y, num_mixtures=2, inducing_points=12
        ),
        "infinite_width_bnn": lambda: InfiniteWidthBNNBinaryGPClassifier(
            train_X, train_Y, depth=2, inducing_points=12
        ),
    }
    results = []
    for name, factory in factories.items():
        torch.manual_seed(seed)
        model = factory()
        start = time.perf_counter()
        model.fit(epochs=epochs, lr=0.03)
        fit_seconds = time.perf_counter() - start
        with torch.no_grad():
            predicted = model.predict_proba(test_X)[..., 1].clamp(1e-7, 1 - 1e-7)
        brier = (predicted - test_Y).square().mean().item()
        nll = -(
            test_Y * predicted.log() + (1 - test_Y) * torch.log1p(-predicted)
        ).mean().item()
        accuracy = ((predicted >= 0.5) == test_Y.bool()).double().mean().item()
        results.append(
            {
                "seed": seed,
                "model": name,
                "n_train": len(train_X),
                "n_test": len(test_X),
                "epochs": epochs,
                "brier": brier,
                "nll": nll,
                "accuracy": accuracy,
                "fit_seconds": fit_seconds,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.seeds < 1 or args.epochs < 1:
        parser.error("--seeds and --epochs must be positive")
    rows = [row for seed in range(args.seeds) for row in run(seed, args.epochs)]
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name in sorted({str(row["model"]) for row in rows}):
        subset = [row for row in rows if row["model"] == name]
        print(
            name,
            "brier", sum(float(row["brier"]) for row in subset) / len(subset),
            "nll", sum(float(row["nll"]) for row in subset) / len(subset),
            "accuracy", sum(float(row["accuracy"]) for row in subset) / len(subset),
        )


if __name__ == "__main__":
    main()
