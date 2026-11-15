"""Reproducible multiclass active-learning benchmark (not a performance claim).

Run: python examples/benchmarks/multiclass_active_learning.py
The same initial labels, evaluation set, pool and label budget are shared across
Random and PredictiveEntropy. Repeat seeds before interpreting performance.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from robotorchan.models.classification import (
    MixedMulticlassSingleTaskGPClassifier,
    MulticlassSingleTaskGPClassifier,
)


def labels(X: torch.Tensor, *, mixed: bool) -> torch.Tensor:
    if mixed:
        return ((X[:, 0] > 0.5).long() + X[:, 1].long()) % 3
    return torch.bucketize(
        X[:, 0] + 0.3 * X[:, 1],
        torch.tensor([0.43, 0.86], dtype=X.dtype),
    )


def dataset(seed: int, *, mixed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    X = torch.rand(120, 2, generator=generator, dtype=torch.double)
    if mixed:
        X[:, 1] = torch.randint(0, 3, (len(X),), generator=generator).double()
    return X, labels(X, mixed=mixed)


def fit_model(X: torch.Tensor, y: torch.Tensor, *, mixed: bool, epochs: int):
    options = {"num_classes": 3, "inducing_points": min(8, len(X))}
    if mixed:
        model = MixedMulticlassSingleTaskGPClassifier(X, y, cat_dims=[1], **options)
    else:
        model = MulticlassSingleTaskGPClassifier(X, y, **options)
    model.fit(epochs=epochs, lr=0.03)
    return model


def evaluate(
    seed: int,
    *,
    mixed: bool,
    method: str,
    budget: int,
    epochs: int,
) -> list[dict[str, str | int | float]]:
    torch.manual_seed(seed)
    X, y = dataset(seed, mixed=mixed)
    # Disjoint fixed splits: initial 15, candidate pool 75, holdout 30.
    train_idx = list(range(15))
    pool_idx = list(range(15, 90))
    test_X, test_y = X[90:], y[90:]
    rows = []
    for step in range(budget + 1):
        torch.manual_seed(seed + step)
        model = fit_model(X[train_idx], y[train_idx], mixed=mixed, epochs=epochs)
        with torch.no_grad():
            probabilities = model.predict_proba(test_X, num_samples=32).clamp_min(1e-12)
            accuracy = (probabilities.argmax(-1) == test_y).double().mean().item()
            nll = -probabilities.gather(1, test_y[:, None]).log().mean().item()
            one_hot = torch.nn.functional.one_hot(test_y, 3).double()
            brier = (probabilities - one_hot).square().sum(-1).mean().item()
        rows.append(
            {
                "seed": seed,
                "mixed": int(mixed),
                "method": method,
                "labels_added": step,
                "accuracy": accuracy,
                "nll": nll,
                "brier": brier,
            }
        )
        if step == budget:
            break
        if method == "random":
            choice = pool_idx[torch.randint(len(pool_idx), (1,)).item()]
        else:
            with torch.no_grad():
                probabilities = model.predict_proba(X[pool_idx], num_samples=32)
                scores = -torch.special.xlogy(probabilities, probabilities).sum(-1)
                choice = pool_idx[scores.argmax().item()]
        train_idx.append(choice)
        pool_idx.remove(choice)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("multiclass_al_results.csv"))
    args = parser.parse_args()
    if args.seeds < 1 or not 0 <= args.budget <= 75 or args.epochs < 1:
        parser.error("seeds and epochs must be positive; budget must be 0..75")
    rows = [
        row
        for seed in range(args.seeds)
        for mixed in (False, True)
        for method in ("random", "entropy")
        for row in evaluate(
            seed, mixed=mixed, method=method, budget=args.budget, epochs=args.epochs
        )
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} observations to {args.output}")


if __name__ == "__main__":
    main()
