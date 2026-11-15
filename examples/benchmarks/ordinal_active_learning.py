"""Reproducible ordinal GP active-learning benchmark, not a performance claim.

Run: python examples/benchmarks/ordinal_active_learning.py --seeds 3 --budget 5
All methods share identical initial data, candidate pool, holdout and label
budget per seed and input type. Selected pool labels are not used until chosen.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from robotorchan.acquisition.active_learning.classification import (
    BALD,
    PredictiveEntropy,
)
from robotorchan.models.classification import (
    MixedOrdinalSingleTaskGPClassifier,
    OrdinalSingleTaskGPClassifier,
)

METHODS = ("random", "entropy", "bald")
NUM_CLASSES = 3
INITIAL_SIZE = 15
POOL_END = 90
DATASET_SIZE = 120


def labels(X: torch.Tensor, *, mixed: bool) -> torch.Tensor:
    """Generate ordered labels using a monotone latent quality score."""
    if mixed:
        quality = X[:, 0] + 0.12 * (X[:, 1] == 2).to(X) - 0.06 * (X[:, 1] == 0).to(X)
    else:
        quality = X[:, 0] + 0.25 * (X[:, 1] - 0.5)
    boundaries = torch.tensor([0.35, 0.70], dtype=X.dtype, device=X.device)
    return torch.bucketize(quality.contiguous(), boundaries)


def dataset(seed: int, *, mixed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate fixed synthetic observations without using the global RNG."""
    generator = torch.Generator().manual_seed(seed)
    X = torch.rand(DATASET_SIZE, 2, generator=generator, dtype=torch.double)
    if mixed:
        X[:, 1] = torch.randint(
            0, NUM_CLASSES, (DATASET_SIZE,), generator=generator
        ).to(dtype=X.dtype)
    return X, labels(X, mixed=mixed)


def fit_model(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    mixed: bool,
    epochs: int,
) -> OrdinalSingleTaskGPClassifier:
    """Fit the existing ordinal model, without a benchmark-specific surrogate."""
    options = {"num_classes": NUM_CLASSES, "inducing_points": min(8, len(X))}
    if mixed:
        model = MixedOrdinalSingleTaskGPClassifier(X, y, cat_dims=[1], **options)
    else:
        model = OrdinalSingleTaskGPClassifier(X, y, **options)
    model.fit(epochs=epochs, lr=0.03)
    return model


def heldout_metrics(
    model: OrdinalSingleTaskGPClassifier,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
) -> dict[str, float]:
    """Evaluate a fixed holdout without using labels for acquisition selection."""
    with torch.no_grad():
        probabilities = model.predict_proba(test_X).clamp_min(1e-12)
        predicted = probabilities.argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(
            test_y, num_classes=NUM_CLASSES
        ).to(probabilities)
        return {
            "accuracy": (predicted == test_y).double().mean().item(),
            "nll": -probabilities.gather(1, test_y[:, None]).log().mean().item(),
            "brier": (probabilities - one_hot).square().sum(-1).mean().item(),
            "ordinal_mae": (predicted - test_y).abs().double().mean().item(),
        }


def evaluate(
    seed: int,
    *,
    mixed: bool,
    method: str,
    budget: int,
    epochs: int,
    num_samples: int = 32,
) -> list[dict[str, str | int | float]]:
    """Run one fair sequential acquisition trace on a fixed dataset split."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}.")
    if epochs < 1 or not 0 <= budget <= POOL_END - INITIAL_SIZE:
        raise ValueError("epochs must be positive and budget must be 0..75.")
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")

    X, y = dataset(seed, mixed=mixed)
    train_idx = list(range(INITIAL_SIZE))
    pool_idx = list(range(INITIAL_SIZE, POOL_END))
    test_X, test_y = X[POOL_END:], y[POOL_END:]
    rows: list[dict[str, str | int | float]] = []
    random_generator = torch.Generator().manual_seed(seed + 10_000)

    for step in range(budget + 1):
        # Each method has the same initialization at the same step.
        # Training data are allowed to differ after selecting different labels.
        torch.manual_seed(seed + 1_000 * (step + 1))
        model = fit_model(X[train_idx], y[train_idx], mixed=mixed, epochs=epochs)
        rows.append(
            {
                "seed": seed,
                "mixed": int(mixed),
                "method": method,
                "labels_added": step,
                **heldout_metrics(model, test_X, test_y),
            }
        )
        if step == budget:
            break

        if method == "random":
            position = torch.randint(
                len(pool_idx), (1,), generator=random_generator
            ).item()
        else:
            with torch.no_grad():
                pool = X[pool_idx].unsqueeze(-2)
                acquisition = (
                    PredictiveEntropy(model)
                    if method == "entropy"
                    else BALD(model, num_samples=num_samples)
                )
                position = acquisition(pool).argmax().item()

        # A pool label enters the training set only after its input is selected.
        selected_index = pool_idx.pop(position)
        train_idx.append(selected_index)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("ordinal_al_results.csv"))
    args = parser.parse_args()

    if args.seeds < 1 or args.epochs < 1 or args.num_samples < 2:
        parser.error("seeds and epochs must be positive; num-samples must be >= 2.")
    if not 0 <= args.budget <= POOL_END - INITIAL_SIZE:
        parser.error("budget must be between 0 and 75.")

    rows = [
        row
        for seed in range(args.seeds)
        for mixed in (False, True)
        for method in METHODS
        for row in evaluate(
            seed,
            mixed=mixed,
            method=method,
            budget=args.budget,
            epochs=args.epochs,
            num_samples=args.num_samples,
        )
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} observations to {args.output}")


if __name__ == "__main__":
    main()
