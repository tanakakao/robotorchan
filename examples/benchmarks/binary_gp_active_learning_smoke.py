"""Minimal binary GP active-learning closed-loop smoke experiment.

Run: python examples/benchmarks/binary_gp_active_learning_smoke.py
This single-seed smoke example is NOT multi-seed performance evidence.
"""

from __future__ import annotations

import argparse

import torch

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.benchmarks.active_learning_problems import (
    active_learning_labels,
    brier_score,
    classification_accuracy,
)
from robotorchan.benchmarks.random_streams import derive_stream_seed
from robotorchan.models.classification import BinarySingleTaskGPClassifier


def run(*, seed: int = 0, budget: int = 2, epochs: int = 8) -> list[dict[str, float | int | str]]:
    """Compare acquisition policies with the same fitted binary GP predictor."""
    if budget < 1 or epochs < 1:
        raise ValueError("budget and epochs must be positive.")
    reference = torch.quasirandom.SobolEngine(2, scramble=True, seed=93827).draw(1024).double()
    reference_labels = active_learning_labels(reference, boundary="wave")
    initial = torch.quasirandom.SobolEngine(
        2, scramble=True, seed=derive_stream_seed(seed, "initial_design")
    ).draw(20).double()
    pool = torch.quasirandom.SobolEngine(
        2, scramble=True, seed=derive_stream_seed(seed, "candidate")
    ).draw(256).double()
    # Labels for pool candidates are intentionally not computed before selection.
    results: list[dict[str, float | int | str]] = []
    for strategy in ("random", "sobol", "predictive_entropy"):
        X = initial.clone()
        Y = active_learning_labels(X, boundary="wave")
        available = list(range(len(pool)))
        generator = torch.Generator().manual_seed(derive_stream_seed(seed, "candidate"))
        for step in range(budget + 1):
            # Identical model class, fit budget and training seed across policies.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed * 1000 + step)
                model = BinarySingleTaskGPClassifier(X, Y)
                model.fit(epochs=epochs, lr=0.03)
            model.eval()
            with torch.no_grad():
                probabilities = model.predict_proba(reference)[..., 1].reshape(-1, 1)
                if (
                    probabilities.shape != reference_labels.shape
                    or not torch.isfinite(probabilities).all()
                ):
                    raise ValueError("Invalid GP reference probabilities.")
                results.append({
                    "strategy": strategy,
                    "seed": seed,
                    "labels_added": step,
                    "accuracy": float(classification_accuracy(reference_labels, probabilities)),
                    "brier": float(brier_score(reference_labels, probabilities)),
                })
                if step == budget:
                    break
                if strategy == "random":
                    index = int(torch.randint(len(available), (1,), generator=generator))
                elif strategy == "sobol":
                    index = 0
                else:
                    acquisition = PredictiveEntropy(model)
                    scores = acquisition(pool[available].unsqueeze(-2)).reshape(-1)
                    if scores.shape != (len(available),) or not torch.isfinite(scores).all():
                        raise ValueError("Invalid GP predictive entropy.")
                    index = int(torch.argmax(scores))
            next_X = pool[available.pop(index)].reshape(1, 2)
            next_Y = active_learning_labels(next_X, boundary="wave")
            X = torch.cat((X, next_X), dim=0)
            Y = torch.cat((Y, next_Y), dim=0)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    for row in run(seed=args.seed, budget=args.budget, epochs=args.epochs):
        print(row)


if __name__ == "__main__":
    main()
