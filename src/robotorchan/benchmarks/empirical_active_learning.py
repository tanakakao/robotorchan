"""Reproducible classification active-learning baseline learning curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import Tensor

from robotorchan.benchmarks.active_learning_problems import (
    active_learning_labels,
    brier_score,
    classification_accuracy,
)
from robotorchan.benchmarks.random_streams import derive_stream_seed

_BOUNDARIES = ("circle", "wave")
_STRATEGIES = ("random", "sobol")


def kernel_label_probabilities(X: Tensor, labels: Tensor, query: Tensor) -> Tensor:
    """Distance-weighted Bernoulli smoothing; a fixed non-GP reference baseline."""
    if X.ndim != 2 or X.shape[1] != 2 or query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("Expected two-dimensional train and reference inputs.")
    if labels.shape != (len(X), 1) or len(X) == 0:
        raise ValueError("Expected nonempty canonical labels.")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Expected binary labels.")
    distances = torch.cdist(query, X).square()
    weights = torch.exp(-distances / (2.0 * 0.15**2))
    numerator = weights @ labels + 0.5
    denominator = weights.sum(dim=-1, keepdim=True) + 1.0
    return numerator / denominator


def run_active_learning_baselines(
    destination: str | Path, *, boundary: str, git_sha: str
) -> dict[str, object]:
    """Record paired 10-seed Random/Sobol classification learning curves."""
    if boundary not in _BOUNDARIES or not git_sha:
        raise ValueError("Unknown active-learning boundary or source revision.")
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite active-learning evidence.")
    root.mkdir(parents=True)
    manifest: dict[str, object] = {
        "boundary": boundary,
        "strategies": list(_STRATEGIES),
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "reference_points": 1024,
        "dtype": "float64",
        "device": "cpu",
        "git_sha": git_sha,
        "predictor": "fixed_bandwidth_kernel_label_smoother_not_gp",
        "reference": "independent_scrambled_sobol_seed_93827",
        "status": "running",
    }
    path = root / "manifest.json"

    def persist() -> None:
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    persist()
    rows = []
    try:
        reference = torch.quasirandom.SobolEngine(2, scramble=True, seed=93827).draw(1024).double()
        reference_labels = active_learning_labels(reference, boundary=boundary)
        for seed in range(10):
            initial = (
                torch.quasirandom.SobolEngine(
                    2, scramble=True, seed=derive_stream_seed(seed, "initial_design")
                )
                .draw(20)
                .double()
            )
            for strategy in _STRATEGIES:
                if strategy == "random":
                    generator = torch.Generator().manual_seed(derive_stream_seed(seed, "candidate"))
                    additional = torch.rand(50, 2, generator=generator).double()
                else:
                    additional = (
                        torch.quasirandom.SobolEngine(
                            2,
                            scramble=True,
                            seed=derive_stream_seed(seed, "candidate"),
                        )
                        .draw(50)
                        .double()
                    )
                X = torch.cat((initial, additional), dim=0)
                labels = active_learning_labels(X, boundary=boundary)
                if labels.shape != (70, 1) or X.dtype != torch.float64:
                    raise ValueError("Invalid active-learning sample contract.")
                for count in range(20, 71):
                    probabilities = kernel_label_probabilities(X[:count], labels[:count], reference)
                    rows.append(
                        {
                            "strategy": strategy,
                            "seed": seed,
                            "post_initial": count - 20,
                            "total_labels": count,
                            "reference_accuracy": float(
                                classification_accuracy(reference_labels, probabilities)
                            ),
                            "reference_brier": float(brier_score(reference_labels, probabilities)),
                            "train_pass_count": int(labels[:count].sum()),
                        }
                    )
        with (root / "classification_learning_curves.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest["status"] = "runtime_validated"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        persist()
    return manifest
