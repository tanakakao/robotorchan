"""Adaptive entropy heuristic evidence against paired active-learning baselines."""

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
from robotorchan.benchmarks.empirical_active_learning import kernel_label_probabilities
from robotorchan.benchmarks.random_streams import derive_stream_seed


def select_kernel_entropy_candidate(
    train_X: Tensor, train_labels: Tensor, candidates: Tensor
) -> int:
    """Select maximum Bernoulli entropy, breaking ties by candidate order."""
    if candidates.ndim != 2 or candidates.shape[1] != 2 or len(candidates) == 0:
        raise ValueError("Expected a nonempty 2D candidate pool.")
    probabilities = kernel_label_probabilities(train_X, train_labels, candidates).squeeze(-1)
    entropy = -torch.special.xlogy(probabilities, probabilities) - torch.special.xlogy(
        1 - probabilities, 1 - probabilities
    )
    if not torch.isfinite(entropy).all():
        raise ValueError("Nonfinite candidate entropy.")
    return int(torch.argmax(entropy))


def run_kernel_entropy_evidence(
    destination: str | Path, *, boundary: str, git_sha: str
) -> dict[str, object]:
    """Compare paired random, Sobol and sequential kernel-entropy selections."""
    if boundary not in ("circle", "wave") or not git_sha:
        raise ValueError("Unknown active-learning boundary or source revision.")
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite active-learning evidence.")
    root.mkdir(parents=True)
    manifest: dict[str, object] = {
        "boundary": boundary,
        "strategies": ["random", "sobol", "kernel_entropy_heuristic"],
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "candidate_pool": 256,
        "reference_points": 1024,
        "dtype": "float64",
        "device": "cpu",
        "git_sha": git_sha,
        "predictor": "fixed_bandwidth_kernel_label_smoother_not_gp",
        "acquisition": "kernel_probability_entropy_not_gp_predictive_entropy_or_bald",
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
            for strategy in manifest["strategies"]:
                X = initial.clone()
                labels = active_learning_labels(X, boundary=boundary)
                generator = torch.Generator().manual_seed(derive_stream_seed(seed, "candidate"))
                sobol = torch.quasirandom.SobolEngine(
                    2, scramble=True, seed=derive_stream_seed(seed, "candidate")
                )
                for step in range(51):
                    probabilities = kernel_label_probabilities(X, labels, reference)
                    rows.append(
                        {
                            "strategy": strategy,
                            "seed": seed,
                            "post_initial": step,
                            "total_labels": len(X),
                            "reference_accuracy": float(
                                classification_accuracy(reference_labels, probabilities)
                            ),
                            "reference_brier": float(brier_score(reference_labels, probabilities)),
                            "train_pass_count": int(labels.sum()),
                        }
                    )
                    if step == 50:
                        break
                    if strategy == "random":
                        next_X = torch.rand(1, 2, generator=generator).double()
                    elif strategy == "sobol":
                        next_X = sobol.draw(1).double()
                    else:
                        pool = sobol.draw(256).double()
                        index = select_kernel_entropy_candidate(X, labels, pool)
                        next_X = pool[index : index + 1]
                    next_label = active_learning_labels(next_X, boundary=boundary)
                    X = torch.cat((X, next_X), dim=0)
                    labels = torch.cat((labels, next_label), dim=0)
        with (root / "entropy_comparison_curves.csv").open(
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
