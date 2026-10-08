"""Phase 20: cross-phase synthetic benchmark integration audit."""

import json
from pathlib import Path

import torch

from robotorchan.benchmarks.heterogeneous_multiseed import summarize_trajectories
from robotorchan.benchmarks.heterogeneous_synthetic import (
    evaluate_truth,
    observe,
    reference_front,
)


def test_synthetic_oracle_observation_and_front_contract() -> None:
    """Validate common shapes, dtypes, and feasible Pareto oracle bounds."""
    X = torch.tensor(
        [[0.2, 0.5, 0.8], [0.8, 0.3, 0.7]],
        dtype=torch.double,
    )
    strength, conductivity, probability = evaluate_truth(X)
    observed = observe(X, seed=20, noise_std=0.0)
    torch.testing.assert_close(observed.strength, strength)
    torch.testing.assert_close(observed.conductivity, conductivity)
    assert observed.passed.shape == (2,)
    assert observed.passed.dtype == torch.long
    assert ((probability >= 0) & (probability <= 1)).all()

    front_X, front_Y = reference_front(grid_size=4, probability_threshold=0.5)
    assert front_X.ndim == 2 and front_X.shape[1] == 3
    assert front_Y.shape == (front_X.shape[0], 2)
    assert torch.isfinite(front_Y).all()
    assert (evaluate_truth(front_X)[2] >= 0.5).all()


def test_statistical_summary_preserves_paired_seed_shape() -> None:
    """The report uses unbiased sample spread across distinct seeds."""
    curves = torch.tensor(
        [[0.2, 0.4, 0.5], [0.2, 0.6, 0.7], [0.2, 0.5, 0.6]],
        dtype=torch.double,
    )
    mean, std, sem = summarize_trajectories(curves)
    assert mean.shape == std.shape == sem.shape == (3,)
    torch.testing.assert_close(mean, curves.mean(dim=0))
    torch.testing.assert_close(std, curves.std(dim=0, unbiased=True))
    torch.testing.assert_close(sem, std / (3**0.5))


def test_notebook_and_report_document_same_experiment() -> None:
    """Keep the reproducibility parameters aligned across code and report."""
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks" / "heterogeneous_bo_benchmark_phase19.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    report = (
        root / "docs" / "optimization" / "heterogeneous-e2e-benchmark-report.md"
    ).read_text(encoding="utf-8")
    assert "SEEDS = (1901, 1902, 1903)" in source
    assert "INITIAL_POINTS = 12" in source
    assert "STEPS = 3" in source
    assert "1901, 1902, 1903" in report
    assert "12 initial" in report
    assert "three additional" in report
