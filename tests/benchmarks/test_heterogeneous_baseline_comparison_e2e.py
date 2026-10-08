"""Comparable fixed-budget BO, random, and Sobol benchmark baselines."""

import torch

from robotorchan.benchmarks.heterogeneous_baselines import run_strategy
from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth


def test_fixed_budget_random_sobol_and_qei_baselines() -> None:
    """Use shared initial observations and count all new function evaluations."""
    seed = 1414
    initial_X = torch.rand(
        12,
        3,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.double,
    )
    results = {
        strategy: run_strategy(strategy, initial_X, seed=seed, steps=3)
        for strategy in ("random", "sobol", "qei")
    }
    initial_truth, _, _ = evaluate_truth(initial_X)
    for evaluated_X, best in results.values():
        assert evaluated_X.shape == (15, 3)
        torch.testing.assert_close(evaluated_X[:12], initial_X)
        assert best.shape == (4,)
        assert torch.isfinite(best).all()
        torch.testing.assert_close(best[0], initial_truth.max())
        assert (best[1:] >= best[:-1]).all()
        final_truth, _, _ = evaluate_truth(evaluated_X)
        torch.testing.assert_close(best[-1], final_truth.max())
    assert len(results) == 3
