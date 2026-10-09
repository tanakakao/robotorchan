"""Tests for paired single-objective comparison metrics."""

import pytest
import torch

from robotorchan.benchmarks.comparative_single_objective import compare_single_objective
from robotorchan.benchmarks.runner import BenchmarkTrajectory
from robotorchan.benchmarks.standard_problems import branin


def _trajectory(seed: int, *, initial: float = 0.0, final: float = 0.0) -> BenchmarkTrajectory:
    X = torch.tensor([[initial, 1.0], [1.0, 2.0], [final, 3.0]], dtype=torch.double)
    Y = branin().evaluate_truth(X)
    return BenchmarkTrajectory(
        seed=seed,
        X=X,
        Y_observed=Y.clone(),
        Y_truth=Y,
        constraints=torch.empty((3, 0), dtype=torch.double),
        costs=torch.ones((3, 1), dtype=torch.double),
        initial_points=1,
    )


def test_comparison_aligns_methods_and_candidate_budgets() -> None:
    first = (_trajectory(0), _trajectory(1))
    second = (_trajectory(0, final=2.0), _trajectory(1, final=2.0))
    result = compare_single_objective(branin(), {"random": first, "qEI": second}, q_by_method={"random": 1, "qEI": 1})
    assert result.seeds == (0, 1)
    assert result.evaluations.tolist() == [0, 1, 2]
    assert result.regret_by_method["random"].shape == (2, 3)
    torch.testing.assert_close(
        result.mean_regret_by_method["qEI"],
        result.regret_by_method["qEI"].mean(dim=0),
    )


def test_single_seed_standard_error_is_zero() -> None:
    result = compare_single_objective(branin(), {"sobol": (_trajectory(0),)}, q_by_method={"sobol": 1})
    assert torch.count_nonzero(result.standard_error_by_method["sobol"]) == 0


def test_comparison_rejects_mismatched_initial_design() -> None:
    with pytest.raises(ValueError, match="Initial designs"):
        compare_single_objective(
            branin(),
            {"random": (_trajectory(0),), "qNEI": (_trajectory(0, initial=1.0),)},
            q_by_method={"random": 1, "qNEI": 1},
        )


def test_comparison_rejects_mismatched_seeds() -> None:
    with pytest.raises(ValueError, match="Seed ordering"):
        compare_single_objective(
            branin(),
            {"random": (_trajectory(0),), "qEI": (_trajectory(1),)},
            q_by_method={"random": 1, "qEI": 1},
        )


def test_comparison_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="At least one method"):
        compare_single_objective(branin(), {}, q_by_method={})


def test_comparison_rejects_mixed_batch_sizes() -> None:
    with pytest.raises(ValueError, match="Batch sizes must match"):
        compare_single_objective(
            branin(),
            {"random": (_trajectory(0),), "qEI": (_trajectory(0),)},
            q_by_method={"random": 1, "qEI": 3},
        )


def test_comparison_rejects_missing_batch_metadata() -> None:
    with pytest.raises(ValueError, match="metadata"):
        compare_single_objective(
            branin(),
            {"random": (_trajectory(0),)},
            q_by_method={},
        )


def test_initial_incumbent_is_evaluation_zero() -> None:
    result = compare_single_objective(
        branin(),
        {"random": (_trajectory(0),), "qEI": (_trajectory(0, final=2.0),)},
        q_by_method={"random": 1, "qEI": 1},
    )
    torch.testing.assert_close(
        result.regret_by_method["random"][:, 0],
        result.regret_by_method["qEI"][:, 0],
    )
