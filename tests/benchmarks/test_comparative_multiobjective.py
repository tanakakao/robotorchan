"""Phase 12 paired multiobjective comparison tests."""

from dataclasses import replace

import pytest
import torch

from robotorchan.benchmarks.comparative_multiobjective import compare_multiobjective
from robotorchan.benchmarks.multiobjective_problems import branin_currin
from robotorchan.benchmarks.runner import BenchmarkTrajectory, sobol_initial_design


def _run(seed: int, offset: float = 0.0, budget: int = 4) -> BenchmarkTrajectory:
    problem = branin_currin()
    initial = sobol_initial_design(
        problem, 4, seed, dtype=torch.double, device=torch.device("cpu")
    )
    additional = torch.full((budget, 2), 0.3 + offset, dtype=torch.double)
    X = torch.cat((initial, additional))
    Y = problem.evaluate_truth(X)
    return BenchmarkTrajectory(
        seed=seed,
        X=X,
        Y_observed=Y,
        Y_truth=Y,
        constraints=problem.evaluate_constraints(X),
        costs=problem.evaluate_cost(X),
        initial_points=4,
    )


def test_paired_hypervolume_includes_initial_and_terminal() -> None:
    runs = {
        "random": (_run(0), _run(1)),
        "sobol": (_run(0, 0.1), _run(1, 0.1)),
    }
    result = compare_multiobjective(
        branin_currin(), runs, q_by_method={"random": 1, "sobol": 1}
    )
    assert result.seeds == (0, 1)
    assert result.evaluations.tolist() == [0, 1, 2, 3, 4]
    assert result.hypervolume_by_method["random"].shape == (2, 5)
    assert result.standard_error_by_method["random"].shape == (5,)
    torch.testing.assert_close(
        result.hypervolume_by_method["random"][:, 0],
        result.hypervolume_by_method["sobol"][:, 0],
    )


def test_batch_checkpoints_include_final_partial_batch() -> None:
    result = compare_multiobjective(
        branin_currin(), {"random": (_run(0),)}, q_by_method={"random": 3}
    )
    assert result.evaluations.tolist() == [0, 3, 4]
    assert result.hypervolume_by_method["random"].shape == (1, 3)
    assert torch.equal(
        result.standard_error_by_method["random"],
        torch.zeros_like(result.standard_error_by_method["random"]),
    )


def test_comparison_rejects_mismatched_initial_design() -> None:
    altered = _run(0)
    altered = replace(altered, X=altered.X.clone())
    altered.X[0, 0] = 0.2
    with pytest.raises(ValueError, match="Initial designs"):
        compare_multiobjective(
            branin_currin(),
            {"random": (_run(0),), "sobol": (altered,)},
            q_by_method={"random": 1, "sobol": 1},
        )


def test_comparison_rejects_mismatched_batch_size() -> None:
    with pytest.raises(ValueError, match="Batch sizes must match"):
        compare_multiobjective(
            branin_currin(),
            {"random": (_run(0),), "sobol": (_run(0),)},
            q_by_method={"random": 1, "sobol": 3},
        )


def test_comparison_rejects_missing_batch_metadata() -> None:
    with pytest.raises(ValueError, match="metadata"):
        compare_multiobjective(branin_currin(), {"random": (_run(0),)}, q_by_method={})


def test_comparison_rejects_seed_mismatch() -> None:
    with pytest.raises(ValueError, match="Seed ordering"):
        compare_multiobjective(
            branin_currin(),
            {"random": (_run(0),), "sobol": (_run(1),)},
            q_by_method={"random": 1, "sobol": 1},
        )
