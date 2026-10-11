"""Smoke tests for oracle-isolated comparative acquisition callback paths."""

from unittest.mock import patch

import pytest
import torch

from robotorchan.benchmarks.candidate_view import CandidateProblemView
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_qehvi import run_comparative_qehvi
from robotorchan.benchmarks.comparative_qei import run_comparative_qei
from robotorchan.benchmarks.comparative_qnehvi import run_comparative_qnehvi
from robotorchan.benchmarks.comparative_qnei import run_comparative_qnei
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.multiobjective_problems import branin_currin
from robotorchan.benchmarks.standard_problems import branin


@pytest.mark.parametrize(
    ("problem", "strategy", "runner", "backend"),
    [
        ("branin", "qEI", run_comparative_qei, "comparative_qei"),
        ("branin", "qNEI", run_comparative_qnei, "comparative_qnei"),
        ("branin_currin", "qEHVI", run_comparative_qehvi, "comparative_qehvi"),
        ("branin_currin", "qNEHVI", run_comparative_qnehvi, "comparative_qnehvi"),
    ],
)
def test_comparative_callback_receives_only_public_problem_view(
    problem, strategy, runner, backend
) -> None:
    """Exercise runner-to-acquisition wiring without an expensive GP fit."""
    cell = ComparativeExperimentCell(
        tier="smoke",
        config=BenchmarkExperimentConfig(
            problem=problem,
            strategy=strategy,
            seeds=(0,),
            initial_points=4,
            evaluation_budget=4,
            q=1,
        ),
    )
    calls = []

    def candidate(view, X, Y, q, generator, **kwargs):
        assert isinstance(view, CandidateProblemView)
        assert not hasattr(view, "evaluate_truth")
        assert not hasattr(view, "optimal_value")
        assert X.shape[0] == Y.shape[0]
        assert Y.shape[1] == view.n_objectives
        assert generator is not None
        calls.append(q)
        return (view.bounds[0] + view.bounds[1]).to(X).expand(q, -1) / 2

    target = "botorch_gp_candidates" if strategy in ("qEI", "qNEI") else "botorch_qehvi_candidates"
    with patch(f"robotorchan.benchmarks.{backend}.{target}", side_effect=candidate):
        trajectories = runner(cell, num_restarts=1, raw_samples=4)
    assert calls == [1, 1, 1, 1]
    assert trajectories[0].evaluation_count == 4


@pytest.mark.parametrize("factory", [branin, branin_currin])
def test_candidate_view_has_no_truth_and_copies_public_reference(factory) -> None:
    problem = factory()
    view = CandidateProblemView.from_problem(problem)
    assert not hasattr(view, "evaluate_truth")
    assert not hasattr(view, "evaluate_constraints")
    assert view.dimension == problem.dimension
    assert torch.equal(view.bounds, problem.bounds)
    assert view.bounds.data_ptr() != problem.bounds.data_ptr()
    if problem.reference_point is not None:
        assert view.reference_point is not None
        assert torch.equal(view.reference_point, problem.reference_point)
        assert view.reference_point.data_ptr() != problem.reference_point.data_ptr()
