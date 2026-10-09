"""Test candidate-domain constraints separately from outcome constraints."""

import pytest
import torch

from robotorchan.benchmarks.candidate_turbo_problems import (
    TrustRegionState,
    candidate_linear_region,
    candidate_nonlinear_region,
    register_candidate_constraint_problems,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


@pytest.mark.parametrize(
    ("factory", "inside", "outside"),
    [
        (candidate_linear_region, [0.3, 0.4], [0.9, 0.9]),
        (candidate_nonlinear_region, [0.5, 0.5], [0.9, 0.9]),
    ],
)
def test_candidate_feasibility(factory, inside, outside) -> None:
    scenario = factory()
    X = torch.tensor([inside], dtype=torch.double)
    scenario.validate(X)
    assert scenario.problem.evaluate_truth(X).shape == (1, 1)
    assert scenario.problem.evaluate_constraints(X).shape == (1, 0)
    with pytest.raises(ValueError, match="Candidate constraints"):
        scenario.validate(torch.tensor([outside], dtype=torch.double))


def test_candidate_batch_validation() -> None:
    scenario = candidate_linear_region()
    X = torch.tensor([[[0.1, 0.2], [0.4, 0.3]]], dtype=torch.double)
    scenario.validate(X)
    X[0, 1] = torch.tensor([0.9, 0.9], dtype=torch.double)
    with pytest.raises(ValueError, match="Candidate constraints"):
        scenario.validate(X)


def test_trust_region_expand_shrink_and_restart() -> None:
    state = TrustRegionState(
        center=torch.tensor([0.5, 0.5]),
        length=0.4,
        min_length=0.15,
        success_tolerance=2,
        failure_tolerance=2,
    )
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    torch.testing.assert_close(
        state.bounds(bounds), torch.tensor([[0.3, 0.3], [0.7, 0.7]])
    )
    state.update(True)
    state.update(True)
    assert state.length == pytest.approx(0.8)
    state.update(False)
    state.update(False)
    assert state.length == pytest.approx(0.4)
    for _ in range(4):
        state.update(False)
    assert state.restart_required


def test_candidate_registry() -> None:
    registry = BenchmarkProblemRegistry()
    register_candidate_constraint_problems(registry)
    assert registry.names() == ("candidate_linear_region", "candidate_nonlinear_region")
    assert registry.create("candidate_linear_region").dimension == 2
