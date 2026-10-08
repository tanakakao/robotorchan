"""Contract tests for the fixed heterogeneous synthetic benchmark."""

import pytest
import torch

from robotorchan.benchmarks.heterogeneous_synthetic import (
    evaluate_truth,
    observe,
    reference_front,
)


def test_truth_has_competing_objective_optima() -> None:
    X = torch.tensor([[0.78, 0.28, 0.68], [0.22, 0.76, 0.35]], dtype=torch.double)
    strength, conductivity, probability = evaluate_truth(X)
    assert strength[0] > strength[1]
    assert conductivity[1] > conductivity[0]
    assert torch.all((probability > 0) & (probability < 1))


def test_truth_has_both_feasible_and_infeasible_regions() -> None:
    X = torch.tensor([[0.5, 0.53, 0.5], [0.0, 0.0, 0.0]], dtype=torch.double)
    _, _, probability = evaluate_truth(X)
    assert probability[0] >= 0.5
    assert probability[1] < 0.5


def test_observations_are_reproducible_and_binary() -> None:
    X = torch.rand(64, 3, dtype=torch.double)
    first = observe(X, seed=42)
    second = observe(X, seed=42)
    assert torch.equal(first.strength, second.strength)
    assert torch.equal(first.conductivity, second.conductivity)
    assert torch.equal(first.passed, second.passed)
    assert set(first.passed.unique().tolist()).issubset({0, 1})


def test_noiseless_observation_preserves_regression_truth() -> None:
    X = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.double)
    strength, conductivity, _ = evaluate_truth(X)
    observation = observe(X, seed=1, noise_std=0.0)
    assert torch.equal(observation.strength, strength)
    assert torch.equal(observation.conductivity, conductivity)


def test_reference_front_is_feasible_and_nondominated() -> None:
    from botorch.utils.multi_objective.pareto import is_non_dominated

    X, Y = reference_front(grid_size=9)
    assert len(X) > 0
    assert Y.shape == (len(X), 2)
    assert torch.all(evaluate_truth(X)[2] >= 0.5)
    assert torch.all(is_non_dominated(Y))


@pytest.mark.parametrize("invalid", [-0.1, 1.1, float("nan")])
def test_invalid_inputs_are_rejected(invalid: float) -> None:
    with pytest.raises(ValueError):
        evaluate_truth(torch.tensor([[invalid, 0.5, 0.5]], dtype=torch.double))
