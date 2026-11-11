"""Focused tests for combinatorial Hamming geometry."""

import pytest
import torch

from robotorchan.optim.constraints import CandidateConstraints
from robotorchan.optim.domains.combinatorial import hamming_distance, hamming_neighbors
from robotorchan.optim.domains.variable_space import MixedVariableSpace


def test_neighbors_and_distance():
    space = MixedVariableSpace(
        torch.tensor([[0.0, 0.0], [1.0, 5.0]], dtype=torch.double),
        integer_dims=(0,),
        categorical_values={1: (0.0, 2.0, 5.0)},
    )
    center = torch.tensor([0.0, 2.0], dtype=torch.double)
    neighbors = hamming_neighbors(center, space)
    assert {tuple(x.tolist()) for x in neighbors} == {(1.0, 2.0), (0.0, 0.0), (0.0, 5.0)}
    assert torch.equal(hamming_distance(neighbors, center, space), torch.ones(3, dtype=torch.long))


def test_constraints_and_exclusions():
    space = MixedVariableSpace(
        torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double),
        integer_dims=(0, 1),
    )
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0, 1]), torch.tensor([1.0, 1.0]), 2.0),),
    )
    neighbors = hamming_neighbors(
        torch.tensor([0.0, 1.0], dtype=torch.double),
        space,
        constraints=constraints,
        excluded=((0.0, 2.0),),
    )
    assert neighbors.shape == (1, 2)
    assert neighbors.tolist() == [[1.0, 1.0]]


def test_reject_continuous_and_invalid_center():
    space = MixedVariableSpace(torch.tensor([[0.0, 0.0], [1.0, 1.0]]), integer_dims=(0,))
    with pytest.raises(ValueError, match="fully discrete"):
        hamming_neighbors(torch.tensor([0.0, 0.0]), space)
    discrete = MixedVariableSpace(torch.tensor([[0.0, 0.0], [1.0, 1.0]]), integer_dims=(0, 1))
    with pytest.raises(ValueError, match="illegal"):
        hamming_neighbors(torch.tensor([0.5, 0.0]), discrete)


def test_float32_fractional_categories_do_not_include_center():
    space = MixedVariableSpace(torch.tensor([[0.0], [1.0]]), categorical_values={0: (0.1, 0.2)})
    center = torch.tensor([0.1])
    neighbors = hamming_neighbors(center, space)
    assert neighbors.shape == (1, 1)
    assert hamming_distance(center, neighbors, space).tolist() == [1]


def test_large_integer_domain_distance_does_not_enumerate():
    space = MixedVariableSpace(
        torch.tensor([[0.0], [1_000_000_000.0]], dtype=torch.double),
        integer_dims=(0,),
    )
    assert (
        hamming_distance(
            torch.tensor([0.0], dtype=torch.double),
            torch.tensor([1_000_000_000.0], dtype=torch.double),
            space,
        ).item()
        == 1
    )
