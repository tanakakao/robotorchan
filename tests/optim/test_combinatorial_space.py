"""Focused tests for combinatorial Hamming geometry."""

import pytest
import torch

from robotorchan.optim.constraints import CandidateConstraints
from robotorchan.optim.domains.combinatorial import hamming_distance, hamming_neighbors
from robotorchan.optim.domains.variable_space import MixedVariableSpace


def test_neighbors_and_distance():
    space = MixedVariableSpace(
        torch.tensor([[0., 0.], [1., 5.]], dtype=torch.double),
        integer_dims=(0,),
        categorical_values={1: (0., 2., 5.)},
    )
    center = torch.tensor([0., 2.], dtype=torch.double)
    neighbors = hamming_neighbors(center, space)
    assert {tuple(x.tolist()) for x in neighbors} == {(1., 2.), (0., 0.), (0., 5.)}
    assert torch.equal(hamming_distance(neighbors, center, space), torch.ones(3, dtype=torch.long))


def test_constraints_and_exclusions():
    space = MixedVariableSpace(
        torch.tensor([[0., 0.], [1., 2.]], dtype=torch.double),
        integer_dims=(0, 1),
    )
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0, 1]), torch.tensor([1., 1.]), 2.),),
    )
    neighbors = hamming_neighbors(
        torch.tensor([0., 1.], dtype=torch.double),
        space,
        constraints=constraints,
        excluded=((0., 2.),),
    )
    assert neighbors.shape == (1, 2)
    assert neighbors.tolist() == [[1., 1.]]


def test_reject_continuous_and_invalid_center():
    space = MixedVariableSpace(torch.tensor([[0., 0.], [1., 1.]]), integer_dims=(0,))
    with pytest.raises(ValueError, match="fully discrete"):
        hamming_neighbors(torch.tensor([0., 0.]), space)
    discrete = MixedVariableSpace(
        torch.tensor([[0., 0.], [1., 1.]]), integer_dims=(0, 1)
    )
    with pytest.raises(ValueError, match="illegal"):
        hamming_neighbors(torch.tensor([0.5, 0.]), discrete)
