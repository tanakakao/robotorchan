"""Focused tests for Hamming acquisition local search."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import CandidateConstraints, HammingLocalSearchStrategy, MixedVariableSpace


class TargetAcquisition(AcquisitionFunction):
    def __init__(self, target: torch.Tensor) -> None:
        super().__init__(model=None)
        self.target = target

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return -(X.squeeze(-2) - self.target).square().sum(dim=-1)


def test_multistart_reaches_best_binary_candidate():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, integer_dims=(0, 1))
    starts = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.double)
    result = HammingLocalSearchStrategy(space, starts).optimize(
        TargetAcquisition(torch.tensor([1.0, 1.0], dtype=torch.double))
    )
    assert result.candidates.tolist() == [[1.0, 1.0]]
    assert result.acquisition_value.item() == 0.0
    assert result.metadata["evaluations"] >= 2


def test_constraints_exclusions_and_q_contract():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, integer_dims=(0, 1))
    starts = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (torch.tensor([0, 1]), torch.tensor([-1.0, -1.0]), -1.0),
        )
    )
    strategy = HammingLocalSearchStrategy(
        space, starts, constraints=constraints, excluded=((1.0, 1.0),)
    )
    acquisition = TargetAcquisition(torch.tensor([1.0, 1.0], dtype=torch.double))
    result = strategy.optimize(acquisition)
    assert result.candidates.tolist() == [[1.0, 0.0]]
    with pytest.raises(NotImplementedError, match="q=1"):
        strategy.optimize(acquisition, q=2)


def test_no_eligible_start_raises():
    space = MixedVariableSpace(torch.tensor([[0.0], [1.0]]), integer_dims=(0,))
    strategy = HammingLocalSearchStrategy(
        space, torch.tensor([[0.0]]), excluded=((0.0,),)
    )
    with pytest.raises(ValueError, match="No feasible"):
        strategy.optimize(TargetAcquisition(torch.tensor([1.0])))
