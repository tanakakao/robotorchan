"""Focused tests for context-fixed candidate proposals."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import ContextualOptimizer


class QuadraticAcquisition(AcquisitionFunction):
    def __init__(self):
        super().__init__(model=None)

    def forward(self, X):
        return -((X[..., 0] - X[..., 1]) ** 2).sum(dim=-1)


def test_context_is_fixed_and_decision_extracted():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    optimizer = ContextualOptimizer(
        bounds=bounds, decision_indices=(0,), context_indices=(1,), optimizer="sobol"
    )
    candidates, _ = optimizer.optimize(
        acq_function=QuadraticAcquisition(),
        context=torch.tensor([0.7], dtype=torch.double),
        q=1,
        seed=4,
        optimizer_options={"num_samples": 64},
    )
    assert candidates.shape == (1, 2)
    assert torch.allclose(candidates[:, 1], torch.tensor([0.7], dtype=torch.double))
    assert optimizer.extract_decisions(candidates).shape == (1, 1)


@pytest.mark.parametrize("decision,context", [((0,), (0,)), ((0,), ()), ((0,), (2,))])
def test_invalid_partition(decision, context):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        ContextualOptimizer(bounds=bounds, decision_indices=decision, context_indices=context)


def test_invalid_context_rejected():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    optimizer = ContextualOptimizer(bounds=bounds, decision_indices=(0,), context_indices=(1,))
    with pytest.raises(ValueError, match="dtype"):
        optimizer.optimize(acq_function=QuadraticAcquisition(), context=torch.tensor([0.5]))
    with pytest.raises(ValueError, match="outside bounds"):
        optimizer.optimize(
            acq_function=QuadraticAcquisition(),
            context=torch.tensor([1.5], dtype=torch.double),
        )
