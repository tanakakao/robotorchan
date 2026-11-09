"""Focused tests for safe BO ask/tell workflow."""

import pytest
import torch

from robotorchan.optim.safe_workflow import SafeBOWorkflow
from robotorchan.semantics import ContinuousConstraint, SafetySpecification


class _Posterior:
    def __init__(self, X: torch.Tensor) -> None:
        self.mean = X[..., :1]
        self.variance = torch.zeros_like(self.mean)


class _SafetyModel:
    def posterior(self, X: torch.Tensor) -> _Posterior:
        return _Posterior(X)


class _Acquisition:
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X[..., 0, 0]


def _workflow() -> SafeBOWorkflow:
    specification = SafetySpecification(
        constraints=(ContinuousConstraint(0, 0.5),),
        initial_X=torch.tensor([[0.0]]),
        observed_constraints=torch.tensor([[0.0]]),
    )
    return SafeBOWorkflow(specification, torch.tensor([[0.0], [1.0]]))


def test_ask_tell_records_observation_and_prevents_double_ask() -> None:
    workflow = _workflow()
    candidate = workflow.ask(_SafetyModel(), acq_function=_Acquisition(), num_samples=64, seed=4)
    assert candidate.item() <= 0.5
    with pytest.raises(RuntimeError, match="tell"):
        workflow.ask(_SafetyModel(), acq_function=_Acquisition())
    workflow.tell(candidate, torch.tensor([[0.7]]))
    assert workflow.observed_X.shape == (2, 1)
    assert workflow.observed_constraints[-1, 0].item() == pytest.approx(0.7)
    assert workflow.ask(_SafetyModel(), mode="expansion", num_samples=32, seed=4).shape == (
        1,
        1,
    )


def test_tell_requires_pending_matching_candidate() -> None:
    workflow = _workflow()
    with pytest.raises(RuntimeError, match="ask"):
        workflow.tell(torch.tensor([[0.1]]), torch.tensor([[0.1]]))
    candidate = workflow.ask(_SafetyModel(), mode="expansion", num_samples=32, seed=2)
    with pytest.raises(ValueError, match="pending"):
        workflow.tell(candidate + 0.1, torch.tensor([[0.1]]))
