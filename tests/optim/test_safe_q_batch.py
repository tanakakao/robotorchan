"""Focused q-batch Safe BO tests."""

import pytest
import torch

from robotorchan.optim.safe import optimize_safe_acqf
from robotorchan.optim.safe_expansion import propose_safe_expansion
from robotorchan.optim.safe_workflow import SafeBOWorkflow
from robotorchan.semantics import ContinuousConstraint, SafetySpecification


class _Posterior:
    def __init__(self, X: torch.Tensor) -> None:
        self.mean = X[..., :1]
        self.variance = torch.zeros_like(self.mean)


class _SafetyModel:
    def posterior(self, X: torch.Tensor) -> _Posterior:
        return _Posterior(X)


class _BatchAcquisition:
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X[..., 0].sum(dim=-1)


@pytest.mark.parametrize("mode", ["objective", "expansion"])
def test_safe_q_batch_respects_every_point(mode: str) -> None:
    kwargs = {
        "bounds": torch.tensor([[0.0], [1.0]]),
        "constraints": (ContinuousConstraint(0, 0.5),),
        "q": 3,
        "num_samples": 128,
        "seed": 3,
    }
    if mode == "objective":
        batch, _ = optimize_safe_acqf(_BatchAcquisition(), _SafetyModel(), **kwargs)
    else:
        batch, _ = propose_safe_expansion(_SafetyModel(), **kwargs)
    assert batch.shape == (3, 1)
    assert bool((batch <= 0.5).all())


def test_safe_q_batch_rejects_empty_safe_pool() -> None:
    with pytest.raises(RuntimeError, match="No posterior-safe candidate batches"):
        optimize_safe_acqf(
            _BatchAcquisition(),
            _SafetyModel(),
            torch.tensor([[0.0], [1.0]]),
            (ContinuousConstraint(0, -1.0),),
            q=2,
            num_samples=32,
            seed=3,
        )


def test_safe_workflow_records_batch() -> None:
    specification = SafetySpecification(
        constraints=(ContinuousConstraint(0, 0.5),),
        initial_X=torch.tensor([[0.0]]),
        observed_constraints=torch.tensor([[0.0]]),
    )
    workflow = SafeBOWorkflow(specification, torch.tensor([[0.0], [1.0]]))
    batch = workflow.ask(
        _SafetyModel(), acq_function=_BatchAcquisition(), q=2, num_samples=64, seed=5
    )
    assert batch.shape == (2, 1)
    with pytest.raises(ValueError, match="shape"):
        workflow.tell(batch, torch.zeros(1, 1))
    workflow.tell(batch, batch.clone())
    assert workflow.observed_X.shape == (3, 1)
    assert workflow.observed_constraints.shape == (3, 1)
