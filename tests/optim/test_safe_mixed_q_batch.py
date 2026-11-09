"""Focused tests for Mixed q-batch Safe BO."""

import pytest
import torch

from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.optim.safe import optimize_safe_acqf
from robotorchan.optim.safe_expansion import propose_safe_expansion
from robotorchan.optim.safe_workflow import SafeBOWorkflow
from robotorchan.semantics import ContinuousConstraint, SafetySpecification


class _Posterior:
    def __init__(self, X: torch.Tensor) -> None:
        self.mean = X[..., :1]
        self.variance = torch.full_like(self.mean, 0.0001)


class _SafetyModel:
    def posterior(self, X: torch.Tensor) -> _Posterior:
        return _Posterior(X)


class _BatchAcquisition:
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X[..., 0].sum(dim=-1)


@pytest.fixture
def space() -> MixedVariableSpace:
    return MixedVariableSpace(
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 3.0, 4.0]]),
        integer_dims=(1,),
        categorical_values={2: (0.0, 2.0, 4.0)},
    )


@pytest.mark.parametrize("mode", ["objective", "expansion"])
def test_mixed_batch_is_legal_and_posterior_safe(space: MixedVariableSpace, mode: str) -> None:
    kwargs = {
        "bounds": space.bounds,
        "constraints": (ContinuousConstraint(0, 0.6),),
        "variable_space": space,
        "q": 3,
        "num_samples": 128,
        "seed": 9,
    }
    if mode == "objective":
        batch, _ = optimize_safe_acqf(_BatchAcquisition(), _SafetyModel(), **kwargs)
    else:
        batch, _ = propose_safe_expansion(_SafetyModel(), **kwargs)
    assert batch.shape == (3, 3)
    assert bool((batch[:, 0] + 0.02 <= 0.6).all())
    assert bool((batch[:, 1] == batch[:, 1].round()).all())
    assert bool(torch.isin(batch[:, 2], torch.tensor([0.0, 2.0, 4.0])).all())


def test_mixed_batch_no_safe_candidates_raises(space: MixedVariableSpace) -> None:
    with pytest.raises(RuntimeError, match="No posterior-safe candidate batches"):
        optimize_safe_acqf(
            _BatchAcquisition(),
            _SafetyModel(),
            space.bounds,
            (ContinuousConstraint(0, -1.0),),
            variable_space=space,
            q=2,
            num_samples=32,
            seed=9,
        )


def test_mixed_batch_workflow_records_all_observations(space: MixedVariableSpace) -> None:
    specification = SafetySpecification(
        constraints=(ContinuousConstraint(0, 0.6),),
        initial_X=torch.tensor([[0.0, 1.0, 2.0]]),
        observed_constraints=torch.tensor([[0.0]]),
    )
    workflow = SafeBOWorkflow(specification, space.bounds, variable_space=space)
    batch = workflow.ask(
        _SafetyModel(), acq_function=_BatchAcquisition(), q=2, num_samples=64, seed=9
    )
    workflow.tell(batch, batch[:, :1])
    assert workflow.observed_X.shape == (3, 3)
    assert workflow.observed_constraints.shape == (3, 1)
