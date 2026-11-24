"""Optimizer factory contracts and representative candidate generation."""

from typing import get_args

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import (
    OPTIMIZER_CAPABILITIES,
    CandidateConstraints,
    MixedVariableSpace,
    OptimizerName,
    create_optimizer,
    list_optimizers,
    optimize_acqf,
)
from robotorchan.optim import backends


class _SumAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.sum(dim=(-1, -2))


def test_all_dispatch_and_specialist_names_resolve() -> None:
    assert set(get_args(OptimizerName)) == set(OPTIMIZER_CAPABILITIES)
    names = list_optimizers()
    assert names == tuple(sorted(set(names)))
    for name in OPTIMIZER_CAPABILITIES:
        configured = create_optimizer(name)
        assert configured.func is optimize_acqf
        assert configured.keywords == {"optimizer": name}
    for name in backends.__all__:
        if name != "TorchOptimizerName":
            assert create_optimizer(name).func is getattr(backends, name)
    for name in ("optimize_mixed_one_shot_acqf", "optimize_safe_acqf", "ContextualOptimizer"):
        assert name in names


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="Unknown optimizer"):
        create_optimizer("unknown")
    with pytest.raises(TypeError):
        create_optimizer("sobol", typo=True)
    with pytest.raises(TypeError, match="only once"):
        create_optimizer("sobol", optimizer="ga")


def test_custom_callable_and_overrides() -> None:
    def custom(*, value: int) -> int:
        return value

    configured = create_optimizer(custom, value=2)
    assert configured() == 2
    assert configured(value=3) == 3


def test_scalar_dispatch_matches_direct_q_batch() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acq = _SumAcquisition()
    configured = create_optimizer("sobol", seed=4, raw_samples=32)
    actual = configured(acq, bounds, q=2)
    expected = optimize_acqf(acq, bounds, q=2, optimizer="sobol", seed=4, raw_samples=32)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference)
    assert actual[0].shape == (2, 1)


def test_mixed_constraints_and_q_batch() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 4.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, integer_dims=(1,))
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([1.0]), 0.2),)
    )
    configured = create_optimizer("sobol", seed=7, raw_samples=128)
    candidates, value = configured(
        _SumAcquisition(), bounds, q=2, variable_space=space, constraints=constraints
    )
    assert candidates.shape == (2, 2)
    assert torch.all(candidates[:, 0] >= 0.2 - 1e-6)
    torch.testing.assert_close(candidates[:, 1], candidates[:, 1].round())
    assert torch.isfinite(value).all()


def test_vector_optimizer_preserves_pareto_contract() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    def objective(X: torch.Tensor) -> torch.Tensor:
        return torch.stack([-X[:, 0].square(), -(X[:, 0] - 1).square()], dim=-1)

    configured = create_optimizer(
        "optimize_vector_nsga2", population_size=16, generations=3, seed=4
    )
    candidates, values = configured(objective, bounds)
    assert candidates.ndim == 2
    assert values.shape == (candidates.shape[0], 2)
    torch.testing.assert_close(values, objective(candidates))
