"""Focused mixed-variable Safe BO regression tests."""

import pytest
import torch

from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.optim.safe import optimize_safe_acqf
from robotorchan.optim.safe_expansion import propose_safe_expansion
from robotorchan.semantics import ContinuousConstraint


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


@pytest.fixture
def mixed_space() -> MixedVariableSpace:
    return MixedVariableSpace(
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 3.0, 4.0]]),
        integer_dims=(1,),
        categorical_values={2: (0.0, 2.0, 4.0)},
    )


@pytest.mark.parametrize("expansion", [False, True])
def test_mixed_safe_candidates_respect_domains_and_safety(
    mixed_space: MixedVariableSpace, expansion: bool
) -> None:
    kwargs = {
        "bounds": mixed_space.bounds,
        "constraints": (ContinuousConstraint(0, 0.6),),
        "variable_space": mixed_space,
        "num_samples": 128,
        "seed": 7,
    }
    if expansion:
        candidate, _ = propose_safe_expansion(_SafetyModel(), **kwargs)
    else:
        candidate, _ = optimize_safe_acqf(_Acquisition(), _SafetyModel(), **kwargs)
    assert candidate.shape == (1, 3)
    assert candidate[0, 0] <= 0.6
    assert candidate[0, 1] == candidate[0, 1].round()
    assert candidate[0, 2].item() in (0.0, 2.0, 4.0)


def test_mixed_empty_safe_pool_raises(mixed_space: MixedVariableSpace) -> None:
    with pytest.raises(RuntimeError, match="No posterior-safe candidates"):
        optimize_safe_acqf(
            _Acquisition(),
            _SafetyModel(),
            mixed_space.bounds,
            (ContinuousConstraint(0, -1.0),),
            variable_space=mixed_space,
            num_samples=32,
            seed=7,
        )
