"""Tests for mixed-variable random acquisition optimization."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import MixedVariableSpace, optimize_acqf


class SumAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(-1, -2))


def test_mixed_random_returns_only_legal_values() -> None:
    bounds = torch.tensor([[0.0, 1.2, 10.0], [1.0, 4.8, 30.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 20.0, 30.0)},
    )

    candidate, _ = optimize_acqf(
        SumAcquisition(),
        bounds,
        q=3,
        optimizer="random",
        raw_samples=256,
        seed=17,
        variable_space=space,
    )

    assert candidate.shape == (3, 3)
    assert torch.all(candidate[:, 1] == candidate[:, 1].round())
    assert torch.all((candidate[:, 1] >= 2) & (candidate[:, 1] <= 4))
    assert set(candidate[:, 2].tolist()) <= {10.0, 20.0, 30.0}


def test_mixed_random_is_reproducible() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 3.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, integer_dims=(1,))

    first, first_value = optimize_acqf(
        SumAcquisition(),
        bounds,
        q=2,
        optimizer="random",
        raw_samples=64,
        seed=9,
        variable_space=space,
    )
    second, second_value = optimize_acqf(
        SumAcquisition(),
        bounds,
        q=2,
        optimizer="random",
        raw_samples=64,
        seed=9,
        variable_space=space,
    )

    assert torch.equal(first, second)
    assert torch.equal(first_value, second_value)


def test_mixed_random_validates_fixed_features() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 3.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, integer_dims=(1,))

    try:
        optimize_acqf(
            SumAcquisition(),
            bounds,
            q=1,
            optimizer="random",
            raw_samples=16,
            variable_space=space,
            fixed_features={1: 1.5},
        )
    except ValueError as exc:
        assert "integer value" in str(exc)
    else:
        raise AssertionError("Illegal mixed fixed feature should be rejected.")


def test_mixed_sobol_is_explicitly_deferred() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 3.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, integer_dims=(1,))

    try:
        optimize_acqf(
            SumAcquisition(),
            bounds,
            q=1,
            optimizer="sobol",
            raw_samples=16,
            variable_space=space,
        )
    except NotImplementedError as exc:
        assert "Mixed-variable Sobol" in str(exc)
    else:
        raise AssertionError("Phase 4 must not silently round Sobol mixed variables.")
