"""Tests for particle-swarm acquisition optimization."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends import optimize_acqf_pso
from robotorchan.optim.constraints import CandidateConstraints


class _Quadratic(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        return -((X - 0.73) ** 2).sum(dim=(-1, -2))


def test_pso_finds_continuous_optimum() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidate, value = optimize_acqf_pso(
        _Quadratic(), bounds, q=1, swarm_size=64, iterations=60, seed=3
    )

    assert candidate.shape == (1, 1)
    assert torch.allclose(candidate, torch.tensor([[0.73]], dtype=torch.double), atol=2e-2)
    assert float(value) > -1e-3


def test_pso_is_reproducible_and_preserves_dtype() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    first = optimize_acqf_pso(_Quadratic(), bounds, q=2, swarm_size=24, iterations=10, seed=8)
    second = optimize_acqf_pso(_Quadratic(), bounds, q=2, swarm_size=24, iterations=10, seed=8)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[0].dtype == bounds.dtype


def test_pso_applies_candidate_constraint_penalty() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (torch.tensor([0]), torch.tensor([-1.0], dtype=torch.double), -0.4),
        )
    )

    candidate, value = optimize_acqf_pso(
        _Quadratic(),
        bounds,
        q=1,
        swarm_size=96,
        iterations=80,
        seed=11,
        constraints=constraints,
    )

    assert float(candidate[0, 0]) <= 0.41
    expected = _Quadratic()(candidate.unsqueeze(0)).reshape(())
    assert torch.equal(value, expected)



def test_pso_supports_continuous_integer_space() -> None:
    from robotorchan.optim.variable_space import MixedVariableSpace

    bounds = torch.tensor([[0.0, 0.0], [1.0, 4.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(bounds, integer_dims=(1,))

    candidate, _ = optimize_acqf_pso(
        _Quadratic(),
        bounds,
        q=1,
        swarm_size=64,
        iterations=50,
        seed=19,
        variable_space=variable_space,
    )

    assert candidate[0, 1] == candidate[0, 1].round()
    assert 0.0 <= float(candidate[0, 1]) <= 4.0


def test_pso_rejects_categorical_variable_space() -> None:
    from robotorchan.optim.variable_space import MixedVariableSpace

    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(bounds, categorical_values={1: [0.0, 1.0, 2.0]})

    try:
        optimize_acqf_pso(_Quadratic(), bounds, q=1, variable_space=variable_space)
    except ValueError as error:
        assert "does not support categorical" in str(error)
    else:
        raise AssertionError("Expected categorical PSO to be rejected.")
