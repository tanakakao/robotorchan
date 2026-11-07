"""Tests for particle-swarm acquisition optimization."""

import pytest
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
    from robotorchan.optim.domains.variable_space import MixedVariableSpace

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


def test_pso_supports_categorical_variable_space() -> None:
    from robotorchan.optim.domains.variable_space import MixedVariableSpace

    bounds = torch.tensor([[0.0, 0.0], [1.0, 90.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(bounds, categorical_values={1: [10.0, 90.0, 30.0]})

    candidate, value = optimize_acqf_pso(
        _Quadratic(),
        bounds,
        q=1,
        variable_space=variable_space,
        swarm_size=24,
        iterations=8,
        seed=19,
    )
    assert candidate.shape == (1, 2)
    assert candidate[0, 1].item() in {10.0, 90.0, 30.0}
    torch.testing.assert_close(value.reshape(()), _Quadratic()(candidate.unsqueeze(0)).reshape(()))


def test_pso_mixed_qbatch_evaluations_preserve_discrete_legality() -> None:
    from robotorchan.optim.domains.variable_space import MixedVariableSpace

    class CheckedMixedAcquisition(AcquisitionFunction):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.calls = 0

        def forward(self, X: Tensor) -> Tensor:
            assert X.shape[-2:] == (2, 3)
            assert torch.all(X[..., 1] == X[..., 1].round())
            assert torch.all((X[..., 1] >= 0) & (X[..., 1] <= 4))
            assert torch.all(torch.isin(X[..., 2], X.new_tensor([10.0, 30.0, 90.0])))
            self.calls += 1
            return -((X[..., 0] - 0.6) ** 2).sum(dim=-1)

    bounds = torch.tensor([[0.0, 0.0, 10.0], [1.0, 4.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 30.0, 90.0)},
    )
    acquisition = CheckedMixedAcquisition()
    candidate, value = optimize_acqf_pso(
        acquisition,
        bounds,
        q=2,
        swarm_size=16,
        iterations=5,
        seed=17,
        variable_space=space,
    )

    assert acquisition.calls == 7
    assert candidate.shape == (2, 3)
    assert value.ndim == 0


def test_pso_rejects_infeasible_result() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([1.0]), 2.0),)
    )

    with pytest.raises(ValueError, match="PSO found no feasible candidate"):
        optimize_acqf_pso(
            _Quadratic(),
            bounds,
            q=1,
            swarm_size=16,
            iterations=4,
            seed=11,
            constraints=constraints,
        )
