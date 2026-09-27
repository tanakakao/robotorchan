"""Tests for mixed-variable genetic-algorithm acquisition optimization."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim import MixedVariableSpace
from robotorchan.optim.backends import optimize_acqf_mixed_ga
from robotorchan.optim.constraints import CandidateConstraints


class _MixedTargetAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        target = torch.tensor([0.7, 3.0, 20.0], device=X.device, dtype=X.dtype)
        return -((X - target) ** 2).sum(dim=(-2, -1))


def test_mixed_ga_respects_integer_and_categorical_domains() -> None:
    bounds = torch.tensor([[0.0, 0.0, 10.0], [1.0, 5.0, 30.0]], dtype=torch.double)
    candidates, value = optimize_acqf_mixed_ga(
        _MixedTargetAcquisition(),
        bounds,
        q=1,
        integer_dims=[1],
        categorical_values={2: [10.0, 20.0, 30.0]},
        population_size=80,
        generations=50,
        seed=7,
    )

    assert candidates.shape == torch.Size([1, 3])
    assert candidates.dtype == bounds.dtype
    assert candidates[0, 1] == candidates[0, 1].round()
    assert float(candidates[0, 2]) in {10.0, 20.0, 30.0}
    assert abs(float(candidates[0, 0]) - 0.7) < 0.1
    assert value.ndim == 0


def test_mixed_ga_supports_joint_q_batch_and_reproducibility() -> None:
    bounds = torch.tensor([[0.0, 0.0, 10.0], [1.0, 5.0, 30.0]], dtype=torch.double)
    kwargs = {
        "integer_dims": [1],
        "categorical_values": {2: [10.0, 20.0, 30.0]},
        "population_size": 32,
        "generations": 10,
        "seed": 11,
    }
    first, first_value = optimize_acqf_mixed_ga(_MixedTargetAcquisition(), bounds, q=2, **kwargs)
    second, second_value = optimize_acqf_mixed_ga(_MixedTargetAcquisition(), bounds, q=2, **kwargs)

    assert first.shape == torch.Size([2, 3])
    assert torch.equal(first, second)
    assert torch.equal(first_value, second_value)
    assert torch.equal(first[:, 1], first[:, 1].round())
    assert all(float(value) in {10.0, 20.0, 30.0} for value in first[:, 2])


def test_mixed_ga_rejects_overlapping_structured_dimensions() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 3.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="both integer and categorical"):
        optimize_acqf_mixed_ga(
            _MixedTargetAcquisition(),
            bounds,
            q=1,
            integer_dims=[1],
            categorical_values={1: [0.0, 1.0]},
        )


def test_mixed_ga_rejects_out_of_bounds_categories() -> None:
    bounds = torch.tensor([[0.0, 0.0, 10.0], [1.0, 5.0, 30.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="must lie within bounds"):
        optimize_acqf_mixed_ga(
            _MixedTargetAcquisition(),
            bounds,
            q=1,
            categorical_values={2: [40.0]},
        )


def test_mixed_ga_accepts_shared_variable_space_and_fixed_features() -> None:
    bounds = torch.tensor([[0.0, 0.0, 10.0], [1.0, 5.0, 30.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 20.0, 30.0)},
    )

    candidates, _ = optimize_acqf_mixed_ga(
        _MixedTargetAcquisition(),
        bounds,
        q=2,
        variable_space=space,
        fixed_features={2: 20.0},
        population_size=32,
        generations=10,
        seed=19,
    )

    assert torch.equal(candidates[:, 2], torch.full((2,), 20.0, dtype=torch.double))
    assert torch.equal(candidates[:, 1], candidates[:, 1].round())


def test_mixed_ga_rejects_duplicate_variable_contracts() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 3.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, integer_dims=(1,))

    with pytest.raises(ValueError, match="variable_space or integer_dims"):
        optimize_acqf_mixed_ga(
            _MixedTargetAcquisition(),
            bounds,
            q=1,
            variable_space=space,
            integer_dims=(1,),
        )


def test_mixed_ga_categorical_crossover_inherits_parent_values() -> None:
    from robotorchan.optim.backends.mixed_genetic_algorithm import _crossover

    parents_a = torch.tensor([[0.2, 10.0], [0.3, 20.0]], dtype=torch.double)
    parents_b = torch.tensor([[0.8, 30.0], [0.7, 10.0]], dtype=torch.double)
    generator = torch.Generator().manual_seed(5)

    offspring = _crossover(
        parents_a,
        parents_b,
        crossover_rate=1.0,
        generator=generator,
        integer_dims=(),
        categorical_dims=(1,),
    )

    assert torch.equal(offspring[:, 1], parents_b[:, 1])


class _IncreasingAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        return X[..., 0].sum(dim=-1)


def test_mixed_ga_tracks_best_candidate_across_generations_by_deb_order() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0]),
                torch.tensor([-1.0], dtype=torch.double),
                -0.25,
            ),
        )
    )

    candidate, _ = optimize_acqf_mixed_ga(
        _IncreasingAcquisition(),
        bounds,
        q=1,
        population_size=24,
        generations=8,
        constraints=constraints,
        seed=23,
    )

    assert candidate[0, 0] <= 0.25 + 1e-8
