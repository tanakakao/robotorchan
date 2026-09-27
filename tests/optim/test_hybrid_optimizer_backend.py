"""Tests for hybrid global-to-local acquisition optimization."""

from unittest.mock import Mock, patch

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends.hybrid import optimize_acqf_hybrid
from robotorchan.optim.constraints import CandidateConstraints
from robotorchan.optim.variable_space import MixedVariableSpace


class _Quadratic(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        return -((X - 0.73) ** 2).sum(dim=(-2, -1))


def test_hybrid_improves_or_preserves_global_candidate() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    global_candidate = torch.tensor([[0.4]], dtype=torch.double)

    def global_optimizer(*args, **kwargs):
        return global_candidate, torch.tensor(-0.1, dtype=torch.double)

    candidate, value = optimize_acqf_hybrid(
        _Quadratic(),
        bounds,
        q=1,
        global_optimizer=global_optimizer,
        local_options={"maxiter": 50},
    )

    assert abs(float(candidate[0, 0]) - 0.73) < 1e-3
    assert float(value) >= float(_Quadratic()(global_candidate.unsqueeze(0)))


def test_hybrid_forwards_constraints_to_global_and_local_stages() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([-1.0]), -0.4),)
    )
    global_optimizer = Mock(
        return_value=(torch.tensor([[0.3]], dtype=torch.double), torch.tensor(-0.1))
    )
    local_candidate = torch.tensor([[0.35]], dtype=torch.double)
    local_value = torch.tensor(-0.01, dtype=torch.double)

    with patch(
        "robotorchan.optim.backends.hybrid.botorch_optimize_acqf",
        return_value=(local_candidate, local_value),
    ) as local_optimizer:
        candidate, value = optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer=global_optimizer,
            constraints=constraints,
            seed=5,
        )

    assert torch.equal(candidate, local_candidate)
    assert torch.equal(value, local_value)
    assert global_optimizer.call_args.kwargs["constraints"] is constraints
    local_kwargs = local_optimizer.call_args.kwargs
    assert local_kwargs["inequality_constraints"] == list(constraints.inequality_constraints)
    expected_initial_conditions = torch.tensor([[[0.3]]], dtype=bounds.dtype)
    assert torch.equal(local_kwargs["batch_initial_conditions"], expected_initial_conditions)


def test_hybrid_sets_batch_limit_for_nonlinear_constraints() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: 0.5 - x[..., 0], True),)
    )
    global_optimizer = Mock(
        return_value=(torch.tensor([[0.3]], dtype=torch.double), torch.tensor(-0.1))
    )

    with patch(
        "robotorchan.optim.backends.hybrid.botorch_optimize_acqf",
        return_value=(torch.tensor([[0.3]]), torch.tensor(-0.1)),
    ) as local_optimizer:
        optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer=global_optimizer,
            constraints=constraints,
        )

    assert local_optimizer.call_args.kwargs["options"]["batch_limit"] == 1


def test_hybrid_rejects_duplicate_restarts() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    global_optimizer = Mock(
        return_value=(torch.tensor([[0.3]], dtype=torch.double), torch.tensor(-0.1))
    )

    with pytest.raises(ValueError, match="num_restarts > 1"):
        optimize_acqf_hybrid(
            _Quadratic(), bounds, q=1, global_optimizer=global_optimizer, num_restarts=2
        )


def test_hybrid_rejects_unknown_global_optimizer() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="global_optimizer"):
        optimize_acqf_hybrid(_Quadratic(), bounds, q=1, global_optimizer="unknown")  # type: ignore[arg-type]


def test_hybrid_forwards_mixed_space_to_global_and_mixed_local_stage() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(
        bounds,
        categorical_values={1: [0.0, 1.0, 2.0]},
    )
    global_optimizer = Mock(
        return_value=(torch.tensor([[0.4, 1.0]], dtype=torch.double), torch.tensor(-0.1))
    )
    fixed_features_list = [{1: 0.0}, {1: 1.0}, {1: 2.0}]
    local_candidate = torch.tensor([[0.7, 1.0]], dtype=torch.double)
    local_value = torch.tensor(-0.01, dtype=torch.double)

    with patch(
        "robotorchan.optim.backends.hybrid.optimize_acqf_mixed_botorch",
        return_value=(local_candidate, local_value),
    ) as local_optimizer:
        candidate, value = optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer=global_optimizer,
            variable_space=variable_space,
            mixed_fixed_features_list=fixed_features_list,
        )

    assert torch.equal(candidate, local_candidate)
    assert torch.equal(value, local_value)
    assert global_optimizer.call_args.kwargs["variable_space"] is variable_space
    local_kwargs = local_optimizer.call_args.kwargs
    assert local_kwargs["fixed_features_list"] == fixed_features_list
    assert torch.equal(
        local_kwargs["batch_initial_conditions"],
        torch.tensor([[[0.4, 1.0]]], dtype=torch.double),
    )


def test_categorical_hybrid_requires_mixed_local_enumeration() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(
        bounds,
        categorical_values={1: [0.0, 1.0, 2.0]},
    )

    with pytest.raises(ValueError, match="mixed_fixed_features_list"):
        optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer="mixed_ga",
            variable_space=variable_space,
        )


def test_hybrid_merges_common_fixed_features_into_mixed_local_configs() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(bounds, categorical_values={1: [0.0, 2.0]})
    global_optimizer = Mock(
        return_value=(torch.tensor([[0.4, 2.0]], dtype=torch.double), torch.tensor(1.0))
    )
    local_candidate = torch.tensor([[0.25, 2.0]], dtype=torch.double)

    with patch(
        "robotorchan.optim.backends.hybrid.optimize_acqf_mixed_botorch",
        return_value=(local_candidate, torch.tensor(1.0)),
    ) as local:
        optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer=global_optimizer,
            variable_space=variable_space,
            fixed_features={0: 0.25},
            mixed_fixed_features_list=[{1: 0.0}, {1: 2.0}],
        )

    assert local.call_args.kwargs["fixed_features_list"] == [
        {1: 0.0, 0: 0.25},
        {1: 2.0, 0: 0.25},
    ]


def test_hybrid_rejects_conflicting_common_and_mixed_fixed_features() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(bounds, categorical_values={1: [0.0, 2.0]})
    global_optimizer = Mock(
        return_value=(torch.tensor([[0.4, 2.0]], dtype=torch.double), torch.tensor(1.0))
    )

    with pytest.raises(ValueError, match="conflicts"):
        optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer=global_optimizer,
            variable_space=variable_space,
            fixed_features={1: 2.0},
            mixed_fixed_features_list=[{1: 0.0}, {1: 2.0}],
        )
