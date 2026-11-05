"""Tests for hybrid global-to-local acquisition optimization."""

from unittest.mock import Mock, patch

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends.hybrid import optimize_acqf_hybrid
from robotorchan.optim.constraints import CandidateConstraints
from robotorchan.optim.domains.variable_space import MixedVariableSpace


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
        "robotorchan.optim.backends.hybrid.optimize_acqf_botorch",
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
    assert local_kwargs["constraints"] is constraints
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
        "robotorchan.optim.backends.hybrid.optimize_acqf_botorch",
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
        optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer="unknown",  # type: ignore[arg-type]
        )


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


def test_integer_hybrid_uses_mixed_local_refinement() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 3.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(bounds, integer_dims=(1,))
    global_optimizer = Mock(
        return_value=(torch.tensor([[0.4, 2.0]], dtype=torch.double), torch.tensor(-0.1))
    )

    with patch(
        "robotorchan.optim.backends.hybrid.optimize_acqf_mixed_botorch",
        return_value=(torch.tensor([[0.7, 2.0]], dtype=torch.double), torch.tensor(0.0)),
    ) as local:
        optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer=global_optimizer,
            variable_space=variable_space,
            mixed_fixed_features_list=[{1: 0.0}, {1: 1.0}, {1: 2.0}, {1: 3.0}],
        )

    assert local.called


def test_hybrid_requires_every_structured_dimension_in_local_configs() -> None:
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 3.0, 2.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: [0.0, 1.0, 2.0]},
    )

    with pytest.raises(ValueError, match="every structured dimension"):
        optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer="mixed_ga",
            variable_space=variable_space,
            mixed_fixed_features_list=[{2: 0.0}],
        )


def test_hybrid_runtime_preserves_dtype_and_improves_qbatch() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    global_candidate = torch.tensor([[0.2], [0.4]], dtype=torch.double)

    def global_optimizer(*args, **kwargs):
        value = _Quadratic()(global_candidate.unsqueeze(0)).reshape(())
        return global_candidate, value

    candidate, value = optimize_acqf_hybrid(
        _Quadratic(),
        bounds,
        q=2,
        global_optimizer=global_optimizer,
        local_options={"maxiter": 50},
    )

    assert candidate.shape == torch.Size([2, 1])
    assert candidate.dtype == bounds.dtype
    assert candidate.device == bounds.device
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    initial_value = _Quadratic()(global_candidate.unsqueeze(0)).reshape(())
    assert value >= initial_value


def test_hybrid_real_mixed_ga_global_and_botorch_local_runtime() -> None:
    class _MixedQuadratic(AcquisitionFunction):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)

        def forward(self, X: Tensor) -> Tensor:
            return -((X[..., 0] - 0.7) ** 2).sum(dim=-1)

    bounds = torch.tensor([[0.0, 10.0], [1.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, categorical_values={1: (10.0, 30.0, 90.0)})
    candidate, value = optimize_acqf_hybrid(
        _MixedQuadratic(),
        bounds,
        q=1,
        global_optimizer="mixed_ga",
        global_options={"population_size": 16, "generations": 4},
        local_options={"maxiter": 20},
        variable_space=space,
        mixed_fixed_features_list=[{1: 10.0}, {1: 30.0}, {1: 90.0}],
        seed=37,
    )

    assert candidate.shape == (1, 2)
    assert candidate.dtype == bounds.dtype
    assert float(candidate[0, 1]) in {10.0, 30.0, 90.0}
    assert torch.isfinite(value)
    torch.testing.assert_close(value, _MixedQuadratic()(candidate.unsqueeze(0)).reshape(()))


def test_hybrid_real_mixed_ga_qbatch_with_integer_and_category() -> None:
    class _MixedQuadratic(AcquisitionFunction):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.X_pending = None

        def set_X_pending(self, X_pending: Tensor | None = None) -> None:
            self.X_pending = X_pending

        def forward(self, X: Tensor) -> Tensor:
            return -((X[..., 0] - 0.65) ** 2).sum(dim=-1)

    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 2.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 30.0, 90.0)},
    )
    acquisition = _MixedQuadratic()
    candidate, value = optimize_acqf_hybrid(
        acquisition,
        bounds,
        q=2,
        global_optimizer="mixed_ga",
        global_options={"population_size": 16, "generations": 4},
        local_options={"maxiter": 20},
        variable_space=space,
        mixed_fixed_features_list=[
            {1: integer, 2: category} for integer in (1.0, 2.0) for category in (10.0, 30.0, 90.0)
        ],
        seed=43,
    )

    assert candidate.shape == (2, 3)
    assert candidate.dtype == bounds.dtype
    assert candidate.device == bounds.device
    assert torch.isfinite(candidate).all()
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])
    assert acquisition.X_pending is None
    assert torch.all(torch.isin(candidate[:, 1], candidate.new_tensor([1.0, 2.0])))
    assert torch.all(torch.isin(candidate[:, 2], candidate.new_tensor([10.0, 30.0, 90.0])))
    assert torch.isfinite(value)
    torch.testing.assert_close(value, acquisition(candidate.unsqueeze(0)).reshape(()))


def test_hybrid_real_mixed_pso_qbatch_with_integer_and_category() -> None:
    class _MixedQuadratic(AcquisitionFunction):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.X_pending = None

        def set_X_pending(self, X_pending: Tensor | None = None) -> None:
            self.X_pending = X_pending

        def forward(self, X: Tensor) -> Tensor:
            return -((X[..., 0] - 0.4) ** 2).sum(dim=-1)

    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 2.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 30.0, 90.0)},
    )
    acquisition = _MixedQuadratic()
    candidate, value = optimize_acqf_hybrid(
        acquisition,
        bounds,
        q=2,
        global_optimizer="pso",
        global_options={"swarm_size": 16, "iterations": 4},
        local_options={"maxiter": 20},
        variable_space=space,
        mixed_fixed_features_list=[
            {1: integer, 2: category} for integer in (1.0, 2.0) for category in (10.0, 30.0, 90.0)
        ],
        seed=47,
    )

    assert candidate.shape == (2, 3)
    assert candidate.dtype == bounds.dtype
    assert candidate.device == bounds.device
    assert torch.isfinite(candidate).all()
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])
    assert torch.all(torch.isin(candidate[:, 1], candidate.new_tensor([1.0, 2.0])))
    assert torch.all(torch.isin(candidate[:, 2], candidate.new_tensor([10.0, 30.0, 90.0])))
    assert acquisition.X_pending is None
    assert torch.isfinite(value)
    torch.testing.assert_close(value, acquisition(candidate.unsqueeze(0)).reshape(()))


def test_hybrid_real_mixed_de_qbatch_with_integer_and_category() -> None:
    class _MixedQuadratic(AcquisitionFunction):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.X_pending = None

        def set_X_pending(self, X_pending: Tensor | None = None) -> None:
            self.X_pending = X_pending

        def forward(self, X: Tensor) -> Tensor:
            return -((X[..., 0] - 0.6) ** 2).sum(dim=-1)

    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 2.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 30.0, 90.0)},
    )
    acquisition = _MixedQuadratic()
    candidate, value = optimize_acqf_hybrid(
        acquisition,
        bounds,
        q=2,
        global_optimizer="de",
        global_options={"options": {"maxiter": 4, "popsize": 5}},
        local_options={"maxiter": 20},
        variable_space=space,
        mixed_fixed_features_list=[
            {1: integer, 2: category} for integer in (1.0, 2.0) for category in (10.0, 30.0, 90.0)
        ],
        seed=53,
    )

    assert candidate.shape == (2, 3)
    assert candidate.dtype == bounds.dtype
    assert candidate.device == bounds.device
    assert torch.isfinite(candidate).all()
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])
    assert torch.all(torch.isin(candidate[:, 1], candidate.new_tensor([1.0, 2.0])))
    assert torch.all(torch.isin(candidate[:, 2], candidate.new_tensor([10.0, 30.0, 90.0])))
    assert acquisition.X_pending is None
    assert torch.isfinite(value)
    torch.testing.assert_close(value, acquisition(candidate.unsqueeze(0)).reshape(()))


def test_hybrid_real_mixed_cmaes_qbatch_with_integer_and_category() -> None:
    pytest.importorskip("cmaes")

    class _MixedQuadratic(AcquisitionFunction):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.X_pending = None

        def set_X_pending(self, X_pending: Tensor | None = None) -> None:
            self.X_pending = X_pending

        def forward(self, X: Tensor) -> Tensor:
            return -((X[..., 0] - 0.5) ** 2).sum(dim=-1)

    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 2.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 30.0, 90.0)},
    )
    acquisition = _MixedQuadratic()
    candidate, value = optimize_acqf_hybrid(
        acquisition,
        bounds,
        q=2,
        global_optimizer="cmaes",
        global_options={"population_size": 8, "max_generations": 3},
        local_options={"maxiter": 20},
        variable_space=space,
        mixed_fixed_features_list=[
            {1: integer, 2: category} for integer in (1.0, 2.0) for category in (10.0, 30.0, 90.0)
        ],
        seed=59,
    )

    assert candidate.shape == (2, 3)
    assert candidate.dtype == bounds.dtype
    assert candidate.device == bounds.device
    assert torch.isfinite(candidate).all()
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])
    assert torch.all(torch.isin(candidate[:, 1], candidate.new_tensor([1.0, 2.0])))
    assert torch.all(torch.isin(candidate[:, 2], candidate.new_tensor([10.0, 30.0, 90.0])))
    assert acquisition.X_pending is None
    assert torch.isfinite(value)
    torch.testing.assert_close(value, acquisition(candidate.unsqueeze(0)).reshape(()))


def test_hybrid_rejects_unsupported_restarts_before_global_search() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    global_optimizer = Mock()

    with pytest.raises(ValueError, match="num_restarts > 1"):
        optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer=global_optimizer,
            num_restarts=2,
        )

    global_optimizer.assert_not_called()


def test_hybrid_mixed_rejects_interpoint_nonlinear_before_global_search() -> None:
    bounds = torch.tensor([[0.0, 10.0], [1.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, categorical_values={1: (10.0, 30.0, 90.0)})
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda X: X.sum() - 1.0, False),)
    )
    global_optimizer = Mock()

    with pytest.raises(ValueError, match="inter-point nonlinear constraints"):
        optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=2,
            global_optimizer=global_optimizer,
            constraints=constraints,
            variable_space=space,
            mixed_fixed_features_list=[{1: 10.0}, {1: 30.0}, {1: 90.0}],
        )

    global_optimizer.assert_not_called()


@pytest.mark.parametrize("configs", [None, []])
def test_hybrid_mixed_requires_nonempty_configs_before_global_search(configs) -> None:
    bounds = torch.tensor([[0.0, 10.0], [1.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, categorical_values={1: (10.0, 30.0, 90.0)})
    global_optimizer = Mock()

    with pytest.raises(ValueError, match="requires mixed_fixed_features_list"):
        optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=2,
            global_optimizer=global_optimizer,
            variable_space=space,
            mixed_fixed_features_list=configs,
        )

    global_optimizer.assert_not_called()
