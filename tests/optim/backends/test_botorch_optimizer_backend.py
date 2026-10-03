from unittest.mock import patch

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions.errors import UnsupportedError

from robotorchan.optim.backends.botorch import (
    optimize_acqf_botorch,
    optimize_acqf_mixed_botorch,
)
from robotorchan.optim.constraints import CandidateConstraints


class _DummyAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)
        self.X_pending = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(-1, -2))


def test_botorch_backend_preserves_native_arguments() -> None:
    acq = _DummyAcquisition()
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    expected_x = torch.tensor([[0.25, 0.75]])
    expected_value = torch.tensor(1.0)

    with patch(
        "robotorchan.optim.backends.botorch.botorch_optimize_acqf",
        return_value=(expected_x, expected_value),
    ) as mocked:
        x, value = optimize_acqf_botorch(
            acq,
            bounds,
            q=1,
            num_restarts=4,
            raw_samples=32,
            fixed_features={1: 0.75},
            sequential=True,
        )

    assert torch.equal(x, expected_x)
    assert torch.equal(value, expected_value)
    kwargs = mocked.call_args.kwargs
    assert kwargs["acq_function"] is acq
    assert torch.equal(kwargs["bounds"], bounds)
    assert kwargs["q"] == 1
    assert kwargs["num_restarts"] == 4
    assert kwargs["raw_samples"] == 32
    assert kwargs["fixed_features"] == {1: 0.75}
    assert kwargs["sequential"] is True


def test_botorch_backend_preserves_fixed_feature_when_sequential() -> None:
    acq = _DummyAcquisition()
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    candidates, values = optimize_acqf_botorch(
        acq,
        bounds,
        q=2,
        num_restarts=3,
        raw_samples=32,
        fixed_features={1: 0.25},
        sequential=True,
    )

    assert candidates.shape == torch.Size([2, 2])
    assert values.shape == torch.Size([2])
    assert torch.isfinite(candidates).all()
    assert torch.all(candidates[:, 1] == 0.25)


def test_botorch_backend_applies_nonlinear_batch_limit_without_mutating_options() -> None:
    acq = _DummyAcquisition()
    bounds = torch.tensor([[0.0], [1.0]])
    initial = torch.full((2, 1, 1), 0.5)
    options = {"maxiter": 20}
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[..., 0] - 0.1, True),)
    )

    with patch(
        "robotorchan.optim.backends.botorch.botorch_optimize_acqf",
        return_value=(torch.tensor([[0.5]]), torch.tensor(0.5)),
    ) as mocked:
        optimize_acqf_botorch(
            acq,
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
            options=options,
            constraints=constraints,
            batch_initial_conditions=initial,
        )

    assert options == {"maxiter": 20}
    assert mocked.call_args.kwargs["options"] == {"maxiter": 20, "batch_limit": 1}


def test_mixed_backend_rejects_interpoint_nonlinear_constraint() -> None:
    acq = _DummyAcquisition()
    bounds = torch.tensor([[0.0], [1.0]])
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x.sum() - 0.1, False),)
    )

    try:
        optimize_acqf_mixed_botorch(
            acq,
            bounds,
            q=2,
            num_restarts=2,
            fixed_features_list=[{0: 0.0}, {0: 1.0}],
            raw_samples=8,
            constraints=constraints,
            batch_initial_conditions=torch.full((2, 2, 1), 0.5),
        )
    except ValueError as exc:
        assert "inter-point nonlinear constraints" in str(exc)
    else:
        raise AssertionError("Expected mixed backend to reject inter-point constraints.")


def test_botorch_backend_solves_q1_with_intrapoint_nonlinear_constraint() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: 0.4 - x[0], True),)
    )
    initial = torch.tensor([[[0.1]], [[0.2]], [[0.3]]], dtype=torch.double)

    candidate, value = optimize_acqf_botorch(
        _DummyAcquisition(),
        bounds,
        q=1,
        num_restarts=3,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial,
    )

    assert candidate.shape == torch.Size([1, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert 0.0 <= candidate[0, 0] <= 0.4 + 1e-6


def test_botorch_backend_solves_joint_q_with_intrapoint_nonlinear_constraint() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: 0.6 - x[0], True),)
    )
    initial = torch.tensor(
        [
            [[0.2], [0.3]],
            [[0.3], [0.4]],
            [[0.4], [0.5]],
        ],
        dtype=torch.double,
    )

    candidate, value = optimize_acqf_botorch(
        _DummyAcquisition(),
        bounds,
        q=2,
        num_restarts=3,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial,
    )

    assert candidate.shape == torch.Size([2, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= 0.6 + 1e-6)


def test_botorch_backend_enforces_multiple_intrapoint_nonlinear_constraints() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=(
            (lambda x: 0.7 - x[0], True),
            (lambda x: 0.8 - x[1], True),
            (lambda x: 1.0 - x.square().sum(), True),
        )
    )
    initial = torch.tensor(
        [
            [[0.2, 0.2]],
            [[0.4, 0.4]],
            [[0.6, 0.5]],
        ],
        dtype=torch.double,
    )

    candidate, value = optimize_acqf_botorch(
        _DummyAcquisition(),
        bounds,
        q=1,
        num_restarts=3,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial,
    )

    x = candidate[0]
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert x[0] <= 0.7 + 1e-6
    assert x[1] <= 0.8 + 1e-6
    assert x.square().sum() <= 1.0 + 1e-6
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])


def test_botorch_backend_forwards_linear_constraint_kinds_unchanged() -> None:
    acq = _DummyAcquisition()
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    inequality = (torch.tensor([0]), torch.tensor([1.0]), 0.2)
    equality = (torch.tensor([1]), torch.tensor([1.0]), 0.5)
    constraints = CandidateConstraints(
        inequality_constraints=(inequality,),
        equality_constraints=(equality,),
    )

    with patch(
        "robotorchan.optim.backends.botorch.botorch_optimize_acqf",
        return_value=(torch.tensor([[0.5, 0.5]]), torch.tensor(1.0)),
    ) as mocked:
        optimize_acqf_botorch(
            acq,
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
            constraints=constraints,
        )

    kwargs = mocked.call_args.kwargs
    assert kwargs["inequality_constraints"] == [inequality]
    assert kwargs["equality_constraints"] == [equality]


def test_botorch_backend_requires_initial_conditions_for_nonlinear_constraints() -> None:
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[..., 0] - 0.1, True),)
    )

    with pytest.raises(ValueError, match="feasible batch_initial_conditions"):
        optimize_acqf_botorch(
            _DummyAcquisition(),
            torch.tensor([[0.0], [1.0]]),
            q=1,
            num_restarts=2,
            raw_samples=8,
            constraints=constraints,
        )


def test_mixed_backend_forwards_supported_constraints_unchanged() -> None:
    acq = _DummyAcquisition()
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    inequality = (torch.tensor([0]), torch.tensor([1.0]), 0.2)
    equality = (torch.tensor([0]), torch.tensor([1.0]), 0.5)
    nonlinear = (lambda x: x[..., 0] - 0.1, True)
    constraints = CandidateConstraints(
        inequality_constraints=(inequality,),
        equality_constraints=(equality,),
        nonlinear_inequality_constraints=(nonlinear,),
    )
    initial = torch.full((2, 1, 2), 0.5)

    with patch(
        "robotorchan.optim.backends.botorch.botorch_optimize_acqf_mixed",
        return_value=(torch.tensor([[0.5, 1.0]]), torch.tensor(1.5)),
    ) as mocked:
        optimize_acqf_mixed_botorch(
            acq,
            bounds,
            q=1,
            num_restarts=2,
            fixed_features_list=[{1: 0.0}, {1: 1.0}],
            raw_samples=8,
            constraints=constraints,
            batch_initial_conditions=initial,
        )

    kwargs = mocked.call_args.kwargs
    assert kwargs["inequality_constraints"] == [inequality]
    assert kwargs["equality_constraints"] == [equality]
    assert kwargs["nonlinear_inequality_constraints"] == [nonlinear]
    assert kwargs["options"]["batch_limit"] == 1


def test_mixed_backend_rejects_q_interpoint_nonlinear_before_botorch_call() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: 0.8 - x[:, 0].sum(), False),)
    )

    with (
        patch("robotorchan.optim.backends.botorch.botorch_optimize_acqf_mixed") as optimize,
        pytest.raises(ValueError, match="inter-point nonlinear"),
    ):
        optimize_acqf_mixed_botorch(
            _DummyAcquisition(),
            bounds,
            q=2,
            num_restarts=2,
            fixed_features_list=[{}],
            raw_samples=8,
            constraints=constraints,
            batch_initial_conditions=torch.tensor(
                [[[0.2], [0.3]], [[0.3], [0.2]]],
                dtype=torch.double,
            ),
        )

    optimize.assert_not_called()


def test_botorch_backend_solves_joint_q_batch_with_interpoint_linear_constraint() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    indices = torch.tensor([[0, 0], [1, 0]])
    coefficients = torch.tensor([1.0, 1.0], dtype=torch.double)
    constraints = CandidateConstraints(inequality_constraints=((indices, coefficients, 1.2),))

    candidate, value = optimize_acqf_botorch(
        _DummyAcquisition(),
        bounds,
        q=2,
        num_restarts=4,
        raw_samples=32,
        constraints=constraints,
    )

    assert candidate.shape == torch.Size([2, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert candidate.sum() >= 1.2 - 1e-6
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])


def test_botorch_backend_solves_joint_q_with_interpoint_nonlinear_constraint() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    min_distance = 0.35

    def separation(X: torch.Tensor) -> torch.Tensor:
        return (X[0] - X[1]).square().sum() - min_distance**2

    constraints = CandidateConstraints(nonlinear_inequality_constraints=((separation, False),))
    initial = torch.tensor(
        [
            [[0.1], [0.6]],
            [[0.2], [0.7]],
            [[0.3], [0.8]],
        ],
        dtype=torch.double,
    )

    candidate, value = optimize_acqf_botorch(
        _DummyAcquisition(),
        bounds,
        q=2,
        num_restarts=3,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial,
    )

    assert candidate.shape == torch.Size([2, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert separation(candidate) >= -1e-6
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])


def test_botorch_backend_combines_intra_and_interpoint_nonlinear_constraints() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    def upper_bound(x: torch.Tensor) -> torch.Tensor:
        return 0.85 - x[0]

    def minimum_spread(X: torch.Tensor) -> torch.Tensor:
        return (X[0] - X[1]).square().sum() - 0.25**2

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=(
            (upper_bound, True),
            (minimum_spread, False),
        )
    )
    initial = torch.tensor(
        [
            [[0.1], [0.5]],
            [[0.2], [0.6]],
            [[0.3], [0.7]],
        ],
        dtype=torch.double,
    )

    candidate, value = optimize_acqf_botorch(
        _DummyAcquisition(),
        bounds,
        q=2,
        num_restarts=3,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial,
    )

    assert candidate.shape == torch.Size([2, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert torch.all(candidate[:, 0] <= 0.85 + 1e-6)
    assert minimum_spread(candidate) >= -1e-6


def test_botorch_backend_interpoint_callable_receives_joint_q_batch() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    seen_shapes: list[torch.Size] = []

    def joint_constraint(X: torch.Tensor) -> torch.Tensor:
        seen_shapes.append(X.shape)
        return 1.5 - X[:, 0].sum()

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((joint_constraint, False),)
    )
    initial = torch.tensor(
        [
            [[0.2], [0.4], [0.6]],
            [[0.1], [0.5], [0.7]],
        ],
        dtype=torch.double,
    )

    candidate, _ = optimize_acqf_botorch(
        _DummyAcquisition(),
        bounds,
        q=3,
        num_restarts=2,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial,
    )

    assert candidate.shape == torch.Size([3, 1])
    assert seen_shapes
    assert all(shape == torch.Size([3, 1]) for shape in seen_shapes)
    assert joint_constraint(candidate) >= -1e-6


def test_botorch_backend_forwards_joint_qbatch_initial_conditions() -> None:
    initial = torch.tensor(
        [
            [[0.1], [0.9]],
            [[0.2], [0.8]],
            [[0.3], [0.7]],
        ],
        dtype=torch.double,
    )

    with patch(
        "robotorchan.optim.backends.botorch.botorch_optimize_acqf",
        return_value=(torch.tensor([[0.2], [0.8]]), torch.tensor(1.0)),
    ) as mocked:
        optimize_acqf_botorch(
            _DummyAcquisition(),
            torch.tensor([[0.0], [1.0]], dtype=torch.double),
            q=2,
            num_restarts=3,
            raw_samples=None,
            batch_initial_conditions=initial,
        )

    kwargs = mocked.call_args.kwargs
    assert kwargs["q"] == 2
    assert kwargs["num_restarts"] == 3
    assert kwargs["raw_samples"] is None
    assert kwargs["batch_initial_conditions"] is initial
    assert kwargs["sequential"] is False


def test_botorch_backend_delegates_sequential_initialization_semantics() -> None:
    initial = torch.tensor(
        [
            [[0.1], [0.9]],
            [[0.2], [0.8]],
        ],
        dtype=torch.double,
    )

    with patch(
        "robotorchan.optim.backends.botorch.botorch_optimize_acqf",
        return_value=(torch.tensor([[0.2], [0.8]]), torch.tensor([0.2, 0.8])),
    ) as mocked:
        optimize_acqf_botorch(
            _DummyAcquisition(),
            torch.tensor([[0.0], [1.0]], dtype=torch.double),
            q=2,
            num_restarts=2,
            raw_samples=16,
            batch_initial_conditions=initial,
            sequential=True,
        )

    kwargs = mocked.call_args.kwargs
    assert kwargs["batch_initial_conditions"] is initial
    assert kwargs["sequential"] is True


def test_botorch_backend_accepts_ic_generator_for_nonlinear_constraints() -> None:
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[..., 0] - 0.1, True),)
    )

    def ic_generator(**kwargs):
        return torch.full((2, 1, 1), 0.5)

    with patch(
        "robotorchan.optim.backends.botorch.botorch_optimize_acqf",
        return_value=(torch.tensor([[0.5]]), torch.tensor(0.5)),
    ) as mocked:
        optimize_acqf_botorch(
            _DummyAcquisition(),
            torch.tensor([[0.0], [1.0]]),
            q=1,
            num_restarts=2,
            raw_samples=8,
            constraints=constraints,
            ic_generator=ic_generator,
            ic_gen_kwargs={"custom_option": 3},
        )

    kwargs = mocked.call_args.kwargs
    assert kwargs["ic_generator"] is ic_generator
    assert kwargs["custom_option"] == 3
    assert kwargs["options"]["batch_limit"] == 1


def test_mixed_backend_accepts_ic_generator_for_nonlinear_constraints() -> None:
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[..., 0] - 0.1, True),)
    )

    def ic_generator(**kwargs):
        return torch.full((2, 1, 2), 0.5)

    with patch(
        "robotorchan.optim.backends.botorch.botorch_optimize_acqf_mixed",
        return_value=(torch.tensor([[0.5, 1.0]]), torch.tensor(1.5)),
    ) as mocked:
        optimize_acqf_mixed_botorch(
            _DummyAcquisition(),
            torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            q=1,
            num_restarts=2,
            fixed_features_list=[{1: 0.0}, {1: 1.0}],
            raw_samples=8,
            constraints=constraints,
            ic_generator=ic_generator,
            ic_gen_kwargs={"custom_option": 3},
        )

    kwargs = mocked.call_args.kwargs
    assert kwargs["ic_generator"] is ic_generator
    assert kwargs["ic_gen_kwargs"] == {"custom_option": 3}
    assert kwargs["options"]["batch_limit"] == 1


def test_botorch_backend_runs_sequential_nonlinear_with_ic_generator() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[0] - 0.1, True),)
    )
    generated_q: list[int] = []

    def ic_generator(*, q, num_restarts, **kwargs):
        generated_q.append(q)
        return torch.full(
            (num_restarts, q, 1),
            0.5,
            dtype=bounds.dtype,
            device=bounds.device,
        )

    candidate, value = optimize_acqf_botorch(
        _DummyAcquisition(),
        bounds,
        q=2,
        num_restarts=2,
        raw_samples=8,
        constraints=constraints,
        sequential=True,
        ic_generator=ic_generator,
    )

    assert candidate.shape == torch.Size([2, 1])
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert torch.all(candidate[:, 0] >= 0.1 - 1e-6)
    assert generated_q == [1, 1]


def test_botorch_backend_runs_sequential_with_intrapoint_linear_constraint() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0]),
                torch.tensor([1.0], dtype=torch.double),
                0.4,
            ),
        ),
    )

    candidate, value = optimize_acqf_botorch(
        _DummyAcquisition(),
        bounds,
        q=2,
        num_restarts=2,
        raw_samples=16,
        constraints=constraints,
        sequential=True,
    )

    assert candidate.shape == torch.Size([2, 1])
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert torch.all(candidate[:, 0] >= 0.4 - 1e-6)


@pytest.mark.parametrize("constraint_kind", ["inequality", "equality"])
def test_botorch_backend_rejects_sequential_interpoint_linear_constraint(
    constraint_kind: str,
) -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraint = (
        torch.tensor([[0, 0], [1, 0]]),
        torch.tensor([1.0, 1.0], dtype=torch.double),
        1.0,
    )
    constraints = CandidateConstraints(
        inequality_constraints=(constraint,) if constraint_kind == "inequality" else (),
        equality_constraints=(constraint,) if constraint_kind == "equality" else (),
    )

    with pytest.raises(UnsupportedError, match="inter-point"):
        optimize_acqf_botorch(
            _DummyAcquisition(),
            bounds,
            q=2,
            num_restarts=2,
            raw_samples=16,
            constraints=constraints,
            sequential=True,
        )


def test_botorch_backend_rejects_sequential_interpoint_nonlinear_constraint() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda X: X[:, 0].sum() - 1.0, False),)
    )

    def ic_generator(*, q, num_restarts, **kwargs):
        return torch.full(
            (num_restarts, q, 1),
            0.6,
            dtype=bounds.dtype,
            device=bounds.device,
        )

    with pytest.raises(UnsupportedError, match="inter-point"):
        optimize_acqf_botorch(
            _DummyAcquisition(),
            bounds,
            q=2,
            num_restarts=2,
            raw_samples=16,
            constraints=constraints,
            sequential=True,
            ic_generator=ic_generator,
        )


def test_botorch_backend_rejects_infeasible_nonlinear_initial_conditions() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: 0.4 - x[0], True),)
    )
    initial = torch.tensor([[[0.2]], [[0.8]]], dtype=torch.double)

    with pytest.raises(ValueError, match="restart 1 is infeasible"):
        optimize_acqf_botorch(
            _DummyAcquisition(),
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=None,
            constraints=constraints,
            batch_initial_conditions=initial,
        )


def test_botorch_backend_rejects_out_of_bounds_nonlinear_initial_conditions() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[0] + 1.0, True),)
    )
    initial = torch.tensor([[[0.2]], [[1.2]]], dtype=torch.double)

    with pytest.raises(ValueError, match="must lie within bounds"):
        optimize_acqf_botorch(
            _DummyAcquisition(),
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=None,
            constraints=constraints,
            batch_initial_conditions=initial,
        )


def test_botorch_backend_rejects_wrong_shape_nonlinear_initial_conditions() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[0] + 1.0, True),)
    )
    initial = torch.tensor([[[0.2]], [[0.4]]], dtype=torch.double)

    with pytest.raises(ValueError, match=r"\[num_restarts, 2, 1\]"):
        optimize_acqf_botorch(
            _DummyAcquisition(),
            bounds,
            q=2,
            num_restarts=2,
            raw_samples=None,
            constraints=constraints,
            batch_initial_conditions=initial,
        )


def test_botorch_backend_preserves_optimizer_failure_without_fallback() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[0] + 1.0, True),)
    )
    initial = torch.tensor([[[0.2]], [[0.4]]], dtype=torch.double)

    with (
        patch(
            "robotorchan.optim.backends.botorch.botorch_optimize_acqf",
            side_effect=RuntimeError("optimizer failed"),
        ) as mocked,
        pytest.raises(RuntimeError, match="optimizer failed"),
    ):
        optimize_acqf_botorch(
            _DummyAcquisition(),
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=None,
            constraints=constraints,
            batch_initial_conditions=initial,
        )

    mocked.assert_called_once()
