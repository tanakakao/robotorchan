from unittest.mock import patch

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim.backends.botorch import (
    optimize_acqf_botorch,
    optimize_acqf_mixed_botorch,
)
from robotorchan.optim.constraints import CandidateConstraints


class _DummyAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

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
    constraints = CandidateConstraints(
        inequality_constraints=((indices, coefficients, 1.2),)
    )

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
