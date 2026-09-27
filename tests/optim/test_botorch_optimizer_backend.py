from unittest.mock import patch

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
