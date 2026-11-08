from __future__ import annotations

from unittest.mock import patch

import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim.backends.torch import optimize_acqf_torch
from robotorchan.optim.constraints import CandidateConstraints


class _DummyAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -(x**2).sum(dim=(-1, -2))


def test_torch_backend_delegates_through_botorch_optimize_acqf() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acq = _DummyAcquisition()
    candidates = torch.tensor([[0.25]], dtype=torch.double)
    value = torch.tensor(-0.0625, dtype=torch.double)

    with patch(
        "robotorchan.optim.backends.torch.botorch_optimize_acqf",
        return_value=(candidates, value),
    ) as mocked:
        result = optimize_acqf_torch(
            acq, bounds, q=1, num_restarts=4, raw_samples=16, optimizer="adam"
        )

    assert result == (candidates, value)
    kwargs = mocked.call_args.kwargs
    assert kwargs["acq_function"] is acq
    assert kwargs["q"] == 1
    assert kwargs["num_restarts"] == 4
    assert kwargs["raw_samples"] == 16
    assert kwargs["gen_candidates"].keywords["optimizer"] is torch.optim.Adam


def test_torch_backend_accepts_torch_optimizer_class() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    with patch(
        "robotorchan.optim.backends.torch.botorch_optimize_acqf",
        return_value=(torch.zeros(1, 1), torch.tensor(0.0)),
    ) as mocked:
        optimize_acqf_torch(
            _DummyAcquisition(),
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
            optimizer=torch.optim.AdamW,
        )

    assert mocked.call_args.kwargs["gen_candidates"].keywords["optimizer"] is torch.optim.AdamW


def test_torch_backend_linear_constraint_modes() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([1.0]), 0.5),)
    )
    candidate = torch.tensor([[0.25]], dtype=torch.double)
    with patch(
        "robotorchan.optim.backends.torch.botorch_optimize_acqf",
        return_value=(candidate, torch.tensor(-0.0625, dtype=torch.double)),
    ):
        try:
            optimize_acqf_torch(
                _DummyAcquisition(),
                bounds,
                q=1,
                num_restarts=2,
                raw_samples=8,
                constraints=constraints,
            )
        except ValueError as error:
            assert "no feasible candidate" in str(error)
        else:
            raise AssertionError("Strict mode must reject infeasible candidates.")
        returned, value = optimize_acqf_torch(
            _DummyAcquisition(),
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
            constraints=constraints,
            constraint_mode="penalty",
        )
    torch.testing.assert_close(returned, candidate)
    torch.testing.assert_close(value.reshape(()), torch.tensor(-0.0625, dtype=torch.double))


def test_torch_backend_linear_equality_modes() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        equality_constraints=((torch.tensor([0]), torch.tensor([1.0]), 0.5),)
    )
    candidate = torch.tensor([[0.25]], dtype=torch.double)
    with patch(
        "robotorchan.optim.backends.torch.botorch_optimize_acqf",
        return_value=(candidate, torch.tensor(-0.0625, dtype=torch.double)),
    ) as mocked:
        with torch.no_grad():
            try:
                optimize_acqf_torch(
                    _DummyAcquisition(),
                    bounds,
                    q=1,
                    num_restarts=2,
                    raw_samples=8,
                    constraints=constraints,
                )
            except ValueError as error:
                assert "no feasible candidate" in str(error)
            else:
                raise AssertionError("Strict mode must reject infeasible equalities.")
            returned, value = optimize_acqf_torch(
                _DummyAcquisition(),
                bounds,
                q=1,
                num_restarts=2,
                raw_samples=8,
                constraints=constraints,
                constraint_mode="penalty",
            )
    assert mocked.call_args.kwargs["acq_function"] is not None
    torch.testing.assert_close(returned, candidate)
    torch.testing.assert_close(value.reshape(()), torch.tensor(-0.0625, dtype=torch.double))


def test_torch_backend_rejects_unknown_optimizer_name() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    try:
        optimize_acqf_torch(
            _DummyAcquisition(),
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
            optimizer="not-an-optimizer",  # type: ignore[arg-type]
        )
    except ValueError as error:
        assert "Unknown torch optimizer" in str(error)
    else:
        raise AssertionError("Expected an unknown optimizer name to be rejected.")
