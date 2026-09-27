"""Tests for the public optimizer dispatcher."""

from unittest.mock import patch

import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import optimize_acqf


class _SumAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.sum(dim=(-1, -2))


def test_dispatch_defaults_to_botorch_contract() -> None:
    bounds = torch.tensor([[0.0], [1.0]])
    expected = (torch.tensor([[0.5]]), torch.tensor(0.5))
    with patch("robotorchan.optim.dispatch.optimize_acqf_botorch", return_value=expected) as mocked:
        result = optimize_acqf(
            _SumAcquisition(),
            bounds,
            q=1,
            num_restarts=4,
            raw_samples=32,
        )
    assert result == expected
    assert mocked.call_args.args[2] == 1
    assert mocked.call_args.args[3:5] == (4, 32)


def test_dispatch_runs_ga_with_backend_options() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidate, value = optimize_acqf(
        _SumAcquisition(),
        bounds,
        q=1,
        optimizer="ga",
        seed=4,
        optimizer_options={"population_size": 16, "generations": 3},
    )
    assert candidate.shape == (1, 1)
    assert value.ndim == 0


def test_dispatch_rejects_unknown_optimizer() -> None:
    bounds = torch.tensor([[0.0], [1.0]])
    try:
        optimize_acqf(_SumAcquisition(), bounds, q=1, optimizer="unknown")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "Unknown optimizer" in str(exc)
    else:
        raise AssertionError("Unknown optimizer should fail explicitly.")
