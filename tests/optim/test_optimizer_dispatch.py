"""Tests for the public optimizer dispatcher."""

from unittest.mock import patch

import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import MixedVariableSpace, optimize_acqf


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


def test_dispatch_forwards_variable_space_to_mixed_ga() -> None:
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 4.0, 2.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: [0.0, 1.0, 2.0]},
    )
    with patch("robotorchan.optim.dispatch.optimize_acqf_mixed_ga") as mocked:
        mocked.return_value = (torch.zeros(1, 3), torch.tensor(0.0))
        optimize_acqf(
            _SumAcquisition(),
            bounds,
            q=1,
            optimizer="ga",
            variable_space=variable_space,
        )
    assert mocked.call_args.kwargs["variable_space"] is variable_space


def test_dispatch_rejects_categorical_space_for_pso_before_backend() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(bounds, categorical_values={1: [0.0, 1.0, 2.0]})
    with patch("robotorchan.optim.dispatch.optimize_acqf_pso") as mocked:
        try:
            optimize_acqf(
                _SumAcquisition(),
                bounds,
                q=1,
                optimizer="pso",
                variable_space=variable_space,
            )
        except NotImplementedError as exc:
            assert "categorical variables" in str(exc)
        else:
            raise AssertionError("Categorical PSO should fail before backend execution.")
    mocked.assert_not_called()


def test_dispatch_rejects_structured_space_for_gradient_torch_optimizer() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 4.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(bounds, integer_dims=(1,))
    with patch("robotorchan.optim.dispatch.optimize_acqf_torch") as mocked:
        try:
            optimize_acqf(
                _SumAcquisition(),
                bounds,
                q=1,
                optimizer="torch_adam",
                variable_space=variable_space,
            )
        except NotImplementedError as exc:
            assert "integer variables" in str(exc)
        else:
            raise AssertionError("Integer Adam should fail before backend execution.")
    mocked.assert_not_called()
