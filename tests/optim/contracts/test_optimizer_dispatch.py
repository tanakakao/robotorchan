"""Tests for the public optimizer dispatcher."""

from unittest.mock import patch

import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import CandidateConstraints, MixedVariableSpace, optimize_acqf


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


def test_dispatch_rejects_invalid_common_bounds_before_backend() -> None:
    invalid_bounds = (
        torch.tensor([[0.0], [float("nan")]]),
        torch.tensor([[0.0], [float("inf")]]),
        torch.tensor([[1.0], [0.0]]),
    )
    for bounds in invalid_bounds:
        with patch("robotorchan.optim.dispatch.optimize_acqf_botorch") as mocked:
            try:
                optimize_acqf(_SumAcquisition(), bounds, q=1)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid bounds should fail before backend execution.")
        mocked.assert_not_called()


def test_dispatch_rejects_nonpositive_q_before_backend() -> None:
    bounds = torch.tensor([[0.0], [1.0]])
    with patch("robotorchan.optim.dispatch.optimize_acqf_botorch") as mocked:
        try:
            optimize_acqf(_SumAcquisition(), bounds, q=0)
        except ValueError as exc:
            assert "q must be at least 1" in str(exc)
        else:
            raise AssertionError("Non-positive q should fail before backend execution.")
    mocked.assert_not_called()


def test_dispatch_rejects_variable_space_bounds_mismatch_before_backend() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(
        torch.tensor([[0.0, 0.0], [1.0, 3.0]], dtype=torch.double),
        integer_dims=(1,),
    )
    with patch("robotorchan.optim.dispatch.optimize_acqf_mixed_ga") as mocked:
        try:
            optimize_acqf(
                _SumAcquisition(),
                bounds,
                q=1,
                optimizer="ga",
                variable_space=variable_space,
            )
        except ValueError as exc:
            assert "variable_space bounds must match bounds" in str(exc)
        else:
            raise AssertionError("Mismatched variable-space bounds should fail before backend.")
    mocked.assert_not_called()


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


def test_dispatch_forwards_botorch_initial_condition_generator() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    def ic_generator(**kwargs):
        return torch.full((2, 1, 1), 0.5, dtype=bounds.dtype)

    with patch("robotorchan.optim.dispatch.optimize_acqf_botorch") as mocked:
        mocked.return_value = (torch.tensor([[0.5]]), torch.tensor(0.5))
        optimize_acqf(
            _SumAcquisition(),
            bounds,
            q=1,
            ic_generator=ic_generator,
            ic_gen_kwargs={"custom_option": 3},
        )

    assert mocked.call_args.kwargs["ic_generator"] is ic_generator
    assert mocked.call_args.kwargs["ic_gen_kwargs"] == {"custom_option": 3}


def test_dispatch_runs_nonlinear_constraint_with_ic_generator() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: 0.4 - x[0], True),)
    )

    def ic_generator(*, q, num_restarts, **kwargs):
        return torch.full(
            (num_restarts, q, 1),
            0.2,
            dtype=bounds.dtype,
            device=bounds.device,
        )

    candidate, value = optimize_acqf(
        _SumAcquisition(),
        bounds,
        q=1,
        num_restarts=2,
        raw_samples=8,
        constraints=constraints,
        ic_generator=ic_generator,
    )

    assert candidate.shape == torch.Size([1, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert candidate[0, 0] <= 0.4 + 1e-6


def test_dispatch_runs_nonlinear_constraint_with_explicit_initial_conditions() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: 0.6 - x[0], True),)
    )
    initial = torch.tensor([[[0.2]], [[0.4]]], dtype=torch.double)

    candidate, value = optimize_acqf(
        _SumAcquisition(),
        bounds,
        q=1,
        num_restarts=2,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial,
    )

    assert candidate.shape == torch.Size([1, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert candidate[0, 0] <= 0.6 + 1e-6


def test_dispatch_rejects_ic_generator_for_non_botorch_backend() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    def ic_generator(**kwargs):
        return torch.full((2, 1, 1), 0.5, dtype=bounds.dtype)

    try:
        optimize_acqf(
            _SumAcquisition(),
            bounds,
            q=1,
            optimizer="ga",
            ic_generator=ic_generator,
        )
    except ValueError as exc:
        assert "ic_generator is only used by optimizer='botorch'" in str(exc)
    else:
        raise AssertionError("Non-BoTorch backend should reject ic_generator.")
