"""Compatibility contracts for robotorchan and BoTorch risk measures."""

import torch
from botorch.acquisition.risk_measures import (
    CVaR as BoTorchCVaR,
)
from botorch.acquisition.risk_measures import (
    Expectation as BoTorchExpectation,
)
from botorch.acquisition.risk_measures import (
    VaR as BoTorchVaR,
)
from botorch.acquisition.risk_measures import (
    WorstCase as BoTorchWorstCase,
)

from robotorchan.objectives import CVaR, Expectation, VaR, WorstCase


def _scenario_values() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
        ],
        dtype=torch.double,
    )


def _as_botorch_samples(values: torch.Tensor) -> torch.Tensor:
    # BoTorch risk objectives receive n_w scenarios flattened into q * n_w.
    return values.unsqueeze(0).unsqueeze(-1)


def test_expectation_matches_botorch_for_equivalent_scenario_layout() -> None:
    values = _scenario_values()
    native = BoTorchExpectation(n_w=values.shape[-1])

    actual = Expectation()(values)
    expected = native(_as_botorch_samples(values)).squeeze(0).squeeze(-1)

    torch.testing.assert_close(actual, expected)


def test_worst_case_matches_botorch_for_maximization() -> None:
    values = _scenario_values()
    native = BoTorchWorstCase(n_w=values.shape[-1])

    actual = WorstCase()(values)
    expected = native(_as_botorch_samples(values)).squeeze(0)

    torch.testing.assert_close(actual, expected)


def test_var_matches_botorch_lower_tail_order_statistic_on_empirical_quantile() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.double)
    alpha = 0.8
    native = BoTorchVaR(alpha=alpha, n_w=values.shape[-1])

    botorch_value = native(_as_botorch_samples(values)).squeeze()
    robotorchan_value = VaR(alpha=alpha)(values).squeeze()

    assert botorch_value.item() == 2.0
    assert robotorchan_value.item() == 1.8


def test_cvar_can_match_botorch_for_an_exact_empirical_tail() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.double)
    alpha = 0.7
    native = BoTorchCVaR(alpha=alpha, n_w=values.shape[-1])

    botorch_value = native(_as_botorch_samples(values)).squeeze()
    robotorchan_value = CVaR(alpha=alpha)(values).squeeze()

    torch.testing.assert_close(botorch_value, robotorchan_value)


def test_botorch_risk_measure_preserves_candidate_axis_after_scenario_reduction() -> None:
    sample_shape = torch.Size([7])
    q = 3
    n_w = 5
    samples = torch.randn(*sample_shape, q * n_w, 1, dtype=torch.double)
    risk_measure = BoTorchExpectation(n_w=n_w)

    result = risk_measure(samples)

    assert result.shape == torch.Size([7, q])
