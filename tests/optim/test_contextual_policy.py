"""Focused empirical context-policy evaluation tests."""

import pytest
import torch

from robotorchan.optim.contextual_policy import (
    expected_policy_value,
    monte_carlo_policy_value,
)


def test_exact_policy_values_and_ranking() -> None:
    weights = torch.tensor([1.0, 3.0], dtype=torch.double)
    outcomes = torch.tensor([[2.0, 6.0], [5.0, 1.0]], dtype=torch.double)
    values = expected_policy_value(outcomes, weights)
    torch.testing.assert_close(values, torch.tensor([5.0, 2.0], dtype=torch.double))
    assert values.argmax().item() == 0
    torch.testing.assert_close(expected_policy_value(outcomes[0], weights), values[0])


def test_monte_carlo_policy_values_reproducible_and_close() -> None:
    weights = torch.tensor([1.0, 3.0], dtype=torch.double)
    outcomes = torch.tensor([[2.0, 6.0], [5.0, 1.0]], dtype=torch.double)
    generator = torch.Generator().manual_seed(42)
    estimate = monte_carlo_policy_value(outcomes, weights, n_samples=20000, generator=generator)
    expected = expected_policy_value(outcomes, weights)
    torch.testing.assert_close(estimate, expected, atol=0.08, rtol=0)
    again = monte_carlo_policy_value(
        outcomes, weights, n_samples=20000, generator=torch.Generator().manual_seed(42)
    )
    torch.testing.assert_close(estimate, again)


@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([-1.0, 2.0], dtype=torch.double),
        torch.tensor([0.0, 0.0], dtype=torch.double),
        torch.tensor([float("nan"), 1.0], dtype=torch.double),
    ],
)
def test_invalid_context_weights_rejected(weights: torch.Tensor) -> None:
    outcomes = torch.tensor([1.0, 2.0], dtype=torch.double)
    with pytest.raises(ValueError):
        expected_policy_value(outcomes, weights)


def test_invalid_outcomes_and_sample_count_rejected() -> None:
    weights = torch.tensor([0.5, 0.5], dtype=torch.double)
    with pytest.raises(ValueError, match="context dimension"):
        expected_policy_value(torch.tensor([1.0], dtype=torch.double), weights)
    with pytest.raises(ValueError, match="dtype and device"):
        expected_policy_value(torch.tensor([1.0, 2.0], dtype=torch.float), weights)
    with pytest.raises(ValueError, match="n_samples"):
        monte_carlo_policy_value(torch.tensor([1.0, 2.0], dtype=torch.double), weights, n_samples=0)
