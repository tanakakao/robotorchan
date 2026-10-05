"""Tests for robust classification probability transforms."""

import pytest
import torch

from robotorchan.models.classification import (
    ClassificationProbabilityRiskType,
    RobustProbabilityTransform,
    robust_class_probability,
)


@pytest.fixture
def scenario_probabilities() -> torch.Tensor:
    positive = torch.tensor(
        [[0.9, 0.7, 0.2, 0.6], [0.4, 0.8, 0.5, 0.3]],
        dtype=torch.double,
    )
    return torch.stack((1.0 - positive, positive), dim=-1)


def test_expected_probability_preserves_class_axis(
    scenario_probabilities: torch.Tensor,
) -> None:
    actual = RobustProbabilityTransform("expected")(scenario_probabilities)
    expected = scenario_probabilities.mean(dim=-2)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 2)


def test_worst_case_is_classwise_lower_probability(
    scenario_probabilities: torch.Tensor,
) -> None:
    actual = RobustProbabilityTransform("worst_case")(scenario_probabilities)

    torch.testing.assert_close(actual, scenario_probabilities.min(dim=-2).values)


def test_var_and_quantile_are_lower_tail_probability(
    scenario_probabilities: torch.Tensor,
) -> None:
    alpha = 0.75
    expected = torch.quantile(scenario_probabilities, 1.0 - alpha, dim=-2)

    var = RobustProbabilityTransform("var", alpha=alpha)(scenario_probabilities)
    quantile = RobustProbabilityTransform("quantile", alpha=alpha)(scenario_probabilities)

    torch.testing.assert_close(var, expected)
    torch.testing.assert_close(quantile, expected)


def test_cvar_is_mean_of_lower_tail_probabilities(
    scenario_probabilities: torch.Tensor,
) -> None:
    transform = RobustProbabilityTransform(
        ClassificationProbabilityRiskType.CVAR,
        alpha=0.75,
    )
    actual = transform(scenario_probabilities)

    moved = scenario_probabilities.movedim(-2, -1)
    threshold = torch.quantile(moved, 0.25, dim=-1, keepdim=True)
    mask = moved <= threshold
    expected = (moved * mask).sum(dim=-1) / mask.sum(dim=-1)

    torch.testing.assert_close(actual, expected)


def test_class_helper_returns_only_requested_robust_probability(
    scenario_probabilities: torch.Tensor,
) -> None:
    actual = robust_class_probability(
        scenario_probabilities,
        class_index=1,
        risk_type="worst_case",
    )

    torch.testing.assert_close(actual, scenario_probabilities[..., 1].min(dim=-1).values)


def test_probability_transform_keeps_autograd() -> None:
    logits = torch.tensor(
        [[0.2, -0.4], [0.5, 0.1], [-0.3, 0.7]],
        dtype=torch.double,
        requires_grad=True,
    )
    probabilities = torch.softmax(logits, dim=-1)

    robust = RobustProbabilityTransform("expected")(probabilities)
    robust[..., 1].sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_scenario_axis_must_not_be_class_axis(
    scenario_probabilities: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="class dimension"):
        RobustProbabilityTransform("expected", scenario_dim=-1)(scenario_probabilities)


def test_invalid_probability_simplex_is_rejected() -> None:
    probabilities = torch.tensor([[[0.8, 0.4], [0.3, 0.7]]], dtype=torch.double)

    with pytest.raises(ValueError, match="sum to one"):
        RobustProbabilityTransform("expected")(probabilities)
