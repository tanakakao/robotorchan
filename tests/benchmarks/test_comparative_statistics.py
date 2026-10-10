"""Tests for paired benchmark statistical estimates."""

from dataclasses import replace

import pytest
import torch

from robotorchan.benchmarks.comparative_constrained import ConstrainedComparison
from robotorchan.benchmarks.comparative_statistics import paired_statistics


def _comparison(metric: str = "feasible_regret") -> ConstrainedComparison:
    baseline = torch.tensor([[4.0, 3.0], [5.0, 4.0], [6.0, 5.0]])
    challenger = baseline - torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    return ConstrainedComparison(
        seeds=(0, 1, 2),
        evaluations=torch.tensor([0, 1]),
        metric=metric,
        scores_by_method={"random": baseline, "candidate": challenger},
        feasibility_by_method={},
        mean_score_by_method={},
        standard_error_by_method={},
        mean_feasibility_by_method={},
    )


def test_paired_regret_improvement_and_reproducibility() -> None:
    comparison = _comparison()
    first = paired_statistics(
        comparison, reference="random", challenger="candidate", bootstrap_samples=100
    )
    second = paired_statistics(
        comparison, reference="random", challenger="candidate", bootstrap_samples=100
    )
    torch.testing.assert_close(first.mean_improvement, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(first.win_rate, torch.ones(2))
    torch.testing.assert_close(first.confidence_lower, second.confidence_lower)
    assert (first.confidence_lower <= first.mean_improvement).all()
    assert (first.confidence_upper >= first.mean_improvement).all()


def test_hypervolume_reverses_direction() -> None:
    comparison = _comparison("feasible_hypervolume")
    result = paired_statistics(comparison, reference="random", challenger="candidate")
    torch.testing.assert_close(result.mean_improvement, torch.tensor([-2.0, -3.0]))
    torch.testing.assert_close(result.win_rate, torch.zeros(2))


def test_identical_methods_have_tie_rate_one() -> None:
    comparison = _comparison()
    comparison.scores_by_method["candidate"] = comparison.scores_by_method["random"].clone()
    result = paired_statistics(comparison, reference="random", challenger="candidate")
    torch.testing.assert_close(result.tie_rate, torch.ones(2))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confidence_level": 1.0}, "confidence_level"),
        ({"bootstrap_samples": 0}, "bootstrap_samples"),
        ({"tie_tolerance": -1.0}, "tie_tolerance"),
    ],
)
def test_invalid_settings(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        paired_statistics(_comparison(), reference="random", challenger="candidate", **kwargs)


def test_single_seed_rejected() -> None:
    comparison = _comparison()
    scores = {name: values[:1] for name, values in comparison.scores_by_method.items()}
    comparison = replace(comparison, seeds=(0,), scores_by_method=scores)
    with pytest.raises(ValueError, match="At least two"):
        paired_statistics(comparison, reference="random", challenger="candidate")


def test_nonfinite_scores_rejected() -> None:
    comparison = _comparison()
    comparison.scores_by_method["candidate"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        paired_statistics(comparison, reference="random", challenger="candidate")
