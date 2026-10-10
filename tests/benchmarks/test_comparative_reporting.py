"""Phase 22 Markdown report regression tests."""

import pytest
import torch

from robotorchan.benchmarks.comparative_reporting import (
    report_single_objective,
)
from robotorchan.benchmarks.comparative_single_objective import SingleObjectiveComparison


def _comparison(mean: torch.Tensor, error: torch.Tensor) -> SingleObjectiveComparison:
    return SingleObjectiveComparison(
        seeds=(0, 1),
        q=1,
        evaluations=torch.tensor([0, 1]),
        regret_by_method={},
        mean_regret_by_method={"random": mean},
        standard_error_by_method={"random": error},
    )


def test_report_includes_metric_direction_and_uncertainty() -> None:
    result = report_single_objective(
        _comparison(torch.tensor([3.0, 2.0]), torch.tensor([0.3, 0.2]))
    )
    assert result.direction == "minimize"
    assert result.final_evaluations == 1
    assert "Seeds: 2 (0, 1)" in result.markdown
    assert "| random | 2 | 0.2 |" in result.markdown
    assert "not a significance test" in result.markdown


def test_report_handles_pre_feasibility_sentinel() -> None:
    result = report_single_objective(
        _comparison(
            torch.tensor([float("inf"), float("inf")]),
            torch.tensor([float("nan"), float("nan")]),
        )
    )
    assert "| random | +inf | undefined |" in result.markdown


def test_finite_score_requires_defined_error() -> None:
    with pytest.raises(ValueError, match="defined standard errors"):
        report_single_objective(
            _comparison(torch.tensor([1.0, 2.0]), torch.tensor([0.1, float("nan")]))
        )
