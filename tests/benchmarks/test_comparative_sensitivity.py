"""Phase 19 sensitivity experiment and aggregation tests."""

import pytest
import torch

from robotorchan.benchmarks.comparative_sensitivity import (
    sensitivity_configs,
    summarize_sensitivity,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig


def test_sensitivity_configs_change_only_selected_axis() -> None:
    base = BenchmarkExperimentConfig(problem="branin", strategy="random")
    configs = sensitivity_configs(base, axis="q", values=[1, 3, 5])
    assert [config.q for config in configs] == [1, 3, 5]
    assert all(config.initial_points == base.initial_points for config in configs)
    assert all(config.evaluation_budget == base.evaluation_budget for config in configs)


def test_budget_sensitivity_uses_only_shared_checkpoints() -> None:
    result = summarize_sensitivity(
        axis="evaluation_budget",
        scores={
            2: torch.tensor([[4.0, 3.0, 2.0], [5.0, 4.0, 3.0]]),
            4: torch.tensor([[4.0, 2.0, 1.0], [5.0, 3.0, 2.0]]),
        },
        evaluations={2: torch.tensor([0, 1, 2]), 4: torch.tensor([0, 2, 4])},
        seeds={2: (0, 1), 4: (0, 1)},
        metric="simple_regret",
    )
    assert result.evaluations.tolist() == [0, 2]
    torch.testing.assert_close(result.mean_by_value[2], torch.tensor([4.5, 2.5]))
    torch.testing.assert_close(result.mean_by_value[4], torch.tensor([4.5, 2.5]))


def test_different_seeds_rejected() -> None:
    with pytest.raises(ValueError, match="Seed ordering"):
        summarize_sensitivity(
            axis="q",
            scores={1: torch.ones(2, 2), 2: torch.ones(2, 2)},
            evaluations={1: torch.tensor([0, 2]), 2: torch.tensor([0, 2])},
            seeds={1: (0, 1), 2: (1, 0)},
            metric="regret",
        )


def test_nonfinite_scores_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        summarize_sensitivity(
            axis="q",
            scores={1: torch.tensor([[float("nan")]])},
            evaluations={1: torch.tensor([0])},
            seeds={1: (0,)},
            metric="regret",
        )


def test_duplicate_sweep_values_rejected() -> None:
    base = BenchmarkExperimentConfig(problem="branin", strategy="random")
    with pytest.raises(ValueError, match="unique"):
        sensitivity_configs(base, axis="q", values=[1, 1])


def test_single_seed_has_zero_standard_error() -> None:
    result = summarize_sensitivity(
        axis="initial_points",
        scores={4: torch.tensor([[1.0, 0.5]])},
        evaluations={4: torch.tensor([0, 2])},
        seeds={4: (0,)},
        metric="regret",
    )
    torch.testing.assert_close(result.standard_error_by_value[4], torch.zeros(2))
