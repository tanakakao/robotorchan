"""Tests for canonical mixed-variable benchmark scenarios."""

import torch

from robotorchan.benchmarks.mixed_optimization import (
    MixedInteractionAcquisition,
    mixed_benchmark_scenarios,
)


def test_mixed_benchmark_suite_covers_required_variable_structures() -> None:
    scenarios = mixed_benchmark_scenarios()

    assert set(scenarios) == {
        "continuous_integer",
        "continuous_categorical",
        "full_mixed",
        "constrained_full_mixed",
    }
    assert scenarios["continuous_integer"].variable_space.integer_dims == (1,)
    assert scenarios["continuous_categorical"].variable_space.categorical_dims == (1,)
    assert scenarios["full_mixed"].variable_space.integer_dims == (1,)
    assert scenarios["full_mixed"].variable_space.categorical_dims == (2,)
    assert scenarios["constrained_full_mixed"].constraints is not None


def test_mixed_interaction_acquisition_has_category_specific_optima() -> None:
    acquisition = MixedInteractionAcquisition()
    X = torch.tensor(
        [
            [[0.2, 2.0, 0.0]],
            [[0.8, 2.0, 1.0]],
            [[0.5, 2.0, 2.0]],
            [[0.8, 0.0, 0.0]],
        ],
        dtype=torch.double,
    )

    values = acquisition(X)

    assert torch.allclose(values[:3], torch.zeros(3, dtype=torch.double))
    assert values[3] < values[0]
