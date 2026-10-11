"""Phase 2 reproducibility regression checks for the empirical benchmark cycle."""

from __future__ import annotations

import torch

from robotorchan.benchmarks import BenchmarkExperimentConfig
from robotorchan.benchmarks.comparative_reproducibility import (
    verify_random_baseline_reproducibility,
)


def test_random_baseline_repeated_run_and_json_roundtrip() -> None:
    """Require identical numerical histories on repeated seeded CPU runs."""
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(0, 1),
        initial_points=10,
        evaluation_budget=10,
        q=1,
        dtype="float64",
        device="cpu",
    )
    result = verify_random_baseline_reproducibility(config)

    assert result.seeds == config.seeds
    assert result.evaluation_budget == config.evaluation_budget
    assert result.repeat_equal
    assert result.persistence_equal


def test_random_baseline_rejects_unsupported_device_and_strategy() -> None:
    """Keep the baseline verification scope explicit."""
    import pytest

    for strategy, device in (("qEI", "cpu"), ("random", "cuda")):
        config = BenchmarkExperimentConfig(
            problem="branin",
            strategy=strategy,
            device=device,
        )
        with pytest.raises(ValueError):
            verify_random_baseline_reproducibility(config)


def test_float64_is_explicit_in_empirical_baseline() -> None:
    """Avoid silently changing the empirical benchmark precision."""
    config = BenchmarkExperimentConfig(problem="branin", strategy="random", dtype="float64")
    assert config.torch_dtype == torch.float64
    assert BenchmarkExperimentConfig.from_dict(config.to_dict()) == config
