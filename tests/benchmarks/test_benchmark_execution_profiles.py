"""Regression tests for immutable benchmark execution profiles."""

import pytest
import torch

from robotorchan.benchmarks.execution_profiles import (
    BenchmarkExecutionProfile,
    get_execution_profile,
    list_execution_profiles,
)


def test_builtin_profile_budgets() -> None:
    assert list_execution_profiles() == ("extended", "smoke", "standard")
    for name in list_execution_profiles():
        profile = get_execution_profile(name)
        config = profile.make_config("branin", "random")
        assert config.initial_points == profile.initial_points
        assert config.evaluation_budget == profile.evaluation_budget
        assert config.seeds == profile.seeds
        assert config.q == profile.q
        assert config.dtype == "float64"
        assert config.device == "cpu"


def test_precision_override_does_not_mutate_builtin() -> None:
    profile = get_execution_profile("smoke", dtype="float32")
    assert profile.make_config("branin", "random").torch_dtype == torch.float32
    assert get_execution_profile("smoke").dtype == "float64"


def test_invalid_profile_name_and_parameters() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        get_execution_profile("missing")
    with pytest.raises(ValueError, match="dtype"):
        get_execution_profile("smoke", dtype="float16")
    with pytest.raises(ValueError, match="unique"):
        BenchmarkExecutionProfile(
            name="smoke", seeds=(0, 0), initial_points=2, evaluation_budget=2, q=1
        )


def test_cuda_availability_checked(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        get_execution_profile("smoke", device="cuda")
