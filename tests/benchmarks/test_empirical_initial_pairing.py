"""Initial-condition fairness tests for cross-strategy empirical runs."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.initial_pairing import validate_initial_pairing
from robotorchan.benchmarks.observation_noise import SeededObservationNoise
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import BenchmarkTrajectory, random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def _registry() -> BenchmarkProblemRegistry:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    return registry


def _noise(truth: Tensor, generator: torch.Generator) -> Tensor:
    return torch.randn(truth.shape, generator=generator, dtype=truth.dtype) * 0.1


def _run(strategy: str, policy: str = "paired") -> tuple[BenchmarkTrajectory, ...]:
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy=strategy,
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=5,
        q=3,
    )
    return run_benchmark(
        config,
        random_candidates,
        registry=_registry(),
        restrict_candidate_problem=True,
        independent_rng_streams=True,
        observation_noise=SeededObservationNoise(_noise, policy=policy),
    )


def test_paired_initial_conditions_pass() -> None:
    result = validate_initial_pairing({"random": _run("random"), "sobol": _run("sobol")})
    assert result.fully_paired
    assert result.seeds == (0, 1)


def test_independent_noise_can_be_explicitly_allowed() -> None:
    result = validate_initial_pairing(
        {"random": _run("random", "independent"), "sobol": _run("sobol", "independent")},
        require_observation_pairing=False,
    )
    assert result.design_paired
    assert result.truth_paired
    assert not result.observation_paired


def test_initial_design_mismatch_is_rejected() -> None:
    first = _run("random")
    second = _run("sobol")
    altered = second[0].X.clone()
    altered[0, 0] += 0.01
    changed = (replace(second[0], X=altered), second[1])
    with pytest.raises(ValueError, match="Initial designs"):
        validate_initial_pairing({"random": first, "sobol": changed})


def test_duplicate_seed_is_rejected() -> None:
    first = _run("random")
    with pytest.raises(ValueError, match="duplicate seeds"):
        validate_initial_pairing({"random": first, "sobol": (first[0], first[0])})


def test_initial_observation_mismatch_is_rejected() -> None:
    first = _run("random")
    second = _run("sobol")
    changed_y = second[0].Y_observed.clone()
    changed_y[0, 0] += 0.01
    changed = (replace(second[0], Y_observed=changed_y), second[1])
    with pytest.raises(ValueError, match="Initial observations"):
        validate_initial_pairing({"random": first, "sobol": changed})
