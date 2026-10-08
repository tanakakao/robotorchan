"""Benchmark experiment configuration tests."""

import json

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _problem() -> BenchmarkProblem:
    return BenchmarkProblem(
        name="quadratic",
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        objective=lambda X: -(X - 0.5).square(),
        directions=("maximize",),
        variable_types=("continuous",),
    )


def test_roundtrip_and_dtype() -> None:
    config = BenchmarkExperimentConfig(
        problem="quadratic",
        strategy="sobol",
        seeds=(0, 2),
        evaluation_budget=9,
        q=4,
    )
    record = json.loads(json.dumps(config.to_dict()))
    assert BenchmarkExperimentConfig.from_dict(record) == config
    assert config.torch_dtype == torch.float64
    assert config.torch_device == torch.device("cpu")


def test_problem_resolution() -> None:
    registry = BenchmarkProblemRegistry()
    registry.register("quadratic", _problem)
    config = BenchmarkExperimentConfig(problem="quadratic", strategy="random")
    assert config.resolve_problem(registry).name == "quadratic"
    with pytest.raises(KeyError, match="Unknown benchmark"):
        BenchmarkExperimentConfig(problem="missing", strategy="random").resolve_problem(registry)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"seeds": ()}, "seeds"),
        ({"seeds": (1, 1)}, "unique"),
        ({"seeds": (-1,)}, "nonnegative"),
        ({"seeds": (True,)}, "nonnegative"),
        ({"initial_points": 0}, "positive"),
        ({"evaluation_budget": 0}, "positive"),
        ({"q": 0}, "positive"),
        ({"dtype": "float16"}, "dtype"),
        ({"device": "mps"}, "device"),
        ({"problem": ""}, "problem"),
        ({"strategy": ""}, "strategy"),
    ],
)
def test_invalid_config(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BenchmarkExperimentConfig(problem="quadratic", strategy="sobol", **changes)


def test_reject_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        BenchmarkExperimentConfig.from_dict(
            {"problem": "quadratic", "strategy": "sobol", "extra": 1}
        )
