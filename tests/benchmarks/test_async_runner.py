"""Asynchronous benchmark simulation tests."""

import pytest
import torch

from robotorchan.benchmarks.async_runner import run_async_benchmark
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


def _registry() -> BenchmarkProblemRegistry:
    registry = BenchmarkProblemRegistry()
    registry.register(
        "linear",
        lambda: BenchmarkProblem(
            name="linear",
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            objective=lambda X: X,
            directions=("maximize",),
            variable_types=("continuous",),
        ),
    )
    return registry


def _candidates(problem, X, Y, pending, q, generator):
    del problem, X, Y, pending, generator
    return torch.tensor([[0.2]] * q, dtype=torch.double)


def test_async_completion_order_and_capacity() -> None:
    calls = iter((3.0, 1.0, 2.0, 1.0))
    config = BenchmarkExperimentConfig(
        problem="linear", strategy="async", initial_points=2, evaluation_budget=4, q=2
    )
    result = run_async_benchmark(
        config, _candidates, lambda X: next(calls), max_concurrency=2, registry=_registry()
    )[0]
    assert result.X.shape == (6, 1)
    assert result.max_concurrency == 2
    assert result.completion_order == (1, 0, 2, 3)
    assert result.simulated_makespan == 4.0
    assert [event.submitted_at for event in result.evaluations] == [0.0, 0.0, 1.0, 3.0]


def test_async_reproducible_seed_and_single_worker() -> None:
    config = BenchmarkExperimentConfig(
        problem="linear", strategy="async", seeds=(7,), initial_points=3,
        evaluation_budget=3, q=3
    )

    def random(problem, X, Y, pending, q, generator):
        del problem, Y, pending
        return torch.rand((q, X.shape[-1]), dtype=X.dtype, generator=generator)

    first = run_async_benchmark(
        config, random, lambda X: 1.0, max_concurrency=1, registry=_registry()
    )[0]
    second = run_async_benchmark(
        config, random, lambda X: 1.0, max_concurrency=1, registry=_registry()
    )[0]
    torch.testing.assert_close(first.X, second.X)
    assert first.completion_order == (0, 1, 2)
    assert first.simulated_makespan == 3.0
    assert first.max_concurrency == 1


@pytest.mark.parametrize("duration", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_duration_rejected(duration: float) -> None:
    config = BenchmarkExperimentConfig(
        problem="linear", strategy="async", initial_points=2, evaluation_budget=1
    )
    with pytest.raises(ValueError, match="durations"):
        run_async_benchmark(
            config, _candidates, lambda X: duration, max_concurrency=1,
            registry=_registry()
        )


def test_invalid_concurrency_and_candidate_shape() -> None:
    config = BenchmarkExperimentConfig(
        problem="linear", strategy="async", initial_points=2, evaluation_budget=1
    )
    with pytest.raises(ValueError, match="max_concurrency"):
        run_async_benchmark(
            config, _candidates, lambda X: 1.0, max_concurrency=0, registry=_registry()
        )
    with pytest.raises(ValueError, match="shape"):
        run_async_benchmark(
            config, lambda *args: torch.zeros(2, 1), lambda X: 1.0,
            max_concurrency=1, registry=_registry()
        )
